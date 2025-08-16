# file: sec_ingest_pipeline.py
import os
import re
import json
import uuid
from datetime import datetime
from typing import List, Dict, Iterable, Tuple, Optional
from api.sec_files import fetch_html

import requests
from bs4 import BeautifulSoup
import pandas as pd
from tenacity import retry, wait_exponential, stop_after_attempt
import tiktoken

from io import StringIO

from dotenv import load_dotenv
load_dotenv()

from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI

from bs4 import XMLParsedAsHTMLWarning
import warnings
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


USE_OPENAI = True
OPENAI_MODEL = "text-embedding-3-small"  # adjust as you like (returns 1536 dims)
MAX_MODEL_TOKENS = 8192                 # per-input limit for the model
DEFAULT_CHUNK_TOKENS = 700              # your desired target size
SAFETY_MARGIN = 256                     # keep well under limit


from pinecone import Pinecone, ServerlessSpec

PINECONE_ENV = "us-east-1"  # serverless region; change if needed
INDEX_NAME = "company-research"  # choose your index name


def filter_sec_files(input_data):
    filings = input_data["filings"]["recent"]
    name = input_data["name"]
    ticker = input_data["tickers"]
    cikPath = int(str(input_data["cik"]))  # remove leading zeros

    now = datetime.now()
    try:
        fiveYearsAgo = now.replace(year=now.year - 5)
    except ValueError:
        fiveYearsAgo = now.replace(month=2, day=28, year=now.year - 5)

    typesWanted = ["10-K", "10-Q", "8-K"]
    results = []

    for i in range(len(filings["accessionNumber"])):
        formType = filings["form"][i]
        filingDate_str = filings["filingDate"][i]
        filingDate = datetime.strptime(filingDate_str, "%Y-%m-%d")

        if formType in typesWanted and filingDate >= fiveYearsAgo:
            accessionNoNoDashes = filings["accessionNumber"][i].replace("-", "")
            primaryDoc = filings["primaryDocument"][i]
            filingUrl = f"https://www.sec.gov/Archives/edgar/data/{cikPath}/{accessionNoNoDashes}/{primaryDoc}"

            results.append({
                "name": name,
                "ticker": ticker,
                "formType": formType,
                "filingDate": filingDate_str,
                "filingUrl": filingUrl
            })

    return results

def extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "xml")
    for t in soup(["script", "style", "noscript"]):
        t.decompose()
    # Try to reduce boilerplate: collapse multiple newlines
    text = soup.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text

def extract_tables(html: str) -> List[pd.DataFrame]:
    try:
        return pd.read_html(StringIO(html), flavor="lxml")  # fast way to get <table> as DataFrames
    except ValueError:
        return []

def extract_ixbrl_facts(html: str) -> List[Dict]:
    soup = BeautifulSoup(html, "xml")
    facts = []
    for tag_name in ("ix:nonFraction", "ix:nonNumeric"):
        for node in soup.find_all(tag_name):
            facts.append({
                "type": tag_name,
                "name": node.get("name"),
                "contextRef": node.get("contextref"),
                "unitRef": node.get("unitref"),
                "decimals": node.get("decimals"),
                "value": node.get_text(strip=True),
            })
    return facts

# Extract tables and ixbrl facts
def tables_to_markdown(dfs: List[pd.DataFrame]) -> List[str]:
    out = []
    for df in dfs:
        # Fill NaNs for cleaner markdown
        md = df.fillna("").to_markdown(index=False)
        out.append(md)
    return out

def ixbrl_to_lines(facts: List[Dict]) -> str:
    # One fact per line, tab-separated; easy to search & chunk
    lines = []
    for f in facts:
        pieces = [
            f.get("type") or "",
            f.get("name") or "",
            f.get("contextRef") or "",
            f.get("unitRef") or "",
            f.get("decimals") or "",
            (f.get("value") or "").replace("\n", " ").strip(),
        ]
        lines.append("\t".join(pieces))
    return "\n".join(lines)

# =============== CHUNKING (TOKEN-AWARE) =================

def get_token_encoder(model: str = "cl100k_base"):
    # cl100k_base is good for most modern OpenAI models
    return tiktoken.get_encoding(model)

def count_tokens(text: str, enc=None) -> int:
    enc = enc or get_token_encoder()
    return len(enc.encode(text))

def chunk_text(
    text: str,
    max_tokens: int = 700,
    overlap: int = 100,
    enc=None
) -> List[str]:
    """
    Simple token-aware sliding window chunker.
    Splits by paragraphs first, then packs into ~max_tokens chunks with overlap.
    """
    enc = enc or get_token_encoder()
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    buf = []
    buf_tokens = 0

    def flush():
        nonlocal buf, buf_tokens
        if not buf:
            return
        chunk = "\n\n".join(buf).strip()
        if chunk:
            chunks.append(chunk)
        buf = []
        buf_tokens = 0

    for para in paras:
        t = count_tokens(para, enc)
        if t > max_tokens:
            # If a single paragraph is too long, split by sentences.
            sentences = re.split(r"(?<=[\.!\?])\s+", para)
            for s in sentences:
                st = count_tokens(s, enc)
                if buf_tokens + st > max_tokens and buf:
                    flush()
                buf.append(s)
                buf_tokens += st
        else:
            if buf_tokens + t > max_tokens and buf:
                flush()
            buf.append(para)
            buf_tokens += t

    flush()

    # Add overlap by merging tail/head tokens between neighbors
    if overlap > 0 and len(chunks) > 1:
        overlapped = []
        for i, ch in enumerate(chunks):
            if i == 0:
                overlapped.append(ch)
                continue
            # prepend tail of previous chunk (approximate by tokens)
            prev = chunks[i - 1]
            prev_tokens = enc.encode(prev)
            ch_tokens = enc.encode(ch)
            overlap_text = enc.decode(prev_tokens[-overlap:]) if len(prev_tokens) > overlap else prev
            merged = overlap_text + "\n\n" + enc.decode(ch_tokens)
            overlapped.append(merged)
        chunks = overlapped

    return chunks

# =============== EMBEDDINGS =================

def embed_texts(texts: List[str]) -> Tuple[List[List[float]], int]:
    """
    Returns (embeddings, dimension)
    """
  
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    resp = client.embeddings.create(model=OPENAI_MODEL, input=texts)
    embs = [d.embedding for d in resp.data]
    dim = len(embs[0]) if embs else 0
    return embs, dim
    

# =============== PINECONE =================

def ensure_pinecone_index(index_name: str, dimension: int):
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    # Create if not exists (serverless)
    if index_name not in [i["name"] for i in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV),
            metric="cosine",
        )
    return pc.Index(index_name)

def upsert_chunks(
    index,
    namespace: str,
    chunks: List[str],
    embeddings: List[List[float]],
    base_metadata: Dict
):
    vectors = []
    for i, (text, emb) in enumerate(zip(chunks, embeddings)):
        vid = str(uuid.uuid4())
        md = {**base_metadata, "chunk_index": i, "length": len(text)}
        vectors.append({"id": vid, "values": emb, "metadata": {"text": text, **md}})

    # Batch upsert to avoid payload limits
    BATCH = 100
    for start in range(0, len(vectors), BATCH):
        index.upsert(vectors=vectors[start:start+BATCH], namespace=namespace)

# =============== ORCHESTRATION =================

def ingest_sec_html(url: str, company: Optional[str] = None, ticker: Optional[str] = None):
    html = fetch_html(url)

    text = extract_text(html)
    dfs = extract_tables(html)
    tables_md = tables_to_markdown(dfs)
    ixbrl = extract_ixbrl_facts(html)
    ixbrl_text = ixbrl_to_lines(ixbrl) if ixbrl else ""

    enc = get_token_encoder()

    # Build document parts with simple headers for traceability
    docs: List[Tuple[str, Dict]] = []

    if text:
        docs.append((
            f"# TEXT\n\n{url}\n\n{text}",
            {"section": "text"}
        ))

    for ti, table_md in enumerate(tables_md):
        docs.append((
            f"# TABLE {ti}\n\n{url}\n\n{table_md}",
            {"section": "table", "table_index": ti}
        ))

    if ixbrl_text:
        docs.append((
            f"# IXBRL FACTS\n\n{url}\n\n{ixbrl_text}",
            {"section": "ixbrl"}
        ))

    # Chunk, embed, and upsert each doc part
    # Use a stable namespace per company/ticker if provided; else by host
    namespace = (ticker or company or "sec").lower()

    base_meta_common = {
        "source_url": url,
        "company": company,
        "ticker": ticker,
        "ingested_at": datetime.utcnow().isoformat() + "Z",
    }

    # Determine embedding dimension from first batch
    pc_client = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = None
    dim_cache = None

    for part_text, part_meta in docs:
        chunks = chunk_text(part_text, max_tokens=700, overlap=80, enc=enc)
        if not chunks:
            continue

        embeddings, dim = embed_texts(chunks)
        dim_cache = dim_cache or dim
        if index is None:
            index = ensure_pinecone_index(INDEX_NAME, dim_cache)

        upsert_chunks(
            index,
            namespace=namespace,
            chunks=chunks,
            embeddings=embeddings,
            base_metadata={**base_meta_common, **part_meta},
        )

    return {
        "status": "ok",
        "url": url,
        "parts_ingested": len(docs),
        "namespace": namespace,
        "index": INDEX_NAME,
        "dimension": dim_cache,
    }
