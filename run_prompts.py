import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse
from dotenv import load_dotenv
import time

import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from textwrap import dedent

load_dotenv()

OPENAI_MODEL = os.getenv("OPENAI_GPT5_MODEL", "gpt-5")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# >>> NEW: Tavily config
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
TAVILY_SEARCH_ENDPOINT = "https://api.tavily.com/search"   # POST
# Optional: you can also use Extract at https://api.tavily.com/extract

@dataclass
class SourceDoc:
    idx: int
    title: str
    url: str
    snippet: str
    text: str

def _visible_text_from_html(html: str, max_chars: int = 8000) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "form"]):
        tag.decompose()
    for attr in ["aria-hidden", "hidden"]:
        for el in soup.find_all(attrs={attr: True}):
            try:
                if attr == "aria-hidden" and el.get(attr) == "true":
                    el.decompose()
                if attr == "hidden":
                    el.decompose()
            except Exception:
                continue
    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars] + " …"
    return text

def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""

# >>> REPLACEMENT: Tavily search
def tavily_search(
    query: str,
    *,
    topic: Optional[str] = None,          # e.g., "news" for news-only
    time_range: Optional[str] = None,     # "day"|"week"|"month"|"year"
    max_results: int = 8,
    depth: str = "advanced",              # "basic" | "advanced"
    include_raw_content: bool = True,
    timeout: float = 30.0
) -> Dict[str, Any]:
    """
    Perform a Tavily search.
    Returns JSON with keys: results (list of {title,url,content,raw_content?,score,...}), answer?, etc.
    """
    if not TAVILY_API_KEY:
        raise RuntimeError("TAVILY_API_KEY is not set in environment")

    payload = {
        "query": query,
        "search_depth": depth,
        "include_answer": False,
        "include_raw_content": include_raw_content,
        "include_images": False,
        "max_results": max_results,
    }
    if topic:
        payload["topic"] = None              # "news" narrows to news sources
    if time_range:
        payload["time_range"] = time_range    # day|week|month|year

    headers = {
        "Authorization": f"Bearer {TAVILY_API_KEY}",
        "Content-Type": "application/json",
        "User-Agent": "research-ai/1.0 (+https://api.tavily.com)"
    }
    resp = requests.post(TAVILY_SEARCH_ENDPOINT, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()

def fetch_url(url: str, timeout: float = 20.0) -> Tuple[Optional[str], Optional[str]]:
    """
    Fallback downloader if Tavily didn't return raw_content.
    """
    try:
        headers = {"User-Agent": "research-ai/1.0 (+https://api.tavily.com)"}
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        html = r.text
        soup = BeautifulSoup(html, "html.parser")
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else url
        text = _visible_text_from_html(html)
        return title, text
    except Exception as e:
        logging.debug(f"Failed to fetch {url}: {e}")
        return None, None

def gpt_json(client: OpenAI, prompt: str, schema_hint: str) -> Dict[str, Any]:
    try:
        resp = client.responses.create(
            model=OPENAI_MODEL,
            instructions="You are a precise research planner. Return ONLY valid JSON per the user's schema.",
            input=[{"role": "user", "content": [{"type": "input_text", "text": f"Schema:\n{schema_hint}\n\nTask:\n{prompt}"}]}],
            response_format={"type": "json_object"},
        )
        text = resp.output_text
        return json.loads(text)
    except Exception as e:
        logging.debug(f"JSON mode failed, falling back: {e}")
        resp = client.responses.create(
            model=OPENAI_MODEL,
            instructions="Return JSON only. Do not include any prose.",
            input=f"{schema_hint}\n\nTask:\n{prompt}",
        )
        text = resp.output_text
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            return {}
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}

def propose_queries(client: OpenAI, user_prompt: str, news: bool) -> Dict[str, Any]:
    """
    Returns: {"queries": [...], "freshness": Optional[str], "notes": str}
    """
    # >>> FIX: add freshness to schema (your old schema required it but didn't define it)
    schema = {
        "type": "object",
        "properties": {
            "queries": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "maxItems": 6,
            },
            "freshness": {
                "type": ["string", "null"],
                "enum": ["day", "week", "month", "year", None]
            },
            "notes": {"type": "string"}
        },
        "required": ["queries"]
    }
    schema_hint = json.dumps(schema, indent=2)
    plan = gpt_json(
        client,
        prompt=(
            "Given the user's research request, propose 3–6 highly targeted web search queries. "
            f"Prefer {'news' if news else 'regular'} search operators if useful (site:, filetype:, intitle:, etc.). "
            "Include a short 'notes' string if there are special considerations. "
            "If time sensitivity matters, set freshness to day|week|month|year."
            f"\n\nUser prompt:\n{user_prompt}"
        ),
        schema_hint=schema_hint,
    )
    queries = plan.get("queries") or [user_prompt]
    notes = plan.get("notes", "")
    freshness = plan.get("freshness")
    return {"queries": queries, "freshness": freshness, "notes": notes}

def dedupe_results(items: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    seen_urls, seen_domains, deduped = set(), set(), []
    for it in items:
        url = it.get("link") or it.get("url")
        if not url:
            continue
        dom = _domain(url)
        if url in seen_urls or dom in seen_domains:
            continue
        seen_urls.add(url)
        seen_domains.add(dom)
        deduped.append(it)
        if len(deduped) >= limit:
            break
    return deduped

def gather_sources(user_prompt: str, *, max_sources: int, verbose: bool, client: OpenAI) -> List[SourceDoc]:
    """
    Use Tavily instead of SerpAPI.
    """
    news = False
    plan = propose_queries(client, user_prompt, news=news)
    freshness = plan.get("freshness")  # "day"|"week"|"month"|"year"|None

    if verbose:
        print(f"[plan] queries={plan['queries']} freshness={freshness} notes={plan.get('notes','')}", file=sys.stderr)

    # Aggregate results from Tavily
    results_pool: List[Dict[str, Any]] = []
    for q in plan["queries"]:
        try:
            data = tavily_search(
                q,
                topic=None,
                time_range=None,
                max_results=max_sources,
                depth="advanced",
                include_raw_content=True
            )
            # Tavily returns results under "results"
            if "results" in data:
                results_pool.extend(data["results"])
        except Exception as e:
            if verbose:
                print(f"[warn] Tavily search failed for '{q}': {e}", file=sys.stderr)

    if verbose:
        print(f"[info] gathered {len(results_pool)} raw results", file=sys.stderr)

    # Deduplicate and build SourceDoc list
    chosen = dedupe_results(results_pool, limit=max_sources)
    sources: List[SourceDoc] = []
    for idx, item in enumerate(chosen, start=1):
        url = item.get("url")
        title = item.get("title") or url
        snippet = item.get("content") or ""
        raw = item.get("raw_content") or ""  # present when include_raw_content=True

        text_blob = (raw or snippet or "")
        f_title, fetched_text = (None, None)
        if not text_blob and url:
            f_title, fetched_text = fetch_url(url)  # fallback when raw not provided

        final_title = f_title or title or (url or f"Result {idx}")
        final_text = (text_blob or fetched_text or "")
        sources.append(
            SourceDoc(
                idx=idx,
                title=final_title,
                url=url or "",
                snippet=str(snippet)[:500],
                text=final_text
            )
        )
        if verbose and url:
            print(f"[fetch] {idx}. {_domain(url)} — {final_title[:80]}", file=sys.stderr)

    return sources

def build_context(user_prompt: str, sources: List[SourceDoc]) -> str:
    parts = [f"USER PROMPT:\n{user_prompt}\n", "SOURCES (numbered):"]
    for s in sources:
        parts.append(
            f"[{s.idx}] {s.title}\nURL: {s.url}\nEXCERPT: {s.snippet[:300]}\nTEXT: {s.text[:2000]}\n"
        )
    return "\n".join(parts)

def synthesize_answer(client: OpenAI, user_prompt: str, sources: List[SourceDoc]) -> str:
    context = build_context(user_prompt, sources)
    sys_prompt = dedent("""
        You are a meticulous research AI. Using ONLY the numbered sources provided,
        write ONE synthesized answer to the user's prompt.
        Rules:
        - Use inline numeric citations like [1], [2] that map to the provided sources.
        - If multiple sources support a claim, you may cite multiple like [1][3].
        - Be concise but complete; avoid fluff.
        - If something is uncertain, say so plainly.
        - End with a short "Sources" section listing [n] URL per line.
        - Output plain UTF-8 text only.
    """).strip()

    mapping_lines = ["Source Map:"] + [f"[{s.idx}] {s.url}" for s in sources]
    mapping = "\n".join(mapping_lines)

    resp = client.responses.create(
        model=OPENAI_MODEL,
        instructions=sys_prompt,
        input=[{"role": "user", "content": [{"type": "input_text", "text": context},
                                            {"type": "input_text", "text": mapping}]}],
    )
    return resp.output_text.strip()

def run(user_prompt: str, *,  verbose: bool, client) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set in environment")
    if not TAVILY_API_KEY:
        raise RuntimeError("TAVILY_API_KEY is not set in environment")

    sources = gather_sources(
        user_prompt,
        verbose=verbose,
        max_sources=20,
        client=client
    )
    if not sources:
        raise RuntimeError("No sources found; try relaxing filters (e.g., remove topic='news' or widen freshness)")

    answer = synthesize_answer(client, user_prompt, sources)
    return answer

def run_research(user_prompt: str) -> None:
    start = time.time()
    client = OpenAI(api_key=OPENAI_API_KEY)
    try:
        answer = run(
            user_prompt,
            verbose=True,
            client=client
        )
        print(answer)
        # print(f"Time taken for search: {time.time() - start}")
        print("\n\n\n")
        print("======================================================================================================================================")
        print("\n\n\n")
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print("\n\nTime taken before error: ", + (time.time() - start), "\n\n")
        sys.exit(1)
