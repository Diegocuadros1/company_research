import requests
import json

def get_sec_files(company: str, cik_number: str):
    url = f"https://data.sec.gov/submissions/CIK{cik_number}.json"
    headers = {
        "User-Agent": "Diego Cuadros (cuadrosda21@gmail.com)",
        "Accept": "application/json"
    }
    response = requests.get(url, headers=headers)
    return response.json()

def fetch_html(url: str) -> str:
    r = requests.get(
        url,
        headers={
            "User-Agent": "Diego Cuadros (cuadrosda21@gmail.com)",
            "Accept": "text/html,application/xhtml+xml,*/*",
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.text