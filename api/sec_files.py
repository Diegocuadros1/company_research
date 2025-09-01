import requests
from datetime import datetime

def get_sec_files( cik_number: str):
    url = f"https://data.sec.gov/submissions/CIK{cik_number}.json"
    headers = {
        "User-Agent": "Diego Cuadros (cuadrosda21@gmail.com)",
        "Accept": "application/json"
    }
    response = requests.get(url, headers=headers)
    return response.json()

def filter_sec_files(input_data):
    filings = input_data["filings"]["recent"]
    name = input_data["name"]
    ticker = input_data["tickers"]
    cikPath = int(str(input_data["cik"]))  # remove leading zeros

    now = datetime.now()
    # Five years back, handling Feb 29 edge case
    try:
        fiveYearsAgo = now.replace(year=now.year - 5)
    except ValueError:
        fiveYearsAgo = now.replace(month=2, day=28, year=now.year - 5)

    results = []

    for i in range(len(filings["accessionNumber"])):
        formType = filings["form"][i]
        filingDate_str = filings["filingDate"][i]
        filingDate = datetime.strptime(filingDate_str, "%Y-%m-%d")

        include = False
        if formType == "10-Q":
            # Only include 10-Qs from the current year
            include = (filingDate.year == now.year)
        elif formType in ("10-K", "8-K"):
            # Include 10-Ks and 8-Ks from the last 5 years
            include = (filingDate >= fiveYearsAgo)

        if include:
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
