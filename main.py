from api.sec_files import get_sec_files, fetch_html
from helpers.filter import filter_sec_files, ingest_sec_html
import asyncio

def main():
    print("Please add the company you want to search for:")
    company = "Cushman & Wakefield"
    cik_number = "0001628369"
    # company = input("Company: ")
    # cik_number = input("CIK Number: ")

    print(f"Searching for {company} with CIK number {cik_number}")

    sec_files = get_sec_files(company, cik_number)
    ticker = sec_files["tickers"][0]

    filtered_sec_files = filter_sec_files(sec_files)

    for item in filtered_sec_files:
        results = ingest_sec_html(item["filingUrl"], company=company, ticker=ticker)
        print(results)
        break
    return 0


if __name__ == "__main__":
    main()