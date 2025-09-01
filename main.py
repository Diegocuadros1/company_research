from api.sec_files import get_sec_files, filter_sec_files
from api.prompts import initiatives, overview, job_postings, five_year, collecting_material
from run_prompts import run_research
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def main():
    company = "Cognizant Technology Solutions Corp"
    cik_number = "0001058290"

    print(f"Searching for {company} with CIK number {cik_number}")

    sec_files = get_sec_files(cik_number)

    filtered_sec_files = filter_sec_files(sec_files)

    count = 0
    for item in filtered_sec_files:
        count += 1
        print(item["filingUrl"])

    print(count, "SEC Files found")
    # print("Running Prompts")

    overview_prompt = overview(company)
    initiatives_prompt = initiatives(company)
    five_year_prompt = five_year(company)
    job_postings_prompt = job_postings(company)
    collecting_material_prompt = collecting_material(company)

    prompts =  [
        overview_prompt,
        initiatives_prompt,
        five_year_prompt,
        job_postings_prompt,
        collecting_material_prompt
    ]

    start = time.time()

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(run_research, p): idx for idx, p in enumerate(prompts)}
        results = [None] * len(prompts)
        for fut in as_completed(futures):
            idx = futures[fut]
            results[idx] = fut.result()


    print("TOTAL TIME TAKEN: ", int((time.time() - start) // 60), " minutes and ", int((time.time() - start) % 60), " seconds.")
    return

if __name__ == "__main__":
    main()