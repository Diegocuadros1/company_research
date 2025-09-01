import time 
from concurrent.futures import ThreadPoolExecutor, as_completed

def count(num, ):
    for i in range(10):
        time.sleep(1)
        print(i)
    
    return "done", num

prompts=[1,2,3,4,5,6]

def main():
    print("starting")
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(count, p): idx for idx, p in enumerate(prompts)}
        results = [None] * len(prompts)
        for fut in as_completed(futures):
            idx = futures[fut]
            results[idx] = fut.result()

    for i, r in enumerate(results, 1):
        print(f"\n=== Answer {i} ===\n{r}")


    

if __name__ == "__main__":
    main()

