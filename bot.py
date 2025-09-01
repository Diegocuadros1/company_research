import os
from dotenv import load_dotenv
import discord
from discord.ext import commands
from api.sec_files import get_sec_files, filter_sec_files
from api.prompts import initiatives, overview, job_postings, five_year, collecting_material
from run_prompts import run_research
from concurrent.futures import ThreadPoolExecutor
import asyncio
import time

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

executor = ThreadPoolExecutor(max_workers=5)


@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')


@bot.command(name="hello")
async def hello(ctx):
    await ctx.send("Hello!")


@bot.command(name="sec")
async def sec(ctx, cik: str):
    await ctx.send(f"Running SEC search on {cik}!")
    sec_files = get_sec_files(cik)
    filtered_sec_files = filter_sec_files(sec_files)

    count = 0
    links = []
    for item in filtered_sec_files:
        count += 1
        links.append(item["filingUrl"])
        # await ctx.send(item["filingUrl"])

    links = '\n'.join(map(str, links))
    for i in range(0, len(links), 1900):
        await ctx.send(links[i:i + 1900])

    await ctx.send(f"{count}, SEC Files found")


@bot.command(name="research")
async def research(ctx, company: str):
    await ctx.send(f"Running private research on {company}!")

    overview_prompt = overview(company)
    initiatives_prompt = initiatives(company)
    five_year_prompt = five_year(company)
    job_postings_prompt = job_postings(company)
    collecting_material_prompt = collecting_material(company)

    prompts = [
        overview_prompt, initiatives_prompt, five_year_prompt,
        job_postings_prompt, collecting_material_prompt
    ]

    start = time.time()

    async def task(p):
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(executor, run_research, p)
        for i in range(0, len(result), 1900):
            await ctx.send(result[i:i + 1900])

    tasks = [asyncio.create_task(task(p)) for p in prompts]
    for coro in asyncio.as_completed(tasks):
        await coro

    await ctx.send(
        f"TOTAL TIME TAKEN: {int((time.time() - start) // 60)} minutes and {int((time.time() - start) % 60)} seconds."
    )
    return


@bot.command(name="ping")
async def ping(ctx, name: str):
    await ctx.send(f"Hello {name}!")


bot.run(TOKEN)
