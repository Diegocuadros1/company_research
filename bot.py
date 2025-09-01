import os 
import asyncio
from dotenv import load_dotenv
import discord
from discord.ext import commands

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')

intents = discord.Intents.default()
bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')

@bot.command(name="hello")
async def hello(ctx):
    await ctx.send("Hello!")

@bot.command(name="ping")
async def ping(ctx, name:str):
    await ctx.send(f"Hello {name}!")


bot.run(TOKEN)