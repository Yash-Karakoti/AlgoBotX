import discord
from discord import app_commands
from discord.ext import commands
import os
import requests
from groq import Groq
from dotenv import load_dotenv
from algosdk import encoding

load_dotenv()

# Initialize clients
groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
bot = commands.Bot(command_prefix="!", intents=discord.Intents.all())

def is_valid_algorand_address(address: str) -> bool:
    """Validate Algorand address format using SDK"""
    return encoding.is_valid_address(address)

@bot.event
async def on_ready():
    print(f'Logged on as {bot.user}!')
    try:
        await bot.tree.sync()
        print("Slash commands synced")
    except Exception as e:
        print(f"Error syncing commands: {e}")

@bot.tree.command(name="faucet", description="Get TestNet ALGO tokens")
@app_commands.describe(wallet_address="Your Algorand TestNet address")
async def faucet(interaction: discord.Interaction, wallet_address: str):
    """Official Algorand TestNet Dispenser API Implementation"""
    try:
        if not is_valid_algorand_address(wallet_address):
            await interaction.response.send_message("❌ Invalid Algorand address format")
            return

        headers = {"X-API-Key": os.getenv("ALGORAND_API_KEY")}
        response = requests.post(
            "https://testnet-api.algonode.cloud/account/fund",
            headers=headers,
            json={"address": wallet_address, "amount": 1000000}
        )

        if response.status_code == 200:
            tx_data = response.json()
            explorer_link = f"https://testnet.algoexplorer.io/tx/{tx_data['txID']}"
            await interaction.response.send_message(
                f"✅ 1 ALGO sent to `{wallet_address}`\n"
                f"Transaction: {explorer_link}"
            )
        else:
            error_msg = response.text.replace('\n', ' ')[:150]
            await interaction.response.send_message(
                f"❌ Error ({response.status_code}): {error_msg}"
            )
            
    except Exception as e:
        await interaction.response.send_message(f"⚠️ Failed: {str(e)[:200]}")

