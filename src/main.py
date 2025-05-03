import discord
import os
from discord import app_commands
from groq import Groq
from discord.ext import commands
from dotenv import load_dotenv
from algosdk import encoding
from algokit_utils import AlgorandClient

load_dotenv()

# Initialize clients
groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
bot = commands.Bot(command_prefix="!", intents=discord.Intents.all())
algorand_client = AlgorandClient.default_localnet()
testnet_dispenser = algorand_client.client.get_testnet_dispenser(auth_token=os.getenv("ALGOKIT_DISPENSER_ACCESS_TOKEN"))

def is_valid_algorand_address(address: str) -> bool:
    """Validate Algorand address format using SDK"""
    try:
        return encoding.is_valid_address(address)
    except:
        return False

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
    await interaction.response.defer()

    if not is_valid_algorand_address(wallet_address):
        # Use followup after defer
        return await interaction.followup.send("❌ Invalid Algorand address format")

    try:
        result = testnet_dispenser.fund(
            address=wallet_address,
            amount=1_000_000,
            asset_id=0
        )
        tx_id = getattr(result, "tx_id", None)

        if tx_id:
            explorer_link = f"https://testnet.algoexplorer.io/tx/{tx_id}"
            await interaction.followup.send(
                f"✅ 1 TestNet ALGO sent to `{wallet_address}`!\n"
                f"View transaction: {explorer_link}"
            )
        else:
            await interaction.followup.send(
                f"✅ 1 TestNet ALGO sent to `{wallet_address}`!\n"
                f"(No transaction ID returned; please check your wallet in a minute.)"
            )

    except Exception as e:
        # Still use followup
        await interaction.followup.send(f"⚠️ Failed: {str(e)[:200]}")

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    if bot.user in message.mentions:
        response = groq.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": message.content}],
            max_tokens=800,
            temperature=0.7,
            top_p=1,
            stream=True,
        )
        
        reply_buffer = []
        current_length = 0
        
        for chunk in response:
            content = chunk.choices[0].delta.content or ""
            if current_length + len(content) > 1900:
                await message.channel.send("".join(reply_buffer))
                reply_buffer = []
                current_length = 0
            
            reply_buffer.append(content)
            current_length += len(content)
        
        if reply_buffer:
            full_reply = "".join(reply_buffer)
            for i in range(0, len(full_reply), 1900):
                await message.channel.send(full_reply[i:i+1900])

    await bot.process_commands(message)

if __name__ == "__main__":
    bot.run(os.getenv("SECRET_KEY"))