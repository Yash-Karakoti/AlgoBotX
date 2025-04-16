import discord
from discord import app_commands
from discord.ext import commands
import os
import requests
from groq import Groq
from dotenv import load_dotenv
from rag import vector_db
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

# Configuration
EMBEDDINGS = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
MAX_RESPONSE_LENGTH = 1900  # Discord message limit

# Initialize models with quantization
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="auto",
    load_in_8bit=True
)

# Initialize clients
groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
bot = commands.Bot(command_prefix="!", intents=discord.Intents.default())

@bot.event
async def on_ready():
    print(f'Bot {bot.user} connected!')
    try:
        await bot.tree.sync()
        print("Commands synced")
    except Exception as e:
        print(f"Command sync error: {e}")

def get_rag_context(query: str, k: int = 3) -> str:
    """Retrieve relevant context from documents"""
    docs = vector_db.similarity_search(query, k=k)
    return "\n".join([d.page_content for d in docs])

@bot.tree.command(name="ask", description="Ask about Algorand blockchain")
async def ask_algorand(interaction: discord.Interaction, question: str):
    """Handle technical questions with RAG + SLM"""
    try:
        # Retrieve context
        context = get_rag_context(question)
        
        if not context:
            raise ValueError("No relevant documents found")
        
        # Generate answer
        prompt = f"""You are an Algorand expert assistant. Use this context:
        {context}
        
        Question: {question}
        Answer:"""
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=400)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean and send response
        clean_answer = answer.replace(prompt, "").strip()
        await send_chunked_response(interaction, clean_answer)
        
    except Exception as e:
        print(f"RAG failed: {e}")
        # Fallback to Groq
        response = groq.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": question}],
            max_tokens=800
        )
        await send_chunked_response(interaction, response.choices[0].message.content)

@bot.tree.command(name="faucet", description="Get TestNet ALGO tokens")
@app_commands.describe(wallet_address="Your Algorand TestNet address")
async def send_faucet(interaction: discord.Interaction, wallet_address: str):
    """Handle faucet requests"""
    try:
        headers = {"x-api-key": os.getenv("PURESTAKE_API_KEY")}
        response = requests.post(
            "https://testnet-algorand.api.purestake.io/ps2/v2/faucet",
            headers=headers,
            json={"address": wallet_address, "amount": 1000000}
        )
        
        if response.status_code == 200:
            tx_id = response.json().get("txID")
            explorer_link = f"https://testnet.algoexplorer.io/tx/{tx_id}"
            await interaction.response.send_message(
                f"✅ 1 ALGO sent to `{wallet_address}`\n"
                f"Transaction: {explorer_link}"
            )
        else:
            await interaction.response.send_message(
                f"❌ Error: {response.text} (Code {response.status_code})"
            )
            
    except Exception as e:
        await interaction.response.send_message(f"⚠️ Faucet error: {str(e)}")

async def send_chunked_response(interaction, text):
    """Split long responses into Discord-friendly chunks"""
    for i in range(0, len(text), MAX_RESPONSE_LENGTH):
        chunk = text[i:i+MAX_RESPONSE_LENGTH]
        if i == 0:
            await interaction.response.send_message(chunk)
        else:
            await interaction.followup.send(chunk)

if __name__ == "__main__":
    bot.run(os.getenv("SECRET_KEY"))