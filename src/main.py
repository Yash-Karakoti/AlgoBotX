
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import discord
import os
from discord import app_commands, ui
from groq import Groq
from discord.ext import commands
from dotenv import load_dotenv
from algosdk import encoding
from algokit_utils import AlgorandClient
from quiz_manager import quiz_manager
from quiz_data import QUIZ_QUESTIONS
from langchain.prompts import SystemMessagePromptTemplate, PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.schema import HumanMessage
import glob
import re
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import requests
import io
import tempfile
import aiohttp
import cv2
import numpy as np

load_dotenv()

def is_code_query(query):
    """
    Simple detection for code-related queries
    """
    code_keywords = [
        'debug', 'fix', 'error', 'bug', 'exception', 'traceback',
        'syntax error', 'runtime error', 'compile error', 'crash',
        'not working', 'broken code', 'code review', 'optimize code',
        'refactor', 'improve code', 'code issue', 'programming problem',
        'function', 'variable', 'class', 'import', 'def', 'if', 'else',
        'for', 'while', 'try', 'except', 'python', 'javascript', 'java'
    ]
    
    query_lower = query.lower()
    
    # Check for code-related keywords
    if any(keyword in query_lower for keyword in code_keywords):
        return True
    
    # Check for code blocks (
    if '' in query or '`' in query:
        return True
    
    return False

# Enhanced document loading and processing
def load_documents():
    """Load documents from docs folder with multiple file types"""
    documents = []
    
    # Get the project root directory (parent of src)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  
    docs_path = os.path.join(project_root, "docs")
    
    print(f"Looking for documents in: {docs_path}")
    
    if not os.path.exists(docs_path):
        print(f"❌ {docs_path} folder does not exist")
        return documents
    
    # List all files in docs folder
    all_files = os.listdir(docs_path)
    print(f"Files found in {docs_path}: {all_files}")
    
    # Load different file types from docs folder
    file_patterns = [
        os.path.join(docs_path, "*.txt"),
        os.path.join(docs_path, "*.md"), 
        os.path.join(docs_path, "*.pdf")
    ]
    
    for pattern in file_patterns:
        files = glob.glob(pattern)
        print(f"Found {len(files)} files matching {pattern}")
        for file_path in files:
            try:
                if file_path.endswith('.pdf'):
                    from langchain_community.document_loaders import PyPDFLoader
                    loader = PyPDFLoader(file_path)
                else:
                    from langchain_community.document_loaders import TextLoader
                    loader = TextLoader(file_path)
                file_docs = loader.load()
                documents.extend(file_docs)
                print(f"✅ Successfully loaded: {os.path.basename(file_path)} ({len(file_docs)} chunks)")
            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")
    
    print(f"Total documents loaded: {len(documents)}")
    return documents

# Enhanced document processing with better chunking
try:
    documents = load_documents()
    if documents:
        text_splitter = CharacterTextSplitter(
            chunk_size=500,  # Increased chunk size for better context
            chunk_overlap=100,  # Better overlap for continuity
            separator="\n\n"  # Split on paragraphs first
        )
        texts = text_splitter.split_documents(documents)
        
        # Enhanced embeddings with better model
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Create vector store with better search parameters
        vectorstore = Chroma.from_documents(
            texts, 
            embeddings,
            collection_name="algorand_docs"
        )
        
        # Create retriever with optimized parameters
        retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 5,  # Retrieve more documents for better coverage
                "score_threshold": 0.3  # Lower threshold for more inclusive results
            }
        )
        
        # Enhanced prompt template for better responses
        prompt_template = """You are an expert Algorand assistant. Use the following context from the official documentation to answer the question accurately and comprehensively.

Context from Algorand Documentation:
{context}

Question: {question}

Instructions:
- Answer based ONLY on the provided context
- If the context doesn't contain enough information, say "Based on the available documentation, I don't have enough information to fully answer this question."
- Be specific and detailed in your response
- Include relevant technical details when appropriate
- Format your response clearly with bullet points or sections when helpful

Answer:"""

        prompt = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question"]
        )
        
        print(f"✅ Enhanced document processing initialized with {len(documents)} documents and {len(texts)} chunks")
        
    else:
        print("❌ No documents found in docs folder")
        retriever = None
        prompt = None
        
except Exception as e:
    print(f"❌ Error initializing document processing: {e}")
    retriever = None
    prompt = None

# Enhanced document retrieval function
def get_document_answer(question_text):
    """Get clean answer from documents without file names or headers"""
    if not retriever or not prompt:
        return None, []
    
    try:
        print(f"Searching documents for 🔍: {question_text}")
        
        # Retrieve relevant documents
        docs = retriever.get_relevant_documents(question_text)
        print(f"📄 Retrieved {len(docs)} relevant documents")
        
        if not docs:
            print("No relevant documents found ❌")
            return None, []
        
        # Extract and clean content - remove headers and file names
        context_parts = []
        for doc in docs:
            content = doc.page_content.strip()
            if len(content) > 50:
                # Split content into lines and filter out headers/file names
                lines = content.split('\n')
                cleaned_lines = []
                
                for line in lines:
                    line = line.strip()
                    
                    # Skip common headers and file references
                    skip_patterns = [
                        'BASIC ALGORAND INFORMATION',
                        'Algorand Fundamentals', 
                        'What is Algorand?',
                        'What is',
                        'How to',
                        'Introduction',
                        'Overview',
                        'Document',
                        '.pdf',
                        '.txt',
                        '.md'
                    ]
                    
                    # Check if line should be skipped
                    should_skip = any(pattern in line for pattern in skip_patterns)
                    
                    # Only keep substantial content lines that aren't headers
                    if not should_skip and len(line) > 20 and not line.isupper():
                        cleaned_lines.append(line)
                
                if cleaned_lines:
                    context_parts.extend(cleaned_lines)
        
        if not context_parts:
            print("❌ No substantial content found in retrieved documents")
            return None, []
        
        # Join the cleaned content
        full_context = " ".join(context_parts)
        
        # Extract most relevant sentences for the answer
        sentences = full_context.split('.')
        relevant_sentences = []
        question_lower = question_text.lower()
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 30: 
                question_words = set(question_lower.split())
                sentence_words = set(sentence.lower().split())
                overlap = len(question_words.intersection(sentence_words))
                
                if overlap > 0:
                    relevant_sentences.append((sentence, overlap))
        
        # Sort by relevance and create formatted answer
        relevant_sentences.sort(key=lambda x: x[1], reverse=True)
        
        if relevant_sentences:
            # Create formatted answer with bold headings and spacing
            formatted_answer = ""
            
            # Extract topic name for heading
            if question_lower.startswith('what is'):
                topic = question_text.replace('what is', '').replace('?', '').strip()
                formatted_answer += f"**{topic.title()}**\n\n"
            elif question_lower.startswith('how'):
                formatted_answer += f"**How To Guide**\n\n"
            elif question_lower.startswith('why'):
                formatted_answer += f"**Explanation**\n\n"
            else:
                formatted_answer += f"**Answer**\n\n"
            
            # Main answer content
            main_content = relevant_sentences[0][0].strip()
            if not main_content.endswith('.'):
                main_content += "."
            formatted_answer += f"{main_content}\n\n"
            
            # Additional details if available
            if len(relevant_sentences) > 1:
                for sentence, _ in relevant_sentences[1:3]:
                    sentence = sentence.strip()
                    if sentence and len(sentence) > 30:
                        if not sentence.endswith('.'):
                            sentence += "."
                        formatted_answer += f"{sentence}\n\n"
            
            return formatted_answer.strip(), []
        
        return None, []
        
    except Exception as e:
        print(f"❌ Document retrieval error: {e}")
        return None, []

# Initialize Groq client (as fallback only)
def initialize_groq():
    groq_api_key = os.getenv("GROQ_API_KEY")
    
    if not groq_api_key:
        print("❌ GROQ_API_KEY not found in environment variables")
        return None
    
    groq_api_key = groq_api_key.strip()
    
    if groq_api_key == "":
        print("❌ GROQ_API_KEY is empty")
        return None
    
    if not groq_api_key.startswith('gsk_'):
        print("⚠ Warning: Groq API key should start with 'gsk_'")
    
    try:
        client = Groq(api_key=groq_api_key)
        print("✅ Groq client initialized successfully (as fallback)")
        return client
    except Exception as e:
        print(f"❌ Error initializing Groq client: {e}")
        return None

groq = initialize_groq()

bot = commands.Bot(command_prefix="!", intents=discord.Intents.all())

# ENHANCED question handling with simple code query routing
async def handle_question(question_text, send_func):
    """Handle question answering with code query routing to Groq"""
    try:
        print(f"🔍 Processing question: {question_text}")
        
        # STEP 1: Check if it's a code query - redirect to Groq
        if is_code_query(question_text):
            print("🔧 Code query detected - redirecting to Groq")
            
            if not groq:
                await send_func("❌ Code assistance service is unavailable. Please try again later.")
                return False
            
            # Simple Groq call for code queries
            try:
                response = groq.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert programming assistant. Help with code debugging, fixing, and programming questions. Provide clear explanations and working solutions."
                        },
                        {
                            "role": "user",
                            "content": question_text
                        }
                    ],
                    max_tokens=1500,
                    temperature=0.1
                )
                
                groq_answer = response.choices[0].message.content
                
                # Create embed for code response
                embed = discord.Embed(
                    title="🔧 Code Assistant",
                    description=groq_answer,
                    color=0x00ff88
                )
                embed.set_footer(text="🔧 Powered by Algorand Assistant")
                
                await send_func(embed=embed)
                return True
                
            except Exception as e:
                print(f"❌ Error in code query: {e}")
                await send_func(f"❌ Error processing code query: {str(e)[:200]}")
                return False
        
        # STEP 2: For non-code queries, try documents first
        print("📚 Non-code query - searching documents first")
        document_answer, source_info = get_document_answer(question_text)
        
        if document_answer and len(document_answer.strip()) > 50:
            print("✅ Found answer in documents")
            
            # Create rich embed for document-based answer
            embed = discord.Embed(
                title="Algorand Assistant Response",
                description=document_answer,
                color=0x00ff88
            )
            
            embed.set_footer(text="✅ Answer sourced from official documentation")
            
            await send_func(embed=embed)
            return True
        
        print("❌ No sufficient answer found in documents")
        
        # STEP 3: Groq fallback for non-code queries
        if not groq:
            await send_func("❌ I couldn't find relevant information in the Algorand documentation, and the AI fallback service is unavailable.")
            return False
            
        print("🤖 Using Groq as fallback")
        
        response = groq.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": f"As an Algorand expert, please answer: {question_text}"}],
            max_tokens=1000,
            temperature=0.3
        )
        
        groq_answer = response.choices[0].message.content
        
        # Create embed for Groq fallback answer
        embed = discord.Embed(
            title="Algorand Assistant Response",
            description=groq_answer,
            color=0x00ff88
        )
        embed.set_footer(text="✅ Answer sourced from official documentation")
        
        await send_func(embed=embed)
        return True
        
    except Exception as e:
        print(f"❌ Error in handle_question: {e}")
        error_embed = discord.Embed(
            title="❌ Error",
            description=f"Sorry, I encountered an error while processing your question: {str(e)[:200]}",
            color=0xff0000
        )
        await send_func(embed=error_embed)
        return False

@bot.event
async def on_ready():
    print(f'Logged on as {bot.user}!')
    print(f'Document system status: {"✅ Active" if retriever else "❌ Inactive"}')
    print(f'Groq fallback status: {"✅ Available" if groq else "❌ Unavailable"}')
    try:
        await bot.tree.sync()
        print("Slash commands synced")
    except Exception as e:
        print(f"Error syncing commands: {e}")

# Enhanced message handler with document priority and OCR support
@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    
    # Check if message has image attachments
    image_attachments = [att for att in message.attachments if att.content_type and att.content_type.startswith('image/')]
    
    if bot.user in message.mentions:
        # Clean the message content to remove the mention
        question_text = message.content.replace(f'<@{bot.user.id}>', '').strip()
        
        if image_attachments and question_text:
            # Handle OCR + RAG workflow
            async with message.channel.typing():
                await handle_image_question(image_attachments[0], question_text, message.channel.send)
        elif image_attachments:
            # Just OCR without specific question
            async with message.channel.typing():
                await handle_image_question(image_attachments[0], "What information can you extract from this image?", message.channel.send)
        elif question_text:
            # Regular text question (existing functionality)
            async with message.channel.typing():
                await handle_question(question_text, message.channel.send)
    
    await bot.process_commands(message)

# Enhanced message handler with document priority
@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    
    if bot.user in message.mentions:
        # Clean the message content to remove the mention
        question_text = message.content.replace(f'<@{bot.user.id}>', '').strip()
        
        if question_text:
            # Show typing indicator
            async with message.channel.typing():
                await handle_question(question_text, message.channel.send)
    
    await bot.process_commands(message)

# Enhanced question command
@bot.command()
async def question(ctx, *, question):
    """Handle questions with code query routing"""
    async with ctx.typing():
        await handle_question(question, ctx.send)

# New slash command for asking questions
@bot.tree.command(name="ask", description="Ask a question (code queries go to Groq, others search docs first)")
@app_commands.describe(question="Your question")
async def ask_slash(interaction: discord.Interaction, question: str):
    await interaction.response.defer()
    
    async def send_response(content):
        if isinstance(content, discord.Embed):
            await interaction.followup.send(embed=content)
        else:
            await interaction.followup.send(content)
    
    await handle_question(question, send_response)

# Status command to check system health
@bot.tree.command(name="status", description="Check bot and document system status")
async def status_command(interaction: discord.Interaction):
    embed = discord.Embed(title="🤖 Bot System Status", color=0x00ff88)
    
    # Document system status
    if retriever:
        embed.add_field(name="📚 Document System", value="✅ Active (Primary)", inline=True)
        doc_count = len(documents) if 'documents' in globals() else 0
        embed.add_field(name="📄 Documents Loaded", value=f"{doc_count}", inline=True)
    else:
        embed.add_field(name="📚 Document System", value="❌ Inactive", inline=True)
    
    # Groq fallback status
    if groq:
        embed.add_field(name="🤖 AI Fallback", value="✅ Available (Secondary)", inline=True)
    else:
        embed.add_field(name="🤖 AI Fallback", value="❌ Unavailable", inline=True)
    
    # Algorand services
    if testnet_dispenser:
        embed.add_field(name="💰 Faucet Service", value="✅ Active", inline=True)
    else:
        embed.add_field(name="💰 Faucet Service", value="❌ Inactive", inline=True)
    
    # Bot performance
    embed.add_field(name="🏓 Latency", value=f"{round(bot.latency * 1000)}ms", inline=True)
    embed.add_field(name="🏠 Servers", value=f"{len(bot.guilds)}", inline=True)
    
    embed.set_footer(text="Priority: Code→Groq | Others→Docs→Groq")
    
    await interaction.response.send_message(embed=embed)

# Initialize Algorand client with error handling
try:
    algorand_client = AlgorandClient.default_localnet()
    testnet_dispenser = algorand_client.client.get_testnet_dispenser(
        auth_token=os.getenv("ALGOKIT_DISPENSER_ACCESS_TOKEN")
    )
    print("✅ Algorand client initialized successfully")
except Exception as e:
    print(f"❌ Error initializing Algorand client: {e}")
    algorand_client = None
    testnet_dispenser = None

def is_valid_algorand_address(address: str) -> bool:
    """Validate Algorand address format using SDK"""
    try:
        return encoding.is_valid_address(address)
    except:
        return False

# Keep all existing faucet and quiz functionality unchanged
@bot.tree.command(name="faucet", description="Get TestNet ALGO tokens")
@app_commands.describe(wallet_address="Your Algorand TestNet address")
async def faucet(interaction: discord.Interaction, wallet_address: str):
    await interaction.response.defer()
    
    if not testnet_dispenser:
        return await interaction.followup.send("❌ Algorand testnet dispenser is not available")
    
    if not is_valid_algorand_address(wallet_address):
        return await interaction.followup.send("❌ Invalid Algorand address format")
    
    try:
        result = testnet_dispenser.fund(
            address=wallet_address,
            amount=1_000_000,
            asset_id=0
        )
        tx_id = getattr(result, "tx_id", None)
        if tx_id:
            explorer_link = f"https://testnet.explorer.perawallet.app/tx/{tx_id}"
            await interaction.followup.send(
                f"✅ 1 TestNet ALGO sent to {wallet_address}!\n"
                f"View transaction: {explorer_link}"
            )
        else:
            await interaction.followup.send(
                f"✅ 1 TestNet ALGO sent to {wallet_address}!\n"
                f"(No transaction ID returned; please check your wallet in a minute.)"
            )
    except Exception as e:
        await interaction.followup.send(f"⚠ Failed: {str(e)[:200]}")

# Quiz UI Classes
class QuizStartView(ui.View):
    def __init__(self, session):
        super().__init__(timeout=300)
        self.session = session
    
    @ui.button(label="Start Quiz", style=discord.ButtonStyle.green, emoji="🚀")
    async def start_quiz(self, interaction: discord.Interaction, button: ui.Button):
        if interaction.user.id != self.session.user_id:
            await interaction.response.send_message("❌ This quiz is not for you!", ephemeral=True)
            return
        
        question = self.session.get_current_question()
        if question:
            embed = self.create_question_embed(question, self.session.current_question + 1, len(self.session.questions))
            view = QuizQuestionView(self.session)
            await interaction.response.edit_message(embed=embed, view=view)
    
    def create_question_embed(self, question, question_num, total_questions):
        embed = discord.Embed(
            title=f"Question {question_num}/{total_questions}",
            description=question["question"],
            color=0x0099ff
        )
        
        options_text = "\n".join(question["options"])
        embed.add_field(name="Options:", value=options_text, inline=False)
        
        return embed

class QuizQuestionView(ui.View):
    def __init__(self, session):
        super().__init__(timeout=300)
        self.session = session
    
    @ui.button(label="A", style=discord.ButtonStyle.secondary)
    async def answer_a(self, interaction: discord.Interaction, button: ui.Button):
        await self.handle_answer(interaction, "A")
    
    @ui.button(label="B", style=discord.ButtonStyle.secondary)
    async def answer_b(self, interaction: discord.Interaction, button: ui.Button):
        await self.handle_answer(interaction, "B")
    
    @ui.button(label="C", style=discord.ButtonStyle.secondary)
    async def answer_c(self, interaction: discord.Interaction, button: ui.Button):
        await self.handle_answer(interaction, "C")
    
    @ui.button(label="D", style=discord.ButtonStyle.secondary)
    async def answer_d(self, interaction: discord.Interaction, button: ui.Button):
        await self.handle_answer(interaction, "D")
    
    async def handle_answer(self, interaction: discord.Interaction, answer: str):
        if interaction.user.id != self.session.user_id:
            await interaction.response.send_message("❌ This quiz is not for you!", ephemeral=True)
            return
        
        is_correct = self.session.answer_question(answer)
        current_answer = self.session.answers[-1]
        
        # Create feedback embed
        embed = discord.Embed(
            title="✅ Correct!" if is_correct else "❌ Incorrect!",
            description=f"**Your answer:** {answer}\n**Correct answer:** {current_answer['correct_answer']}",
            color=0x00ff00 if is_correct else 0xff0000
        )
        embed.add_field(name="Explanation:", value=current_answer['explanation'], inline=False)
        embed.add_field(name="Score:", value=f"{self.session.score}/{self.session.current_question}", inline=True)
        
        if self.session.is_completed():
            # Quiz completed
            results = self.session.get_results()
            view = QuizResultsView(results)
            quiz_manager.end_session(self.session.user_id)
            
            # Update embed for final results
            final_embed = discord.Embed(
                title="🎉 Quiz Completed!",
                description=f"**Final Score:** {results['score']}/{results['total']} ({results['percentage']:.1f}%)",
                color=0x00ff88 if results['percentage'] >= 70 else 0xffaa00 if results['percentage'] >= 50 else 0xff0000
            )
            
            # Performance message
            if results['percentage'] >= 90:
                performance = "🌟 Excellent! You're an Algorand expert!"
            elif results['percentage'] >= 70:
                performance = "👍 Great job! You have solid Algorand knowledge!"
            elif results['percentage'] >= 50:
                performance = "📚 Not bad! Keep learning about Algorand!"
            else:
                performance = "💪 Keep studying! Practice makes perfect!"
            
            final_embed.add_field(name="Performance:", value=performance, inline=False)
            final_embed.add_field(name="Level:", value=results['level'].title(), inline=True)
            
            await interaction.response.edit_message(embed=final_embed, view=view)
        else:
            # Next question
            view = QuizContinueView(self.session)
            await interaction.response.edit_message(embed=embed, view=view)

class QuizContinueView(ui.View):
    def __init__(self, session):
        super().__init__(timeout=300)
        self.session = session
    
    @ui.button(label="Next Question", style=discord.ButtonStyle.primary, emoji="➡")
    async def next_question(self, interaction: discord.Interaction, button: ui.Button):
        if interaction.user.id != self.session.user_id:
            await interaction.response.send_message("❌ This quiz is not for you!", ephemeral=True)
            return
        
        question = self.session.get_current_question()
        if question:
            embed = self.create_question_embed(question, self.session.current_question + 1, len(self.session.questions))
            view = QuizQuestionView(self.session)
            await interaction.response.edit_message(embed=embed, view=view)
    
    def create_question_embed(self, question, question_num, total_questions):
        embed = discord.Embed(
            title=f"Question {question_num}/{total_questions}",
            description=question["question"],
            color=0x0099ff
        )
        
        options_text = "\n".join(question["options"])
        embed.add_field(name="Options:", value=options_text, inline=False)
        
        return embed

class QuizResultsView(ui.View):
    def __init__(self, results):
        super().__init__(timeout=300)
        self.results = results
    
    @ui.button(label="Take Another Quiz", style=discord.ButtonStyle.green, emoji="🔄")
    async def take_another(self, interaction: discord.Interaction, button: ui.Button):
        embed = discord.Embed(
            title="🧠 Choose Your Next Challenge!",
            description="Select a difficulty level for your next quiz:",
            color=0x0099ff
        )
        view = LevelSelectView()
        await interaction.response.edit_message(embed=embed, view=view)
    
    @ui.button(label="View Detailed Results", style=discord.ButtonStyle.secondary, emoji="📊")
    async def detailed_results(self, interaction: discord.Interaction, button: ui.Button):
        embed = discord.Embed(
            title="📊 Detailed Quiz Results",
            description=f"**Level:** {self.results['level'].title()}\n**Score:** {self.results['score']}/{self.results['total']}",
            color=0x0099ff
        )
        
        for i, answer in enumerate(self.results['answers'], 1):
            status = "✅" if answer['is_correct'] else "❌"
            embed.add_field(
                name=f"{status} Question {i}",
                value=f"**Q:** {answer['question'][:50]}...\n**Your Answer:** {answer['user_answer']} | **Correct:** {answer['correct_answer']}",
                inline=False
            )
        
        await interaction.response.edit_message(embed=embed, view=self)

class LevelSelectView(ui.View):
    def __init__(self):
        super().__init__(timeout=300)
    
    @ui.button(label="Beginner", style=discord.ButtonStyle.success, emoji="🟢")
    async def beginner(self, interaction: discord.Interaction, button: ui.Button):
        await self.start_new_quiz(interaction, "beginner")
    
    @ui.button(label="Intermediate", style=discord.ButtonStyle.primary, emoji="🟡")
    async def intermediate(self, interaction: discord.Interaction, button: ui.Button):
        await self.start_new_quiz(interaction, "intermediate")
    
    @ui.button(label="Advanced", style=discord.ButtonStyle.danger, emoji="🔴")
    async def advanced(self, interaction: discord.Interaction, button: ui.Button):
        await self.start_new_quiz(interaction, "advanced")
    
    async def start_new_quiz(self, interaction: discord.Interaction, level: str):
        user_id = interaction.user.id
        session = quiz_manager.start_quiz(user_id, level)
        
        embed = discord.Embed(
            title=f"🧠 Algorand Quiz - {level.title()} Level",
            description=f"Starting a new {level} level quiz!\n\n"
                       f"📝 **Rules:**\n"
                       f"• {len(session.questions)} questions total\n"
                       f"• Click the button with your answer (A, B, C, or D)\n"
                       f"• No time limit per question\n"
                       f"• You'll see explanations after each answer\n\n"
                       f"Ready to test your Algorand knowledge again?",
            color=0x00ff88
        )
        
        view = QuizStartView(session)
        await interaction.response.edit_message(embed=embed, view=view)

# Quiz Commands (ADD THESE - they don't conflict with existing faucet)
@bot.tree.command(name="quiz", description="Start an Algorand knowledge quiz")
@app_commands.describe(level="Choose difficulty level")
@app_commands.choices(level=[
    app_commands.Choice(name="Beginner", value="beginner"),
    app_commands.Choice(name="Intermediate", value="intermediate"),
    app_commands.Choice(name="Advanced", value="advanced")
])
async def quiz(interaction: discord.Interaction, level: str):
    user_id = interaction.user.id
    
    # Check if user already has an active quiz
    existing_session = quiz_manager.get_session(user_id)
    if existing_session and not existing_session.is_completed():
        await interaction.response.send_message(
            "❌ You already have an active quiz! Use the buttons to continue or `/quiz_stop` to end it.",
            ephemeral=True
        )
        return
    
    # Start new quiz session
    session = quiz_manager.start_quiz(user_id, level)
    
    embed = discord.Embed(
        title=f"🧠 Algorand Quiz - {level.title()} Level",
        description=f"Welcome {interaction.user.mention}! You're about to start a {level} level quiz.\n\n"
                   f"📝 **Rules:**\n"
                   f"• {len(session.questions)} questions total\n"
                   f"• Click the button with your answer (A, B, C, or D)\n"
                   f"• No time limit per question\n"
                   f"• You'll see explanations after each answer\n\n"
                   f"Ready to test your Algorand knowledge?",
        color=0x00ff88
    )
    
    view = QuizStartView(session)
    await interaction.response.send_message(embed=embed, view=view, ephemeral=True)

@bot.tree.command(name="quiz_stop", description="Stop your current quiz")
async def quiz_stop(interaction: discord.Interaction):
    user_id = interaction.user.id
    session = quiz_manager.get_session(user_id)
    
    if not session:
        await interaction.response.send_message("❌ You don't have an active quiz!", ephemeral=True)
        return
    
    quiz_manager.end_session(user_id)
    await interaction.response.send_message("✅ Quiz stopped successfully!", ephemeral=True)

@bot.tree.command(name="quiz_stats", description="View available quiz levels and question counts")
async def quiz_stats(interaction: discord.Interaction):
    embed = discord.Embed(
        title="📊 Quiz Statistics",
        description="Available quiz levels and question counts:",
        color=0x0099ff
    )
    
    for level, questions in QUIZ_QUESTIONS.items():
        embed.add_field(
            name=f"{level.title()} Level",
            value=f"{len(questions)} questions available",
            inline=True
        )
    
    embed.add_field(
        name="How to use:",
        value="Use `/quiz <level>` to start a quiz!",
        inline=False
    )
    
    await interaction.response.send_message(embed=embed, ephemeral=True)

if __name__ == "__main__":
    discord_token = os.getenv("SECRET_KEY")
    if not discord_token:
        print("❌ SECRET_KEY (Discord token) not found in environment variables")
        print("Please add SECRET_KEY=your_discord_bot_token to your .env file")
    else:
        print("🚀 Starting bot with simple code query routing to Groq...")
        bot.run(discord_token)
