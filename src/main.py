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

load_dotenv()

# Initialize clients
groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
bot = commands.Bot(command_prefix="!", intents=discord.Intents.all())
algorand_client = AlgorandClient.default_localnet()
testnet_dispenser = algorand_client.client.get_testnet_dispenser(
    auth_token=os.getenv("ALGOKIT_DISPENSER_ACCESS_TOKEN")
)

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

# Existing faucet command
@bot.tree.command(name="faucet", description="Get TestNet ALGO tokens")
@app_commands.describe(wallet_address="Your Algorand TestNet address")
async def faucet(interaction: discord.Interaction, wallet_address: str):
    await interaction.response.defer()
    
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
        await interaction.followup.send(f"⚠️ Failed: {str(e)[:200]}")

# New Quiz Commands
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

# Discord UI Views and Buttons
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
    
    @ui.button(label="Next Question", style=discord.ButtonStyle.primary, emoji="➡️")
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

# Existing message handler for Groq AI
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