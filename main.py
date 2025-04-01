import discord
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class MyClient(discord.Client):
    async def on_ready(self):
        print(f'Logged on as {self.user}!')

    async def on_message(self, message):
        if message.author == self.user:
            return

        if self.user in message.mentions:
            response = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{"role": "user", "content": message.content}],
                max_tokens=1024,
                temperature=0.7,
                top_p=1,
                stream=True,
            )
            reply=""
            for chunk in response:
                content = chunk.choices[0].delta.content
                if content:
                    reply+=content
            await message.channel.send(reply)

intents = discord.Intents.default()
intents.message_content = True
discord_client = MyClient(intents=intents)
discord_client.run(os.getenv("SECRET_KEY"))