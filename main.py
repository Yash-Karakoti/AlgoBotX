import discord
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))  # Renamed for clarity

class MyClient(discord.Client):
    async def on_ready(self):
        print(f'Logged on as {self.user}!')

    async def on_message(self, message):
        if message.author == self.user:
            return

        if self.user in message.mentions:
            response = groq_client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{"role": "user", "content": message.content}],
                max_tokens=800,
                temperature=0.7,
                top_p=1,
                stream=True,
            )
            
            reply_buffer = []  # Store chunks
            current_length = 0
            
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                if current_length + len(content) > 1900:  # Safety margin
                    # Send current buffer and reset
                    await message.channel.send("".join(reply_buffer))
                    reply_buffer = []
                    current_length = 0
                
                reply_buffer.append(content)
                current_length += len(content)
            
            # Send remaining content
            if reply_buffer:
                full_reply = "".join(reply_buffer)
                # Split into 2000-character chunks
                for i in range(0, len(full_reply), 1900):
                    await message.channel.send(full_reply[i:i+1900])

intents = discord.Intents.default()
intents.message_content = True
discord_client = MyClient(intents=intents)
discord_client.run(os.getenv("SECRET_KEY"))