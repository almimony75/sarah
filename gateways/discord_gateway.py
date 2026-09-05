import asyncio
import aiohttp
import discord
import re

# --- CONFIGURATION ---
TOKEN = "put your own token"
SARAH_API_URL = "http://127.0.0.1:9000/turn"

# Add the exact names of the channels you want Sarah to auto-reply in without needing an @mention
DEDICATED_CHANNELS = ["chat-with-sarah", "sarah-testing"]

# Enable the privileged intent you just toggled on the dashboard
intents = discord.Intents.default()
intents.message_content = True

def split_message(text: str, max_len: int = 1900) -> list[str]:
    """Splits long text cleanly around line breaks or spaces to stay under Discord's 2000 char limit."""
    if len(text) <= max_len:
        return [text]
    
    chunks = []
    while len(text) > max_len:
        split_idx = text.rfind("\n", 0, max_len)
        if split_idx == -1:
            split_idx = text.rfind(" ", 0, max_len)
        if split_idx == -1:
            split_idx = max_len
        
        chunks.append(text[:split_idx].strip())
        text = text[split_idx:].strip()
    
    if text:
        chunks.append(text)
    return chunks

class SarahGateway(discord.Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.session = None

    async def setup_hook(self):
        # Create an async HTTP session to talk to your C++ backend
        self.session = aiohttp.ClientSession()

    async def close(self):
        if self.session:
            await self.session.close()
        await super().close()

    async def on_ready(self):
        # Give her a custom status in the Discord sidebar!
        activity = discord.Activity(type=discord.ActivityType.listening, name="Sarah_Core | Local AI")
        await self.change_presence(status=discord.Status.online, activity=activity)
        
        print("=================================================")
        print(f" ONLINE: Discord Gateway connected as {self.user}")
        print("=================================================")

    async def on_message(self, message: discord.Message):
        # 1. Ignore our own messages to prevent infinite loops
        if message.author.id == self.user.id:
            return

        # 2. SMART TRIGGERS
        is_dm = isinstance(message.channel, discord.DMChannel)
        is_mentioned = self.user.mentioned_in(message)
        
        is_reply_to_me = False
        quoted_text = ""
        if message.reference and message.reference.resolved:
            if isinstance(message.reference.resolved, discord.Message):
                if message.reference.resolved.author.id == self.user.id:
                    is_reply_to_me = True
                    # Grab the text of the old message you are replying to
                    quoted_text = message.reference.resolved.content
                    
        is_dedicated_channel = False
        if hasattr(message.channel, "name") and message.channel.name in DEDICATED_CHANNELS:
            is_dedicated_channel = True

        # If none of the triggers are met, ignore the message
        if not (is_dm or is_mentioned or is_reply_to_me or is_dedicated_channel):
            return

        # 3. Clean up the text (strip the @mention tag out of the string)
        clean_text = re.sub(rf"<@!?{self.user.id}>", "", message.content).strip()
        
        if not clean_text:
            return

        # NEW: If the user replied to an old message, format it so the C++ LLM understands the context
        if is_reply_to_me and quoted_text:
            final_prompt = f'[Replying to your past message: "{quoted_text}"]\n{clean_text}'
        else:
            final_prompt = clean_text

        # 4. Continuous typing indicator task (keeps typing if C++ takes longer than 10s)
        stop_typing = asyncio.Event()

        async def keep_typing():
            while not stop_typing.is_set():
                await message.channel.typing()
                try:
                    await asyncio.wait_for(stop_typing.wait(), timeout=8.0)
                except asyncio.TimeoutError:
                    pass

        typing_task = asyncio.create_task(keep_typing())

        # 5. Package the payload for the C++ Engine
        payload = {
            "device_id": f"discord_{message.author.id}",
            "text": final_prompt
        }

        try:
            # 6. Fire the request to localhost:9000
            async with self.session.post(SARAH_API_URL, json=payload, timeout=120) as response:
                if response.status == 200:
                    data = await response.json()
                    reply_text = data.get("reply", "")

                    if reply_text:
                        # 7. Chunk and send the response
                        chunks = split_message(reply_text)
                        for i, chunk in enumerate(chunks):
                            if i == 0:
                                await message.reply(chunk)
                            else:
                                await message.channel.send(chunk)
                    else:
                        await message.reply("*[Sarah nodded, but said nothing.]*")
                else:
                    await message.reply(f"*Brain returned an error (HTTP {response.status}).*")
        except Exception as e:
            print(f"Gateway Error: {e}")
            await message.reply("*My local C++ engine appears to be offline right now.*")
        finally:
            # Ensure we always stop the typing indicator when done
            stop_typing.set()
            await typing_task

# Boot the gateway
client = SarahGateway(intents=intents)
client.run(TOKEN)