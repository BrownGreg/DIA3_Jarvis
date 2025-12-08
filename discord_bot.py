import os
import discord
from discord import app_commands
from dotenv import load_dotenv
from conversation_agent import ConversationAgent 

class DiscordBot(discord.Client):
    def __init__(self, conversation_agent):
        # intents ?
        intents = discord.Intents.default()
        super().__init__(intents=intents)
        
        self.conversation_agent = conversation_agent
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self):
        
        # /ask command
        @self.tree.command(name="ask", description="Pose une question à Jarvis")
        @app_commands.describe(question="Il est un peu enervé, fais attention à ce que tu racontes...")
        async def ask(interaction: discord.Interaction, question: str):
            
            await interaction.response.defer()
            
            try:
                response = self.conversation_agent.ask_llm(user_interaction=question)
                full_response = f"**Question :** {question}\n\n{response}"
                
                # Limit discord message length 
                if len(full_response) > 2000:
                    full_response = full_response[:1990] + "..."
                
                await interaction.followup.send(full_response)
                
            except Exception as e:
                await interaction.followup.send(f"Une erreur est survenue : {e}")
        await self.tree.sync()
        print("Commandes synchronisées. Si elles n'apparaissent pas redémarre Discord (Ctrl+R).")

    # Event when the bot is ready (local only)
    async def on_ready(self):
        print(f'Connecté en tant que {self.user} !')

if __name__ == "__main__":
    load_dotenv()
    discord_token = os.getenv('DISCORD_TOKEN')
    
    if not discord_token:
        print("Erreur : Token manquant dans le .env")
    else:
        agent = ConversationAgent()
        bot = DiscordBot(conversation_agent=agent)
        bot.run(discord_token)