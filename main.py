from groq import Groq
from dotenv import load_dotenv
import os


class ConversationAgent:
    def __init__(self):
        load_dotenv()
        self.client = Groq(api_key=os.environ["GROQ_KEY"])
        self.initiate_history()


    @staticmethod
    def read_file(file_path):
        with open(file_path , "r") as file:
            return file.read()


    def initiate_history(self):
        self.history = [
            {
                "role": "system",
                "content": ConversationAgent.read_file("./context.txt")
            }]


    def update_history(self, role, content):
         self.history.append(
                {
                    "role": role,
                    "content": content,
                })


    def ask_llm(self, user_interaction):

        self.update_history(role="user", content=user_interaction)

        response = self.client.chat.completions.create(
            messages=self.history,
            model="llama-3.3-70b-versatile"
        ).choices[0].message.content
        
        self.update_history(role="assistant", content=response)

        return response



    def terminal_user_interface(self):
        while True:
            user_interaction = input("Vous : ")
            if user_interaction.lower() == "exit":
                break
            elif user_interaction == "":
                print("Jarvis : Vous n'avez rien à dire ?")
            else:
                response = self.ask_llm(user_interaction=user_interaction)
                print(f"Jarvis : {response}")







if __name__ == "__main__":
    conversation_agent = ConversationAgent()
    conversation_agent.terminal_user_interface()