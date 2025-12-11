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


    def stt_audio_to_text(
        self,
        audio_file_path: str,
        model: str = "whisper/whisper-large-v3",
        language: str = "fr",
        prompt: str | None = None,
    ) -> str:
        with open(audio_file_path, "rb") as audio_file:
            transcription = self.client.audio.transcriptions.create(
                file=audio_file,
                model=model,
                response_format="json",
                language=language,
                temperature=0.0,
                prompt=prompt,
            )

        return transcription.text


    def ask_llm_from_audio(
        self,
        audio_file_path: str,
        model: str,
        stt_model: str = "whisper/whisper-large-v3",
        language: str = "fr",
    ) -> str:

        user_text = self.stt_audio_to_text(
            audio_file_path=audio_file_path,
            model=stt_model,
            language=language,
        )

        return self.ask_llm(
            user_interaction=user_text,
            model=model
        )


    def ask_llm(self, user_interaction, model):

        self.update_history(role="user", content=user_interaction)

        response = self.client.chat.completions.create(
            messages=self.history,
            model=model
        ).choices[0].message.content
        
        self.update_history(role="assistant", content=response)

        return response



    def terminal_user_interface(self, model):
        while True:
            user_interaction = input("Vous : ")
            if user_interaction.lower() == "exit":
                break
            elif user_interaction == "":
                print("Jarvis : Vous n'avez rien à dire ?")
            else:
                response = self.ask_llm(user_interaction=user_interaction, model=model)
                print(f"Jarvis : {response}")





if __name__ == "__main__":
    conversation_agent = ConversationAgent()
    conversation_agent.terminal_user_interface(model="openai/gpt-oss-120b")