from groq import Groq
from dotenv import load_dotenv
import os


class ConversationAgent:
    def __init__(self):
        load_dotenv()
        self.client = Groq(api_key=os.environ["GROQ_KEY"])
        self.initiate_history()

    @staticmethod
    def read_file(file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    def initiate_history(self) -> None:
        self.history = [
            {
                "role": "system",
                "content": ConversationAgent.read_file("./context.txt"),
            }
        ]

    def update_history(self, role: str, content: str) -> None:
        self.history.append(
            {
                "role": role,
                "content": content,
            }
        )

    # ========= STT : audio (bytes) -> texte =========
    def stt_audio_to_text_bytes(
        self,
        audio_bytes: bytes,
        filename: str,
        mime_type: str,
        stt_model: str = "whisper-large-v3",
        language: str = "fr",
        prompt: str | None = None,
    ) -> str:
        transcription = self.client.audio.transcriptions.create(
            file=(filename, audio_bytes, mime_type),
            model=stt_model,
            response_format="json",
            language=language,
            temperature=0.0,
            prompt=prompt,
        )
        return (transcription.text or "").strip()

    # ========= Pipeline : audio -> texte -> LLM =========
    def ask_llm_from_audio(
        self,
        audio_bytes: bytes,
        model: str,
        mime_type: str,
        filename: str,
        stt_model: str = "whisper-large-v3",
        language: str = "fr",
        prompt: str | None = None,
    ) -> str:
        user_text = self.stt_audio_to_text_bytes(
            audio_bytes=audio_bytes,
            filename=filename,
            mime_type=mime_type,
            stt_model=stt_model,
            language=language,
            prompt=prompt,
        )

        if not user_text:
            user_text = "Je n’ai pas compris ton audio, parle plus clairement gamin."

        response = self.ask_llm(
            user_interaction=user_text,
            model=model,
        )

        return response

    # ========= LLM texte -> texte =========
    def ask_llm(self, user_interaction: str, model: str) -> str:
        self.update_history(role="user", content=user_interaction)

        response = self.client.chat.completions.create(
            messages=self.history,
            model=model,
        ).choices[0].message.content

        self.update_history(role="assistant", content=response)

        return response

    # ========= Interface terminal (texte) =========
    def terminal_user_interface(self, model: str) -> None:
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
