import os
from typing import Generator
from io import BytesIO
import re
import base64
from pydub import AudioSegment

from groq import Groq
from google.cloud import texttospeech
from config import LLM_MODELS, VISION_MODELS, STT_MODEL, LLM_MODEL, TTS_LANGUAGE_CODE, TTS_VOICE_NAME, TTS_AUDIO_ENCODING

from dotenv import load_dotenv

load_dotenv()


class ConversationAgent:
    def __init__(self) -> None:
        load_dotenv()

        self.history = [
            {
                "role": "system",
                "content": ConversationAgent.read_file("./context.txt")
            }
        ]

        # Initialisation des clients Groq (LLM & STT)
        self.groq_client = Groq(api_key=os.getenv("GROQ_KEY"))
        # Initialisation du client Google Cloud TTS
        self.tts_client = texttospeech.TextToSpeechClient()

        # Laisse les attributs LLM/Vision de l'ancienne version pour ne pas casser frontend.py
        self.large_language_model = LLM_MODELS[0]
        self.vision_model = VISION_MODELS[0]

    @staticmethod
    def read_file(file_path: str) -> str:
        """Lit le contenu d'un fichier texte (pour le contexte)."""
        with open(file_path, "r") as file:
            return file.read()

    @staticmethod
    def read_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    @staticmethod
    def read_audio_file(audio_file_path: str) -> bytes:
        """Lit le contenu d'un fichier audio en bytes."""
        with open(audio_file_path, "rb") as audio_file:
            return audio_file.read()

    @staticmethod
    def format_streamlit_image_to_base64(streamlit_file_object):
        """Méthode conservée pour la compatibilité avec frontend.py."""
        bytes_data = streamlit_file_object.read()
        b64_bytes = base64.b64encode(bytes_data)
        b64_str = b64_bytes.decode("utf-8")
        mime = "image/png" if streamlit_file_object.type == "image/png" else "image/jpeg"
        return f"data:{mime};base64,{b64_str}"

    def drop_memory(self):
        """Réinitialise l'historique de la conversation."""
        self.history = [
            {
                "role": "system",
                "content": ConversationAgent.read_file("./context.txt")
            }
        ]

    def transcribe_audio(self, audio_bytes: bytes) -> str:
        """Transcrit un fichier audio en texte en utilisant Groq (Whisper)."""
        audio_file_buffer = BytesIO(audio_bytes)
        audio_file_buffer.name = "audio.mp3"

        transcript = self.groq_client.audio.transcriptions.create(
            file=audio_file_buffer,
            model=STT_MODEL,
            language="fr"
        )

        return transcript.text

    def ask_model(self, message: str) -> Generator[str, None, None]:
        """Interroge le modèle LLM (Groq) en streaming et yield chaque phrase complète."""
        self.history.append(
            {
                "role": "user",
                "content": message,
            }
        )

        groq_stream = self.groq_client.chat.completions.create(
            messages=self.history,
            model=LLM_MODEL,
            temperature=0.5,
            max_completion_tokens=1024,
            top_p=1,
            stop=None,
            stream=True
        )

        list_of_sentences, current_sentence_parts = [], []

        for chunk in groq_stream:
            chunk_content = chunk.choices[0].delta.content or ""

            splited_chunck = re.split(r"([\.\!?])", chunk_content)

            if len(splited_chunck) == 1:
                current_sentence_parts.append(chunk_content)

            else:
                current_sentence_parts.append(splited_chunck[0])
                current_sentence_parts.append(splited_chunck[1])

                complete_sentence = "".join(current_sentence_parts)
                list_of_sentences.append(complete_sentence)
                yield complete_sentence

                current_sentence_parts = [splited_chunck[2]]

        final_fragment = "".join(current_sentence_parts)
        if final_fragment:
            list_of_sentences.append(final_fragment)
            yield final_fragment

        self.history.append(
            {
                "role": "assistant",
                "content": "".join(list_of_sentences),
            }
        )

    def speak(self, text: str, save: bool = False) -> bytes:
        """Génère l'audio (TTS) avec Google Cloud TTS et retourne les bytes."""
        synthesis_input = texttospeech.SynthesisInput(text=text)

        voice = texttospeech.VoiceSelectionParams(
            language_code=TTS_LANGUAGE_CODE,
            name=TTS_VOICE_NAME
        )

        audio_encoding = texttospeech.AudioEncoding.MP3 if TTS_AUDIO_ENCODING == "MP3" else texttospeech.AudioEncoding.LINEAR16

        audio_config = texttospeech.AudioConfig(
            audio_encoding=audio_encoding
        )

        response = self.tts_client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )

        if save:
            if not os.path.exists("./audio_samples/"):
                os.makedirs("./audio_samples/")
            with open('./audio_samples/answer.mp3', "ab") as audio_file:
                audio_file.write(response.audio_content)

        return response.audio_content

    def try_pipeline(self, audio_file_path="./audio_samples/question.mp3"):
        """Pipeline de test complet: STT -> LLM Streaming -> TTS Sentence-by-Sentence."""
        print(f"Tentative de lecture du fichier : {audio_file_path}")
        if not os.path.exists(audio_file_path):
            print("ERREUR : Le fichier audio de la question n'existe pas. Créez-en un.")
            return

        if os.path.exists('./audio_samples/answer.mp3'):
            os.remove('./audio_samples/answer.mp3')

        audio_bytes = ConversationAgent.read_audio_file(audio_file_path)
        question_text = self.transcribe_audio(audio_bytes)

        print(f"→ Transcription : {question_text}")

        full_answer_audio = AudioSegment.empty()

        for sentence in self.ask_model(question_text):
            print("→ Phrase LLM reçue :", sentence)

            audio_bytes = self.speak(sentence, save=True)

        print("→ Réponse audio complète sauvegardée dans ./audio_samples/answer.mp3")
        self.drop_memory()

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

    def get_history(self, type_model):
        if type_model == "large_language_model":
            filtred_history = []
            for message in self.history:
                if type(message["content"]) == str:
                    filtred_history.append(message)

                elif type(message["content"]) == list:
                    filtred_history.append(
                        {
                            "role": message["role"],
                            "content": f'{message["content"][0]["text"]} : [IMAGE]',
                        },
                    )

            return filtred_history

        elif type_model == "vision_model":
            return self.history

    def ask_llm(self, user_interaction):

        self.update_history(role="user", content=user_interaction)

        response = self.groq_client.chat.completions.create(
                        messages=self.get_history(type_model="large_language_model"),
                        model=self.large_language_model
                    ).choices[0].message.content

        self.update_history(role="assistant", content=response)

        return response

    def ask_vision_model(self, user_interaction, image_b64):

            content = [
                        {
                            "type": "text", 
                            "text": user_interaction},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"{image_b64}",
                        },
                        },
                    ]


            self.update_history(role="user", content=content)

            response = self.groq_client.chat.completions.create(
                            messages=self.get_history(type_model="vision_model"),
                            model=self.vision_model,
            ).choices[0].message.content

            self.update_history(role="assistant", content=response)

            return response


if __name__ == "__main__":
    conversation_agent = ConversationAgent()
    conversation_agent.try_pipeline()
