import os
from typing import Generator
from io import BytesIO
import io
import wave
import re
import base64
from pydub import AudioSegment

from groq import Groq
from google.cloud import texttospeech
from config import LLM_MODELS, VISION_MODELS, STT_MODEL, LLM_MODEL, TTS_LANGUAGE_CODE, TTS_VOICE_NAME, TTS_AUDIO_ENCODING, PERSONAS

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
        # TTS persona settings (can be overridden by frontend)
        default_persona = PERSONAS.get("Jarvis", {})
        self.persona_name = default_persona.get("display_name", "Jarvis")
        self.persona_context = default_persona.get("context", ConversationAgent.read_file("./context.txt"))
        self.tts_language_code = default_persona.get("tts_language_code", TTS_LANGUAGE_CODE)
        self.tts_voice_name = default_persona.get("tts_voice_name", TTS_VOICE_NAME)
        self.gemini_voice = default_persona.get("gemini_voice", None)
        self.gemini_locale = default_persona.get("gemini_locale", None)
        self.gemini_model = default_persona.get("gemini_model", None)

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
            language_code=getattr(self, 'tts_language_code', TTS_LANGUAGE_CODE),
            name=getattr(self, 'tts_voice_name', TTS_VOICE_NAME)
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

    def speak_stream(self, text: str) -> Generator[bytes, None, None]:
        """Génère un flux audio (TTS) en streaming via Google Cloud StreamingSynthesize.

        Yield des chunks `bytes` contenant `audio_content` au fur et à mesure.
        Utile pour la lecture temps réel côté front-end.
        """
        # Prépare les paramètres de voix et de sortie
        voice = texttospeech.VoiceSelectionParams(
            language_code=getattr(self, 'tts_language_code', TTS_LANGUAGE_CODE),
            name=getattr(self, 'tts_voice_name', TTS_VOICE_NAME),
        )

        audio_encoding = texttospeech.AudioEncoding.MP3 if TTS_AUDIO_ENCODING == "MP3" else texttospeech.AudioEncoding.LINEAR16

        # Configuration de streaming — l'API streaming n'accepte pas "audio_config" directement
        # Actuellement l'API streaming n'autorise que les voix Chirp 3 HD. Si la voix
        # configurée n'est pas une voix Chirp, on bascule vers un fallback Chirp HD.
        configured_voice_name = (getattr(self, 'tts_voice_name', TTS_VOICE_NAME) or "")
        if "chirp" in configured_voice_name.lower():
            streaming_voice_name = configured_voice_name
            streaming_language = getattr(self, 'tts_language_code', TTS_LANGUAGE_CODE)
        else:
            streaming_voice_name = "en-US-Chirp3-HD-Charon"
            streaming_language = "en-US"

        streaming_voice = texttospeech.VoiceSelectionParams(
            language_code=streaming_language,
            name=streaming_voice_name,
        )

        streaming_config = texttospeech.StreamingSynthesizeConfig(
            voice=streaming_voice,
        )

        # Premier message: la configuration
        config_request = texttospeech.StreamingSynthesizeRequest(
            streaming_config=streaming_config
        )

        # Générateur de requêtes: d'abord la config, puis le texte
        def request_generator():
            yield config_request
            yield texttospeech.StreamingSynthesizeRequest(
                input=texttospeech.StreamingSynthesisInput(text=text)
            )

        # Appel streaming
        streaming_responses = self.tts_client.streaming_synthesize(request_generator())

        for response in streaming_responses:
            # Chaque réponse peut contenir audio_content (bytes)
            if getattr(response, "audio_content", None):
                yield response.audio_content

    def speak_stream_voice_clone(self, voice_cloning_key: str, text_chunks: list, sample_rate_hz: int = 24000, output_wav_path: str | None = None) -> Generator[bytes, None, None]:
        """Stream TTS en utilisant les paramètres de voice cloning.

        Args:
            voice_cloning_key: clé de clonage vocale fournie par l'API.
            text_chunks: liste de chaînes représentant des fragments de texte "streamés".
            sample_rate_hz: fréquence d'échantillonnage souhaitée pour l'export WAV.
            output_wav_path: si fourni, écrira le flux collecté dans un fichier WAV.

        Yields:
            chunks d'octets audio renvoyés par l'API.
        """
        try:
            voice_clone_params = texttospeech.VoiceCloneParams(
                voice_cloning_key=voice_cloning_key
            )

            streaming_config = texttospeech.StreamingSynthesizeConfig(
                voice=texttospeech.VoiceSelectionParams(
                        language_code=getattr(self, 'tts_language_code', TTS_LANGUAGE_CODE),
                        voice_clone=voice_clone_params,
                    ),
                streaming_audio_config=texttospeech.StreamingAudioConfig(
                    audio_encoding=texttospeech.AudioEncoding.PCM,
                    sample_rate_hertz=sample_rate_hz,
                ),
            )

            config_request = texttospeech.StreamingSynthesizeRequest(
                streaming_config=streaming_config
            )

            def request_generator():
                yield config_request
                for text in text_chunks:
                    yield texttospeech.StreamingSynthesizeRequest(
                        input=texttospeech.StreamingSynthesisInput(text=text)
                    )

            streaming_responses = self.tts_client.streaming_synthesize(request_generator())

            audio_buffer = io.BytesIO()
            for response in streaming_responses:
                if getattr(response, "audio_content", None):
                    audio_buffer.write(response.audio_content)
                    yield response.audio_content

            if output_wav_path:
                with wave.open(output_wav_path, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(sample_rate_hz)
                    wav_file.writeframes(audio_buffer.getvalue())
        except Exception as e:
            # Remonter l'erreur pour que l'appelant puisse la traiter/afficher
            raise

    def speak_stream_gemini(self, prompt: str | None, text_chunks: list[str], model: str | None = None, voice: str | None = None, locale: str | None = None, sample_rate_hz: int = 24000) -> Generator[bytes, None, None]:
        """Utilise le mode streaming Gemini TTS (google-cloud-texttospeech >= 2.29.0).

        Pour chaque chunk de réponse reçu, renvoie un WAV (bytes) jouable par le navigateur.

        Args:
            prompt: instructions de style (appliquées au premier chunk).
            text_chunks: liste de fragments texte à synthétiser (simule un flux).
            model: nom du modèle TTS (ex: "gemini-2.5-flash-tts").
            voice: nom de la voix (ex: "leda").
            locale: code locale (ex: "en-US").
            sample_rate_hz: fréquence d'échantillonnage pour le WAV renvoyé.

        Yields:
            bytes: contenu d'un fichier WAV valide (RIFF) pour chaque chunk reçu.
        """
        client = self.tts_client

        # Use persona defaults when parameters omitted
        model = model or getattr(self, 'gemini_model', None)
        voice = voice or getattr(self, 'gemini_voice', None)
        locale = locale or getattr(self, 'gemini_locale', None)

        # If no explicit prompt provided, craft a persona-aware prompt
        if prompt is None:
            name = getattr(self, 'persona_name', '').lower()
            if 'donatella' in name:
                prompt = "Please read the following French text in a female voice with an elegant Italian accent, refined and authoritative."
            elif 'giorno' in name:
                prompt = "Please read the following French text in a young male voice with an Italian accent, calm and determined."
            else:
                prompt  = "Please read the following French text in a male Italian voice, confident and direct."

        config_request = texttospeech.StreamingSynthesizeRequest(
            streaming_config=texttospeech.StreamingSynthesizeConfig(
                voice=texttospeech.VoiceSelectionParams(
                    name=voice,
                    language_code=locale,
                    model_name=model,
                )
            )
        )

        def request_generator():
            yield config_request
            for i, text in enumerate(text_chunks):
                yield texttospeech.StreamingSynthesizeRequest(
                    input=texttospeech.StreamingSynthesisInput(
                        text=text,
                        prompt=prompt if i == 0 else None,
                    )
                )

        streaming_responses = client.streaming_synthesize(request_generator())

        # Pour chaque réponse, envelopper response.audio_content (PCM) dans un WAV minimal
        for response in streaming_responses:
            if not getattr(response, "audio_content", None):
                continue

            pcm_bytes = response.audio_content

            # Construire un WAV en mémoire pour ce chunk
            buffer = BytesIO()
            with wave.open(buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # int16
                wav_file.setframerate(sample_rate_hz)
                wav_file.writeframes(pcm_bytes)

            wav_bytes = buffer.getvalue()
            yield wav_bytes

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
