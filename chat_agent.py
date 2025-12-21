import os
import io
import wave
import re
import base64
from typing import Generator
from io import BytesIO

import streamlit
from pydub import AudioSegment
from groq import Groq
from google.cloud import texttospeech
from dotenv import load_dotenv

from config import (
    LLM_MODELS,
    VISION_MODELS,
    STT_MODEL,
    LLM_MODEL,
    LLM_MAX_COMPLETION_TOKENS,
    TTS_LANGUAGE_CODE,
    TTS_VOICE_NAME,
    TTS_AUDIO_ENCODING,
    PERSONAS,
)

load_dotenv()


class ConversationAgent:
    def __init__(self) -> None:
        load_dotenv()

        self.history = [
            {
                "role": "system",
                "content": ConversationAgent.read_file("./context.txt"),
            }
        ]

        self.groq_client = Groq(api_key=os.getenv("GROQ_KEY"))
        self.tts_client = texttospeech.TextToSpeechClient()

        self.large_language_model = LLM_MODELS[0]
        self.vision_model = VISION_MODELS[0]

        default_persona = PERSONAS.get("Jarvis", {})
        self.persona_name = default_persona.get("display_name", "Jarvis")
        self.persona_context = default_persona.get("context", ConversationAgent.read_file("./context.txt"))
        self.tts_language_code = default_persona.get("tts_language_code", TTS_LANGUAGE_CODE)
        self.tts_voice_name = default_persona.get("tts_voice_name", TTS_VOICE_NAME)
        self.gemini_voice = default_persona.get("gemini_voice", None)
        self.gemini_locale = default_persona.get("gemini_locale", None)
        self.gemini_model = default_persona.get("gemini_model", None)
        self.gemini_tts_voice_name = default_persona.get("gemini_tts_voice_name", None)
        self.gemini_tts_model = default_persona.get("gemini_tts_model", None)

    @staticmethod
    def read_file(file_path: str) -> str:
        with open(file_path, "r") as file:
            return file.read()

    @staticmethod
    def read_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    @staticmethod
    def read_audio_file(audio_file_path: str) -> bytes:
        with open(audio_file_path, "rb") as audio_file:
            return audio_file.read()

    @staticmethod
    def format_streamlit_image_to_base64(streamlit_file_object):
        bytes_data = streamlit_file_object.read()
        b64_bytes = base64.b64encode(bytes_data)
        b64_str = b64_bytes.decode("utf-8")
        mime = "image/png" if streamlit_file_object.type == "image/png" else "image/jpeg"
        return f"data:{mime};base64,{b64_str}"

    def drop_memory(self):
        self.history = [
            {
                "role": "system",
                "content": ConversationAgent.read_file("./context.txt"),
            }
        ]

    def transcribe_audio(self, audio_bytes: bytes) -> str:
        audio_file_buffer = BytesIO(audio_bytes)
        audio_file_buffer.name = "audio.mp3"

        transcript = self.groq_client.audio.transcriptions.create(
            file=audio_file_buffer,
            model=STT_MODEL,
            language="fr",
        )

        return transcript.text

    def ask_model(self, message: str) -> Generator[str, None, None]:
        self.history.append({"role": "user", "content": message})

        groq_stream = self.groq_client.chat.completions.create(
            messages=self.history,
            model=LLM_MODEL,
            temperature=0.5,
            max_completion_tokens=LLM_MAX_COMPLETION_TOKENS,
            top_p=1,
            stop=None,
            stream=True,
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

        self.history.append({"role": "assistant", "content": "".join(list_of_sentences)})

    def speak(self, text: str, save: bool = False) -> bytes:
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code=getattr(self, "tts_language_code", TTS_LANGUAGE_CODE),
            name=getattr(self, "tts_voice_name", TTS_VOICE_NAME),
        )
        audio_encoding = texttospeech.AudioEncoding.MP3 if TTS_AUDIO_ENCODING == "MP3" else texttospeech.AudioEncoding.LINEAR16
        audio_config = texttospeech.AudioConfig(audio_encoding=audio_encoding)

        response = self.tts_client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config,
        )

        if save:
            os.makedirs("./audio_samples/", exist_ok=True)
            with open("./audio_samples/answer.mp3", "ab") as audio_file:
                audio_file.write(response.audio_content)

        return response.audio_content

    def speak_stream(self, text: str) -> Generator[bytes, None, None]:
        streaming_voice = texttospeech.VoiceSelectionParams(
            language_code=getattr(self, "tts_language_code", TTS_LANGUAGE_CODE),
            name=getattr(self, "tts_voice_name", TTS_VOICE_NAME),
        )

        sample_rate = 24000
        streaming_config = texttospeech.StreamingSynthesizeConfig(
            voice=streaming_voice,
            streaming_audio_config=texttospeech.StreamingAudioConfig(
                audio_encoding=texttospeech.AudioEncoding.PCM,
                sample_rate_hertz=sample_rate,
            ),
        )

        config_request = texttospeech.StreamingSynthesizeRequest(streaming_config=streaming_config)

        def request_generator():
            yield config_request
            yield texttospeech.StreamingSynthesizeRequest(
                input=texttospeech.StreamingSynthesisInput(text=text)
            )

        streaming_responses = self.tts_client.streaming_synthesize(request_generator())

        for response in streaming_responses:
            if not getattr(response, "audio_content", None):
                continue

            pcm_bytes = response.audio_content
            buffer = BytesIO()
            with wave.open(buffer, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(pcm_bytes)

            yield buffer.getvalue()

    def speak_stream_voice_clone(
        self,
        voice_cloning_key: str,
        text_chunks: list,
        sample_rate_hz: int = 24000,
        output_wav_path: str | None = None,
    ) -> Generator[bytes, None, None]:
        try:
            voice_clone_params = texttospeech.VoiceCloneParams(
                voice_cloning_key=voice_cloning_key
            )

            streaming_config = texttospeech.StreamingSynthesizeConfig(
                voice=texttospeech.VoiceSelectionParams(
                    language_code=getattr(self, "tts_language_code", TTS_LANGUAGE_CODE),
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
                with wave.open(output_wav_path, "wb") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(sample_rate_hz)
                    wav_file.writeframes(audio_buffer.getvalue())
        except Exception:
            raise

    def speak_stream_gemini(
        self,
        prompt: str | None,
        text_chunks: list[str],
        model: str | None = None,
        voice: str | None = None,
        locale: str | None = None,
        sample_rate_hz: int = 24000,
    ) -> Generator[bytes, None, None]:
        client = self.tts_client
        model = model or getattr(self, "gemini_model", None)
        voice = voice or getattr(self, "gemini_voice", None)
        locale = locale or getattr(self, "gemini_locale", None)

        config_request = texttospeech.StreamingSynthesizeRequest(
            streaming_config=texttospeech.StreamingSynthesizeConfig(
                voice=texttospeech.VoiceSelectionParams(
                    name=voice,
                    language_code=locale,
                    model_name=model,
                ),
                streaming_audio_config=texttospeech.StreamingAudioConfig(
                    audio_encoding=texttospeech.AudioEncoding.PCM,
                    sample_rate_hertz=sample_rate_hz,
                ),
            )
        )

        def request_generator():
            yield config_request
            for text in text_chunks:
                yield texttospeech.StreamingSynthesizeRequest(
                    input=texttospeech.StreamingSynthesisInput(text=text)
                )

        streaming_responses = client.streaming_synthesize(request_generator())

        emitted = False
        for response in streaming_responses:
            if not getattr(response, "audio_content", None):
                continue

            emitted = True
            pcm_bytes = response.audio_content
            buffer = BytesIO()
            with wave.open(buffer, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate_hz)
                wav_file.writeframes(pcm_bytes)

            yield buffer.getvalue()

        if not emitted:
            synthesis_input = texttospeech.SynthesisInput(text=" ".join(text_chunks))
            voice_params = texttospeech.VoiceSelectionParams(
                name=voice,
                language_code=locale,
            )
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.LINEAR16,
                sample_rate_hertz=sample_rate_hz,
            )
            response = client.synthesize_speech(
                input=synthesis_input,
                voice=voice_params,
                audio_config=audio_config,
            )
            yield response.audio_content

    def stream_persona_tts(
        self,
        sentences: list[str],
        voice_name: str | None = None,
        locale: str | None = None,
        audio_encoding: texttospeech.AudioEncoding = texttospeech.AudioEncoding.MP3,
    ) -> Generator[bytes, None, None]:
        client = self.tts_client
        voice_name = voice_name or getattr(self, "gemini_tts_voice_name", None) or getattr(self, "tts_voice_name", None)
        locale = locale or getattr(self, "gemini_locale", None) or getattr(self, "tts_language_code", None)

        for sentence in sentences:
            text = (sentence or "").strip()
            if not text:
                continue

            synthesis_input = texttospeech.SynthesisInput(text=text)
            voice_params = texttospeech.VoiceSelectionParams(
                name=voice_name,
                language_code=locale,
            )
            audio_config = texttospeech.AudioConfig(audio_encoding=audio_encoding)

            try:
                response = client.synthesize_speech(
                    input=synthesis_input,
                    voice=voice_params,
                    audio_config=audio_config,
                )
                if getattr(response, "audio_content", None):
                    yield response.audio_content
            except Exception:
                continue

    def try_pipeline(self, audio_file_path="./audio_samples/question.mp3"):
        print(f"Tentative de lecture du fichier : {audio_file_path}")
        if not os.path.exists(audio_file_path):
            print("ERREUR : Le fichier audio de la question n'existe pas. Créez-en un.")
            return

        if os.path.exists("./audio_samples/answer.mp3"):
            os.remove("./audio_samples/answer.mp3")

        audio_bytes = ConversationAgent.read_audio_file(audio_file_path)
        question_text = self.transcribe_audio(audio_bytes)

        print(f"→ Transcription : {question_text}")

        for sentence in self.ask_model(question_text):
            print("→ Phrase LLM reçue :", sentence)
            _ = self.speak(sentence, save=True)

        print("→ Réponse audio complète sauvegardée dans ./audio_samples/answer.mp3")
        self.drop_memory()

    def initiate_history(self):
        self.history = [
            {
                "role": "system",
                "content": ConversationAgent.read_file("./context.txt"),
            }
        ]

    def update_history(self, role, content):
        self.history.append({"role": role, "content": content})

    def get_history(self, type_model):
        if type_model == "large_language_model":
            filtred_history = []
            for message in self.history:
                if isinstance(message["content"], str):
                    filtred_history.append(message)
                elif isinstance(message["content"], list):
                    filtred_history.append(
                        {
                            "role": message["role"],
                            "content": f"{message['content'][0]['text']} : [IMAGE]",
                        }
                    )

            return filtred_history
        elif type_model == "vision_model":
            return self.history

    def ask_llm(self, user_interaction):
        self.update_history(role="user", content=user_interaction)
        response = self.groq_client.chat.completions.create(
            messages=self.get_history(type_model="large_language_model"),
            model=self.large_language_model,
        ).choices[0].message.content
        self.update_history(role="assistant", content=response)
        return response

    def ask_vision_model(self, user_interaction, image_b64):
        content = [
            {"type": "text", "text": user_interaction},
            {"type": "image_url", "image_url": {"url": f"{image_b64}"}},
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
