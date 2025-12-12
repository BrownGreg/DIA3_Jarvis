LLM_MODELS = [
	"llama-3.1-8b-instant",
	"openai/gpt-oss-20b",
	"openai/gpt-oss-120b",
	"llama-3.3-70b-versatile",
	"moonshotai/kimi-k2-instruct-0905",
]

VISION_MODELS = [
	"meta-llama/llama-4-scout-17b-16e-instruct",
	"meta-llama/llama-4-maverick-17b-128e-instruct"
]

STT_MODEL = "whisper-large-v3"  # Modèle Whisper de Groq
LLM_MODEL = LLM_MODELS[0]  # Modèle LLM de Groq

TTS_LANGUAGE_CODE = "it-IT"  # Accent italien
TTS_VOICE_NAME = "it-IT-Standard-D"  # Voix masculine italienne
TTS_AUDIO_ENCODING = "MP3"