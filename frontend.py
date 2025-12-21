import streamlit
<<<<<<< HEAD
from conversation_agent import ConversationAgent 
from config import LLM_MODELS
=======
import streamlit as st
import streamlit.components.v1 as components
from chat_agent import ConversationAgent
from config import LLM_MODELS, PERSONAS, TTS_AUDIO_ENCODING, USE_GEMINI_STREAMING
import base64
import os
import re
from io import BytesIO
from pydub import AudioSegment
import asyncio

try:
	from streamlit_mic_recorder import mic_recorder
	MIC_AVAILABLE = True
except ImportError:
	MIC_AVAILABLE = False

streamlit.set_page_config(page_title="Jarvis", page_icon="🤖")

streamlit.markdown(
	"""
	<style>
		section[data-testid="stSidebar"] ~ div div[data-testid="stVerticalBlock"] > div:first-child {
			max-width: 900px;
			margin: 0 auto;
		}
	</style>
	""",
	unsafe_allow_html=True,
)

if "chat_agent" not in streamlit.session_state:
	streamlit.session_state.chat_agent = ConversationAgent()

if "messages" not in streamlit.session_state:
	streamlit.session_state.messages = []

if "tts_placeholder" not in streamlit.session_state:
	streamlit.session_state.tts_placeholder = streamlit.empty()

if "last_transcription" not in streamlit.session_state:
	streamlit.session_state.last_transcription = None

if "audio_js_placeholder" not in streamlit.session_state:
	streamlit.session_state.audio_js_placeholder = streamlit.empty()

if "last_upload_sig" not in streamlit.session_state:
	streamlit.session_state.last_upload_sig = None

if "pending_image_b64" not in streamlit.session_state:
	streamlit.session_state.pending_image_b64 = None

def stop_audio_in_browser():
	"""Stoppe la file audio côté navigateur et coupe la lecture."""
	js = """
	<script>
		if (window.ttsAudio) {
			window.ttsAudio.pause();
			window.ttsAudio.currentTime = 0;
		}
		window.ttsQ = [];
		window.ttsPlaying = false;
	</script>
	"""
	components.html(js)
>>>>>>> feature/conversational_ai


def clear_audio_queue():
	"""Vide la file audio queue-based et stoppe l'élément en cours."""
	js = """
	<script>
	(function() {
		window.q = [];
		window.audioQueue = [];
		window.audio_queue = [];
		if (window.audioPlayer) {
			try { window.audioPlayer.pause(); window.audioPlayer.currentTime = 0; } catch (e) {}
		}
		if (window.currentAudio) {
			try { window.currentAudio.pause(); window.currentAudio.currentTime = 0; } catch (e) {}
			try { window.currentAudio.remove(); } catch (e) {}
			window.currentAudio = null;
		}
		if (window.speechSynthesis && window.speechSynthesis.cancel) {
			try { window.speechSynthesis.cancel(); } catch (e) {}
		}
		window.is_playing = false;
		window.p = false;
	})();
	</script>
	"""
	components.html(js)


def configure_sidebar(agent: ConversationAgent):
	"""Configure la sidebar (persona + réglages modèles/TTS)."""
	if "persona" not in streamlit.session_state:
		streamlit.session_state.persona = "Jarvis"

	with streamlit.sidebar:
		persona_names = list(PERSONAS.keys())
		selected = streamlit.selectbox(
			"Choisis un capo",
			persona_names,
			index=persona_names.index(streamlit.session_state.persona) if streamlit.session_state.persona in persona_names else 0,
		)
		streamlit.session_state.persona = selected
		streamlit.session_state.selected_char = selected

		streamlit.markdown("### Réglages")
		selected_model = streamlit.selectbox("Choisis ton modèle gamin...", LLM_MODELS)
		if "tts_enabled" not in streamlit.session_state:
			streamlit.session_state.tts_enabled = False
		tts_enabled = streamlit.toggle("Lecture automatique (TTS)", value=streamlit.session_state.tts_enabled)
		streamlit.session_state.tts_enabled = tts_enabled
		agent.large_language_model = selected_model
		if not tts_enabled:
			components.html("<script>window.q=[];</script>")
			clear_audio_queue()
			stop_audio_in_browser()
		if streamlit.button("Stop TTS"):
			stop_audio_in_browser()

	return streamlit.session_state.persona


def render_header(persona_key: str):
	"""Affiche le header du personnage sélectionné."""
	display_name = PERSONAS.get(persona_key, {}).get("display_name", persona_key)
	persona_cfg = PERSONAS.get(persona_key, {})
	gender = persona_cfg.get("gender")
	cols = streamlit.columns([1, 8])
	gif_path = os.path.join("gifs", f"{persona_key.lower()}.gif")
	with cols[0]:
		if os.path.exists(gif_path):
			streamlit.image(gif_path, width=80)
		else:
			streamlit.write("")
	with cols[1]:
		if gender == "female":
			title = f"### {display_name} — ta capo préférée !"
		elif gender == "male":
			title = f"### {display_name} — ton capo préféré !"
		else:
			title = f"### {display_name} — ton·ta capo préféré·e !"
		streamlit.markdown(title)

def play_audio(audio_bytes: bytes):
	"""Lecture TTS via file JS window.q (anti-chevauchement)."""
	if not audio_bytes or not streamlit.session_state.get("tts_enabled", False):
		return

	b64 = base64.b64encode(audio_bytes).decode("utf-8")

	js = f"""
	<script>
	if(!window.q){{window.q=[];window.isPlaying=false;}}
	window.q.push("data:audio/mp3;base64,{b64}");
	function runQ(){{
	    if(window.q.length>0 && !window.isPlaying){{
	        window.isPlaying=true;
	        let a=new Audio(window.q.shift());
	        a.onended=()=>{{window.isPlaying=false;runQ();}};
	        a.play().catch(()=>{{window.isPlaying=false;runQ();}});
	    }}
	}}
	runQ();
	</script>
	"""
	components.html(js, height=0)


def default_tts_mime() -> str:
	encoding = (TTS_AUDIO_ENCODING or "").lower()
	if encoding == "mp3":
		return "audio/mpeg"
	if encoding == "wav":
		return "audio/wav"
	if encoding in {"ogg", "opus", "ogg_opus"}:
		return "audio/ogg"
	return "audio/mpeg"


def with_streamlit_ctx(fn):
	"""Wrap a callable so that the current Streamlit run context is attached in new threads."""
	ctx = get_script_run_ctx()
	def wrapper(*args, **kwargs):
		if ctx:
			try:
				_add_ctx(threading.current_thread(), ctx)
			except Exception:
				pass
		return fn(*args, **kwargs)
	return wrapper



def user_interface():
<<<<<<< HEAD
	init_header()
	history_placeholder = streamlit.empty() 
	show_discussion_history(history_placeholder)
	with streamlit.container():
		
		user_input = streamlit.chat_input("N'oublie pas à qui tu parle !")
		_, col2 = streamlit.columns([2, 1])
		with col2:
			streamlit.empty()
			selected_model = streamlit.selectbox("Choisis ton modèle gamin...", LLM_MODELS)

		if user_input:
			streamlit.session_state.conversation_agent.ask_llm(user_interaction=user_input, model=selected_model)
			show_discussion_history(history_placeholder)



=======
	agent = streamlit.session_state.chat_agent
	persona_key = configure_sidebar(agent)
	persona_cfg = PERSONAS.get(persona_key, {})

	previous_persona = streamlit.session_state.get('_previous_persona')
	if previous_persona != persona_key:
		streamlit.session_state['_previous_persona'] = persona_key
		agent.history = [{"role": "system", "content": None}]

	agent.persona_name = persona_cfg.get('display_name', persona_key)
	agent.persona_context = persona_cfg.get('context', agent.persona_context)
	agent.tts_language_code = persona_cfg.get('tts_language_code', agent.tts_language_code)
	agent.tts_voice_name = persona_cfg.get('tts_voice_name', agent.tts_voice_name)
	agent.gemini_voice = persona_cfg.get('gemini_voice', agent.gemini_voice)
	agent.gemini_locale = persona_cfg.get('gemini_locale', agent.gemini_locale)
	agent.gemini_model = persona_cfg.get('gemini_model', agent.gemini_model)
	agent.gemini_tts_model = persona_cfg.get('gemini_tts_model', getattr(agent, 'gemini_tts_model', None))
	agent.gemini_tts_voice_name = persona_cfg.get('gemini_tts_voice_name', getattr(agent, 'gemini_tts_voice_name', None))
	agent.gemini_tts_style_prompt = persona_cfg.get('gemini_tts_style_prompt', getattr(agent, 'gemini_tts_style_prompt', None))
	try:
		if agent.history and isinstance(agent.history[0], dict):
			agent.history[0]['content'] = agent.persona_context
	except Exception:
		pass

	render_header(persona_key)
	streamlit.divider()

	for msg in streamlit.session_state.get("messages", []):
		with streamlit.chat_message(msg["role"]):
			streamlit.markdown(msg["content"])

	response_placeholder = streamlit.empty()

	with streamlit.container():
		input_cols = streamlit.columns([1, 1, 8], gap="small")

		mic_result = None
		with input_cols[0]:
			if MIC_AVAILABLE:
				mic_result = mic_recorder(start_prompt="🎙️", stop_prompt="⏹️", key="recorder")
			else:
				streamlit.button("🎙️", disabled=True, help="Installer streamlit-mic-recorder pour activer le micro")

		with input_cols[1]:
			with streamlit.popover("🖼️"):
				img_file = streamlit.file_uploader(
					"Image",
					type=["png", "jpg", "jpeg", "webp", "bmp"],
					accept_multiple_files=False,
				)
				if img_file:
					try:
						image_b64 = agent.format_streamlit_image_to_base64(img_file)
						streamlit.session_state.pending_image_b64 = image_b64
						streamlit.info("Image prête pour le prochain message.")
					except Exception as exc:
						streamlit.error(f"Image impossible à préparer : {exc}")

		prompt = streamlit.session_state.pop("_pending_prompt", None)
		with input_cols[2]:
			user_prompt = streamlit.chat_input("N'oublie pas à qui tu parles !")
			if user_prompt:
				prompt = user_prompt

	if mic_result:
		try:
			data = mic_result if isinstance(mic_result, bytes) else None
			if data is None and isinstance(mic_result, dict) and mic_result.get("bytes"):
				data = mic_result.get("bytes")
			if data is not None:
				converted = BytesIO()
				AudioSegment.from_file(BytesIO(data)).export(converted, format="mp3")
				converted.seek(0)
				data = converted.read()
				transcription = agent.transcribe_audio(data)
				streamlit.session_state.last_transcription = transcription
				prompt = transcription
				streamlit.success("Transcription micro prête.")
		except Exception as exc:
			streamlit.error(f"Transcription micro impossible : {exc}")

	def sanitize(text: str) -> str:
		"""Retire les didascalies entre parenthèses pour éviter qu'elles soient lues."""
		if not text:
			return text
		clean = re.sub(r"\([^)]*\)", "", text)
		clean = clean.strip()
		return clean or text

	tts_enabled = streamlit.session_state.get("tts_enabled", False)

	def speak(text: str) -> tuple[bytes | None, str | None]:
		"""Génère l'audio pour la réponse et retourne (bytes, mime)."""
		if not tts_enabled:
			return None, None
		clean = sanitize(text)
		try:
			return agent.speak(clean), default_tts_mime()
		except Exception as exc:
			streamlit.warning(f"TTS indisponible : {exc}")
			return None, None

	if prompt:
		streamlit.session_state.messages.append({"role": "user", "content": prompt})
		with response_placeholder:
			with streamlit.chat_message("assistant"):
				placeholder = streamlit.empty()
				full_response = ""
				sentence_buf = ""
				image_b64 = streamlit.session_state.get("pending_image_b64")
				if image_b64:
					try:
						vision_reply = agent.ask_vision_model(prompt, image_b64)
						full_response = vision_reply
						placeholder.markdown(full_response)
					finally:
						streamlit.session_state.pending_image_b64 = None
				else:
					for chunk in agent.ask_model(prompt):
						full_response += chunk
						sentence_buf += chunk
						placeholder.markdown(full_response + "▌")
						if any(punct in sentence_buf for punct in [".", "!", "?"]):
							audio_bytes, _ = speak(sentence_buf)
							if audio_bytes:
								play_audio(audio_bytes)
							sentence_buf = ""
					placeholder.markdown(full_response)
		streamlit.session_state.messages.append({"role": "assistant", "content": full_response})


>>>>>>> feature/conversational_ai
if __name__ == "__main__":
	user_interface()