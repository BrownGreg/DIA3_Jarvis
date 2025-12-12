import streamlit
import streamlit.components.v1 as components
from io import BytesIO
from pydub import AudioSegment
from audio_recorder_streamlit import audio_recorder
from chat_agent import ConversationAgent 
from config import LLM_MODELS, TTS_AUDIO_ENCODING
import base64


if "chat_agent" not in streamlit.session_state :
	streamlit.session_state.chat_agent = ConversationAgent()

if "uploader_key" not in streamlit.session_state:
    streamlit.session_state.uploader_key = 0


def init_header():
	streamlit.set_page_config(page_title="Jarvis", page_icon="🤖")
	streamlit.title("🤖 Jarvis ton baron préféré !")
	streamlit.write("Il est un peu enervé, fais attention à ce que tu racontes...")



def show_discussion_history(history_placeholder):

	container = history_placeholder.container()
	with container:
		for message in streamlit.session_state.chat_agent.history:
			if message["role"] != "system":
				with streamlit.chat_message(message["role"]):
					if type(message["content"]) == str:
						streamlit.write(message["content"])
					elif type(message["content"]) == list:
						textual_content = message["content"][0]["text"]
						image_b64 = message["content"][1]["image_url"]["url"]
						streamlit.write(textual_content)
						streamlit.image(image_b64)



def user_interface():
	init_header()
	tab_chat, tab_voice = streamlit.tabs(["Chat texte", " Chat voix"])

	with tab_chat:
		history_placeholder = streamlit.empty() 
		show_discussion_history(history_placeholder)
		with streamlit.container():
			
			user_input = streamlit.chat_input("N'oublie pas à qui tu parle !")
			uploaded_file = streamlit.file_uploader(
						"📎 Chargez une Image",
						type=["png", "jpg", "jpeg"],
						accept_multiple_files=False,
						key=streamlit.session_state.uploader_key
				)
			_, col2 = streamlit.columns([2, 1])
			with col2:
				streamlit.empty()
				streamlit.session_state.chat_agent.large_language_model = streamlit.selectbox("Choisis ton modèle gamin...", LLM_MODELS)

			if user_input:
				if uploaded_file:

					image_b64 = ConversationAgent.format_streamlit_image_to_base64(streamlit_file_object=uploaded_file)
					response = streamlit.session_state.chat_agent.ask_vision_model(
						user_interaction=user_input,
						image_b64 = image_b64
						)

				else:
					streamlit.session_state.chat_agent.ask_llm(user_interaction=user_input)

				
				show_discussion_history(history_placeholder)
				streamlit.session_state.uploader_key += 1
				streamlit.rerun()

	with tab_voice:
		streamlit.subheader("Tester STT → LLM → TTS")
		col_text, col_audio = streamlit.columns(2)
		with col_text:
			text_prompt = streamlit.text_area("Texte (optionnel si audio)", "")
		with col_audio:
			uploaded_audio = streamlit.file_uploader("Ou charge un audio (mp3/wav/m4a)", type=["mp3", "wav", "m4a"])
			mic_audio = audio_recorder(text="🎙️ Cliquez pour enregistrer / pause pour arrêter", recording_color="#f56565", neutral_color="#2b2b2b")
			if mic_audio:
				streamlit.session_state["mic_audio_bytes"] = mic_audio
				streamlit.success("Audio micro capturé. Prêt pour transcription.")
			elif "mic_audio_bytes" in streamlit.session_state:
				streamlit.info("Audio micro précédemment capturé sera utilisé si aucun fichier n'est fourni.")

		start_voice = streamlit.button("Lancer la réponse vocale")
		if start_voice:
			question_text = None
			mic_bytes = streamlit.session_state.get("mic_audio_bytes")
			if mic_bytes is not None:
				audio_bytes = mic_bytes
				question_text = streamlit.session_state.chat_agent.transcribe_audio(audio_bytes)
				streamlit.info(f"Transcription (micro) : {question_text}")
			elif uploaded_audio is not None:
				audio_bytes = uploaded_audio.read()
				question_text = streamlit.session_state.chat_agent.transcribe_audio(audio_bytes)
				streamlit.info(f"Transcription : {question_text}")
			elif text_prompt.strip():
				question_text = text_prompt.strip()
			else:
				streamlit.warning("Ajoute un texte ou un fichier audio.")
				return

			placeholder_text = streamlit.empty()
			cumulative = ""
			audio_segments = []
			export_format = "mp3" if TTS_AUDIO_ENCODING == "MP3" else "wav"

			# Init unique audio queue once per render (isolated in its iframe)
			components.html(
				"""
				<script>
				  window.jarvisQueue = window.jarvisQueue || [];
				  window.jarvisPlaying = window.jarvisPlaying || false;
				  window.jarvisAudio = window.jarvisAudio || new Audio();
				  window.jarvisPlayNext = window.jarvisPlayNext || function() {
				    if (window.jarvisPlaying) return;
				    if (!window.jarvisQueue.length) return;
				    const src = window.jarvisQueue.shift();
				    const audio = window.jarvisAudio;
				    window.jarvisPlaying = true;
				    audio.src = src;
				    audio.autoplay = true;
				    audio.onended = function(){ window.jarvisPlaying = false; window.jarvisPlayNext(); };
				    audio.onerror = function(){ window.jarvisPlaying = false; window.jarvisPlayNext(); };
				    audio.play().catch(function(){ window.jarvisPlaying = false; });
				  };
				</script>
				""",
				height=0,
				width=0,
				scrolling=False,
			)

			chunk_idx = 0
			for sentence in streamlit.session_state.chat_agent.ask_model(question_text):
				cumulative += sentence
				placeholder_text.markdown(cumulative)

				audio_bytes = streamlit.session_state.chat_agent.speak(sentence)
				try:
					segment = AudioSegment.from_file(BytesIO(audio_bytes), format=export_format)
				except Exception:
					segment = AudioSegment.from_raw(BytesIO(audio_bytes), sample_width=2, frame_rate=22050, channels=1)
				audio_segments.append(segment)

				# Enqueue chunk into the persistent queue (single audio element)
				b64_audio = BytesIO()
				segment.export(b64_audio, format=export_format)
				b64_audio.seek(0)
				b64_str = base64.b64encode(b64_audio.read()).decode("utf-8")
				mime = f"audio/{export_format}"
				components.html(
					f"""
					<script>
					  window.jarvisQueue = window.jarvisQueue || [];
					  window.jarvisQueue.push('data:{mime};base64,{b64_str}');
					  window.jarvisPlayNext && window.jarvisPlayNext();
					</script>
					""",
					height=0,
					width=0,
					scrolling=False,
				)
				chunk_idx += 1

			if audio_segments:
				merged_audio = audio_segments[0]
				for seg in audio_segments[1:]:
					merged_audio += seg

				buffer = BytesIO()
				merged_audio.export(buffer, format=export_format)
				buffer.seek(0)
				streamlit.audio(buffer.getvalue(), format=f"audio/{export_format}")
				streamlit.download_button(
					"Télécharger la réponse audio",
					data=buffer.getvalue(),
					file_name=f"reponse.{export_format}",
					mime=f"audio/{export_format}"
				)


if __name__ == "__main__":
	user_interface()