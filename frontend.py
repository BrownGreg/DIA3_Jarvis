import streamlit
from chat_agent import ChatAgent 
from config import LLM_MODELS
import base64


if "chat_agent" not in streamlit.session_state :
	streamlit.session_state.chat_agent = ChatAgent()

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
	history_placeholder = streamlit.empty() 
	show_discussion_history(history_placeholder)
	with streamlit.container():
		
		user_input = streamlit.chat_input("N'oublie pas à qui tu parle !")
		uploaded_file = streamlit.file_uploader(
						"📎 Chargez une Image",
						type=["png", "jpg", "jpeg"],               # Autoriser tous les types ; précisez si besoin
						accept_multiple_files=False,
						key=streamlit.session_state.uploader_key
				)
		_, col2 = streamlit.columns([2, 1])
		with col2:
			streamlit.empty()
			streamlit.session_state.chat_agent.large_language_model = streamlit.selectbox("Choisis ton modèle gamin...", LLM_MODELS)

		if user_input:
			if uploaded_file:

				image_b64 = ChatAgent.format_streamlit_image_to_base64(streamlit_file_object=uploaded_file)
				response = streamlit.session_state.chat_agent.ask_vision_model(
					user_interaction=user_input,
					image_b64 = image_b64
					)

			else:
				streamlit.session_state.chat_agent.ask_llm(user_interaction=user_input)

			
			show_discussion_history(history_placeholder)
			streamlit.session_state.uploader_key += 1
			streamlit.rerun()


if __name__ == "__main__":
	user_interface()