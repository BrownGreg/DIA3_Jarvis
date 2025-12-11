import streamlit
from conversation_agent import ConversationAgent 
from config import LLM_MODELS


if "conversation_agent" not in streamlit.session_state :
	streamlit.session_state.conversation_agent = ConversationAgent()


def init_header():
	streamlit.set_page_config(page_title="Jarvis", page_icon="🤖")
	streamlit.title("🤖 Jarvis ton baron préféré !")
	streamlit.write("Il est un peu enervé, fais attention à ce que tu racontes...")



def show_discussion_history(history_placeholder):
	container = history_placeholder.container()
	with container:
		for message in streamlit.session_state.conversation_agent.history:
			if message["role"] != "system":
				with streamlit.chat_message(message["role"]):
					streamlit.write(message["content"])


def user_interface():
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



if __name__ == "__main__":
	user_interface()