from groq import Groq
from dotenv import load_dotenv
import os
import base64
from config import LLM_MODELS, VISION_MODELS



class ChatAgent:
	def __init__(self):
		load_dotenv()
		self.groq_client = Groq(api_key=os.environ["GROQ_KEY"])
		self.initiate_history()
		self.large_language_model = LLM_MODELS[0]
		self.vision_model = VISION_MODELS[0]



	@staticmethod
	def read_file(file_path):
		with open(file_path, "r") as file:
			return file.read()



	@staticmethod
	def read_image(image_path):# Ps : pour corriger le bug cf la méthode en dessous :3

		with open(image_path, "rb") as image_file:
			return base64.b64encode(image_file.read()).decode('utf-8')



	@staticmethod
	def format_streamlit_image_to_base64(streamlit_file_object):
		bytes_data = streamlit_file_object.read()
		b64_bytes = base64.b64encode(bytes_data)
		b64_str = b64_bytes.decode("utf-8")
		mime = "image/png" if streamlit_file_object.type == "image/png" else "image/jpeg"
		return f"data:{mime};base64,{b64_str}"



	def initiate_history(self):
		self.history = [
			{
				"role": "system",
				"content": ChatAgent.read_file("./context.txt")
			}]



	def update_history(self, role, content):
		self.history.append(
			{
				"role": role,
				"content": content,
			})



	def get_history(self, type_model):# Il reste à coder le filtre sur l'historique pour enlever les images en base64
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