from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv()

gemini = init_chat_model(
    model="gemini-2.5-flash",  # Nome do modelo de linguagem a ser utilizado
    model_provider="google_genai"  # Provedor do modelo (Google Generative AI)
)
answer = gemini.invoke("Hello World")  # Invoca o modelo com a mensagem de entrada e retorna a resposta
print(answer.content)