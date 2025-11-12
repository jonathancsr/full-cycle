from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(
    model="gpt-5-nano",  # Nome do modelo de linguagem a ser utilizado
    temperature=0.5  # Controla a aleatoriedade das respostas (0.0 = determinístico, 1.0 = muito criativo)
)
message = model.invoke("Hello World")  # Invoca o modelo com a mensagem de entrada e retorna a resposta
print(message.content)