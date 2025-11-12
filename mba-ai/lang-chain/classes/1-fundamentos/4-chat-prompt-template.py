from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

system = ("system", "You are an assistant that answers questions in a {style} style")
user = ("user", "{question}")

chat_prompt = ChatPromptTemplate.from_messages([system, user])  # Cria um template de prompt de chat a partir de uma lista de mensagens

messages = chat_prompt.format_messages(
    style="funny",  # Valor para substituir {style} no template
    question="Who is Alan Turing?"  # Valor para substituir {question} no template
)

for msg in messages:
    print(f"{msg.type}: {msg.content}")

model = ChatOpenAI(
    model="gpt-5",  # Nome do modelo de linguagem a ser utilizado
    temperature=0.5  # Controla a aleatoriedade das respostas (0.0 = determinístico, 1.0 = muito criativo)
)

result = model.invoke(messages)  # Invoca o modelo com as mensagens formatadas e retorna a resposta
print(result.content)