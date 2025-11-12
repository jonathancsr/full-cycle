from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory

load_dotenv()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),  # Mensagem do sistema que define o comportamento do assistente
    MessagesPlaceholder(variable_name="history"),  # variable_name: Nome da variável que será substituída pelo histórico de mensagens
    ("human", "{input}"),  # Mensagem do usuário com placeholder para a entrada
])

chat_model = ChatOpenAI(
    model="gpt-5-nano",  # Nome do modelo de linguagem a ser utilizado
    temperature=0.9  # Controla a aleatoriedade das respostas (0.0 = determinístico, 1.0 = muito criativo)
)

chain = prompt | chat_model

session_store: dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:  # session_id: Identificador único da sessão de conversa
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]

conversational_chain = RunnableWithMessageHistory(
    chain,  # Chain que será executada com histórico de mensagens
    get_session_history,  # Função que retorna o histórico de mensagens para uma sessão específica
    input_messages_key="input",  # Chave no dicionário de entrada que contém a mensagem do usuário
    history_messages_key="history"  # Chave no dicionário de entrada que contém o histórico de mensagens
)

config = {"configurable": {"session_id": "demo-session"}}  # Configuração contendo o ID da sessão para manter o histórico


print("Testing without session and historic saved")
message = chat_model.invoke("Hey my name is Jonathan")  # Invoca o modelo com uma mensagem de texto simples
print("Assistant : ", message.content)
print("-"*30)

message = chat_model.invoke("Can you repeat my name?")  # Invoca o modelo com uma mensagem de texto simples
print("Assistant : ", message.content)
print("-"*30)


print("Testing with session and historic saved")

# Interactions
response1 = conversational_chain.invoke(
    {"input": "Hello, my name is Jonathan. How are you?"},  # Dicionário contendo a mensagem de entrada do usuário
    config=config  # Configuração contendo o ID da sessão
)
print("Assistant : ", response1.content)
print("-"*30)

response2 = conversational_chain.invoke(
    {"input": "Can you repeat my name?"},  # Dicionário contendo a mensagem de entrada do usuário
    config=config  # Configuração contendo o ID da sessão
)
print("Assistant : ", response2.content)
print("-"*30)

response3 = conversational_chain.invoke(
    {"input": "Can you repeat my name in a motivation speech?"},  # Dicionário contendo a mensagem de entrada do usuário
    config=config  # Configuração contendo o ID da sessão
)
print("Assistant : ", response3.content)
print("-"*30)
