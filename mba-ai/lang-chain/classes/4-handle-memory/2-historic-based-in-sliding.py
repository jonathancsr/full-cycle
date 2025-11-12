from typing import Any


from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory, RunnableLambda
from langchain_core.messages import trim_messages

load_dotenv()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that answers with a short joke when possible."),  # Mensagem do sistema que define o comportamento do assistente
    MessagesPlaceholder(variable_name="history"),  # variable_name: Nome da variável que será substituída pelo histórico de mensagens
    ("human", "{input}"),  # Mensagem do usuário com placeholder para a entrada
])

llm = ChatOpenAI(
    model="gpt-5-nano",  # Nome do modelo de linguagem a ser utilizado
    temperature=0.9  # Controla a aleatoriedade das respostas (0.0 = determinístico, 1.0 = muito criativo)
)

def prepare_inputs(payload: str) -> dict:  # payload: Dicionário contendo "raw_history" e "input" com o histórico e mensagem do usuário
    raw_history = payload.get("raw_history", [])
    trimmed = trim_messages(
        raw_history,  # Lista de mensagens do histórico a serem reduzidas
        token_counter=len,              # Função usada para contar "tokens", aqui contando o número de mensagens na lista (poderia ser quantidade de tokens, mas está como len)
        max_tokens=2,                   # Quantidade máxima de mensagens (ou "tokens" conforme a função acima) que serão mantidas no histórico
        strategy="last",                # Estratégia para selecionar as mensagens: "last" pega as mensagens mais recentes do histórico
        start_on="human",               # Define que a seleção deve começar por uma mensagem enviada pelo humano (usuário)
        include_system=True,            # Inclui a mensagem do sistema no histórico reduzido, se presente
        allow_partial=False             # Se true, permite incluir partes de uma mensagem caso não caiba inteira; aqui está como False (não inclui parcialmente)
    )
    return {"input": payload.get("input", ""), "history": trimmed}


prepare = RunnableLambda[str, dict](prepare_inputs)  # Converte a função prepare_inputs em um Runnable do LangChain
chain = prepare | prompt | llm

session_store: dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:  # session_id: Identificador único da sessão de conversa
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]



conversational_chain = RunnableWithMessageHistory(
    chain,  # Chain que será executada com histórico de mensagens
    get_session_history,  # Função que retorna o histórico de mensagens para uma sessão específica
    input_messages_key="input",  # Chave no dicionário de entrada que contém a mensagem do usuário
    history_messages_key="raw_history"  # Chave no dicionário de entrada que contém o histórico de mensagens antes do trimming
)

config = {"configurable": {"session_id": "demo-session"}}  # Configuração contendo o ID da sessão para manter o histórico

# Interactions
response1 = conversational_chain.invoke(
    {"input": "Hello, my name is Jonathan. Reply only with 'Ok'and do not mention my name"},  # Dicionário contendo a mensagem de entrada do usuário
    config=config  # Configuração contendo o ID da sessão
)
print("Assistant : ", response1.content)
print("-"*30)

response2 = conversational_chain.invoke(
    {"input": "Tell me a one-sentence fun fact. Do not mention my name."},  # Dicionário contendo a mensagem de entrada do usuário
    config=config  # Configuração contendo o ID da sessão
)
print("Assistant : ", response2.content)
print("-"*30)

response3 = conversational_chain.invoke(
    {"input": "What is my name?"},  # Dicionário contendo a mensagem de entrada do usuário
    config=config  # Configuração contendo o ID da sessão
)
print("Assistant : ", response3.content)
print("-"*30)

# Como utilizamos max_tokens como 2, ao mandar 3 mensagens o trim_message precisa remover a primeira mensagem que possui o meu nome, assim ele perde o contexto