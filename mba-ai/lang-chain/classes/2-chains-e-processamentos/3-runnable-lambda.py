from langchain_core.runnables import RunnableLambda

def parse_number(text: str) -> int:  # text: String contendo o número a ser convertido para inteiro
    return int(text.strip())

parse_runnable = RunnableLambda(parse_number)  # Converte a função parse_number em um Runnable do LangChain

number = parse_runnable.invoke("123")  # Invoca o runnable com uma string contendo o número
print(number)