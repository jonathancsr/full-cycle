from langchain.prompts import PromptTemplate
from langchain_core.runnables import chain
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

@chain
def square(input_x: dict) -> dict:  # input_x: Dicionário contendo os valores de entrada, espera a chave "x"
    return {"square_result": input_x["x"] * input_x["x"]}

question_template = PromptTemplate(
    input_variables=["name"],  # Lista de variáveis que serão substituídas no template
    template="Hi, I'm {name}! Tell me a joke with my name!"  # Template de texto com placeholders para as variáveis
)

question_template2 = PromptTemplate(
    input_variables=["square_result"],  # Lista de variáveis que serão substituídas no template
    template="Tell me about the number {square_result}"  # Template de texto com placeholders para as variáveis
)

model = ChatOpenAI(
    model="gpt-5-mini",  # Nome do modelo de linguagem a ser utilizado
    temperature=0.5  # Controla a aleatoriedade das respostas (0.0 = determinístico, 1.0 = muito criativo)
)

# A saida do question_template vai para a entrada do model
chain = question_template | model
chain2 = square | question_template2 | model
result = chain2.invoke({"x": 3})  # Invoca a chain com um dicionário contendo o valor de "x" para calcular o quadrado

print(result.content)