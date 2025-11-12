from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

question_template = PromptTemplate(
    input_variables=["name"],  # Lista de variáveis que serão substituídas no template
    template="Hi, I'm {name}! Tell me a joke with my name!"  # Template de texto com placeholders para as variáveis
)

model = ChatOpenAI(
    model="gpt-5-mini",  # Nome do modelo de linguagem a ser utilizado
    temperature=0.01  # Controla a aleatoriedade das respostas (0.0 = determinístico, 1.0 = muito criativo)
)

# A saida do question_template vai para a entrada do model
chain = question_template | model

result = chain.invoke({"name": "Jonathan"})  # Invoca a chain com um dicionário contendo os valores para as variáveis do template
print(result.content)