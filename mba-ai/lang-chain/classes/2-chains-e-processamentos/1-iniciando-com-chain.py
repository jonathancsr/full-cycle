from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

question_template = PromptTemplate(
    input_variables=["name"],
    template="Hi, I'm {name}! Tell me a joke with my name!"
)

model = ChatOpenAI(model="gpt-5-mini", temperature=0.01)

# A saida do question_template vai para a entrada do model
chain = question_template | model

result = chain.invoke({"name": "Jonathan"})
print(result.content)