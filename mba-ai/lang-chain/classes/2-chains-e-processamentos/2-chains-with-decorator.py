from langchain.prompts import PromptTemplate
from langchain_core.runnables import chain
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

@chain
def square(input_x: dict) -> dict:
    return {"square_result": input_x["x"] * input_x["x"]}

question_template = PromptTemplate(
    input_variables=["name"],
    template="Hi, I'm {name}! Tell me a joke with my name!"
)

question_template2 = PromptTemplate(
    input_variables=["square_result"],
    template="Tell me about the number {square_result}"
)

model = ChatOpenAI(model="gpt-5-mini", temperature=0.5)

# A saida do question_template vai para a entrada do model
chain = question_template | model
chain2 = square | question_template2 | model
result = chain2.invoke({"x": 3})

print(result.content)