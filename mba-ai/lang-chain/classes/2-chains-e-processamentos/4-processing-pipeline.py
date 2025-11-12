from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()


template_translate = PromptTemplate(
    input_variables=["initial_text"],  # Lista de variáveis que serão substituídas no template
    template="Translate the following text to English: \n ```{initial_text}```"  # Template de texto com placeholders para as variáveis
)

template_summary = PromptTemplate(
    input_variables=["text"],  # Lista de variáveis que serão substituídas no template
    template="Summarize the following text in 4 words: \n ```{text}```"  # Template de texto com placeholders para as variáveis
)

llm_en = ChatOpenAI(
    model='gpt-5-mini',  # Nome do modelo de linguagem a ser utilizado
    temperature=0  # Controla a aleatoriedade das respostas (0.0 = determinístico, 1.0 = muito criativo)
)

translate = template_translate | llm_en | StrOutputParser()

pipeline = {"text": translate} | template_summary | llm_en | StrOutputParser()

result = pipeline.invoke({"initial_text": "LangChain é um framework para desenvolvimento de aplicações em IA"})  # Invoca o pipeline com o texto inicial a ser traduzido e resumido
print(result)

