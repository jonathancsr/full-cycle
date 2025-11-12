from langchain.prompts import PromptTemplate

template = PromptTemplate(
    input_variables=["name"],  # Lista de variáveis que serão substituídas no template
    template="Hi, I'm {name}! Tell me a joke with my name!"  # Template de texto com placeholders para as variáveis
)

text = template.format(name="Jonathan")  # Substitui a variável {name} no template pelo valor fornecido
print(text)