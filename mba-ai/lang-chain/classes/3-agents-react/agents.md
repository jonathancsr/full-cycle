Agentes de IA são software comum on o centro é a LLM

- Possui comportamentos, e pode decidir o que fazer.
- Precisamos munir de ferramentas os agentes de IA para que possam fazer mais tarefas(MCP)

ReAct(Reasoning, Action)

- Reage a uma linha de pensamento
- Age sobre o que foi pensando
- Observa o resultado e repete o ciclo

Exemplo:

User: Qual é a população atual do Brasil e quantos habitantes por km2?

--Ciclo 1--
Thought: Preciso buscar a populção atual do Brasil e sua área territorial para calcular a densidade demografica

Action: search["população do Brasil 2024"]

Observation: O Brasil tem aproximadamente 216 milhões de habitantes em 2024

--Ciclo 2--
Thought: Agora preciso da área territorial do Brasil para calcular a densidade.

Action: search["área territorial do Brasil km2"]

Observation: A área do Brasil é de 8.515.767 km2.

--Ciclo 3--
Thought: Agora posso calcular 216.000.000 / 8.515.767 = aproximadamente 25,4 hab/km2

Action: calculate[216000000 / 8515767 ]

Observation: O Brasil tem aproximadamente 216 milhões de habitantes (2024) e densidade demográfica de 25,4 hab/km2
