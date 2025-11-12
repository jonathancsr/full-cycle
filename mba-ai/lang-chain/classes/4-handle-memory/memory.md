Gerenciamento de memoria

LLMs são Stateless -> Não guardam informação

- A conversa não é armazenada no modelo, precisamos enviar o histórico de mensagens.

LLM -> Memoria Interna(Componentes de baixo nível)

Histórico: curto prazo vs longo prazo

- Curto Prazo: Memória que é utilizada durante uma transação / conversa / processamento

  - Armazenada temporariamente
  - Armazenada em banco de dados (Menos comum)

- Curto "longo": Histórico da "conversa"

  - Armazenada em banco de dados
  - Possibilidade de restaurar o conteúdo anterior para continuar de onde parou
  - Ler esse histórico para ter contexto para continuar a conversa/processamento
  - Depois de um tempo fica gigantesco e difícil de gerenciar, as vezes ate estourando a janela de contexto, é recomendado utilizar a sumarização

-
