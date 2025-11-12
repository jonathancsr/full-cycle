from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from dotenv import load_dotenv
load_dotenv()

long_text = """
A Grécia é um país situado no sudeste da Europa, banhado pelos mares Mediterrâneo, Jônico e Egeu. Sua geografia é composta por um território continental marcado por cadeias montanhosas, vales férteis e uma extensa costa recortada, além de mais de duas mil ilhas, das quais cerca de duzentas são habitadas. Entre as mais conhecidas estão Creta, Rodes, Santorini e Mikonos. O clima mediterrâneo, caracterizado por verões quentes e secos e invernos suaves, influencia o modo de vida, a agricultura, a arquitetura e a alimentação da população local.
A Grécia é amplamente considerada o berço da civilização ocidental. Foi ali que se desenvolveram importantes pilares da cultura, ciência, política e filosofia que moldaram o mundo moderno. Durante o período da Grécia Antiga, especialmente entre os séculos VIII a.C. e VI a.C., surgiram as cidades-estado — chamadas de pólis — cada uma com sua organização própria. Entre elas, Atenas e Esparta tornaram-se as mais famosas. Atenas destacou-se como centro intelectual, artístico e democrático, sendo reconhecida como o berço da democracia, enquanto Esparta era uma sociedade militarizada, disciplinada e voltada para a força bélica.
A cultura grega foi marcada por nomes que até hoje ressoam como referências universais. Na filosofia, pensadores como Sócrates, Platão e Aristóteles estabeleceram princípios que fundamentaram o pensamento racional ocidental. Na literatura, Homero compôs Ilíada e Odisseia, obras épicas que narram aventuras heroicas e conflitos entre deuses e mortais. No teatro, autores como Ésquilo, Sófocles, Eurípides e Aristófanes criaram tragédias e comédias que exploravam temas humanos profundos, como o destino, a justiça, o amor e a honra.
A mitologia grega ocupa um lugar especial nessa herança cultural. Os gregos imaginavam um mundo em que os deuses habitavam o Monte Olimpo e interagiam constantemente com os seres humanos, influenciando seu destino. Zeus, Hera, Poseidon, Atena, Apolo, Afrodite, Ártemis e Dionísio são alguns dos membros desse panteão complexo, rico em narrativas simbólicas que buscavam explicar fenômenos naturais, emoções humanas e aspectos da existência. Essas histórias, transmitidas oralmente e depois registradas, continuam sendo fonte de inspiração para literatura, arte, cinema e filosofia até os dias de hoje.
No campo político, a criação da democracia ateniense representou um marco na história mundial. Embora restrita a cidadãos livres (excluindo mulheres, estrangeiros e escravos), a ideia de participação popular e votação em assembleias influenciou profundamente os sistemas governamentais posteriores, especialmente nas sociedades ocidentais.
Outro período marcante foi a expansão macedônica sob Alexandre, o Grande, no século IV a.C. Ele unificou as cidades gregas e conquistou um vasto império que se estendia até o Egito e o noroeste da Índia. Esse processo, conhecido como helenização, espalhou a língua e a cultura gregas pelo mundo, influenciando astronomia, medicina, engenharia, matemática e artes. Alexandre desempenhou papel fundamental na criação de uma identidade cultural compartilhada entre povos de diferentes regiões.
Na arquitetura, a Grécia legou ao mundo construções grandiosas e esteticamente harmoniosas. O Parthenon, localizado na Acrópole de Atenas, é um dos exemplos mais conhecidos. Sua estrutura combina proporções matemáticas precisas com um sentido de beleza que influenciou construções renascentistas e neoclássicas em diversos países. Além disso, os gregos desenvolveram três estilos arquitetônicos clássicos — dórico, jônico e coríntio — que ainda inspiram edifícios oficiais, museus, universidades e monumentos contemporâneos.
A Grécia moderna, por sua vez, conquistou a independência do Império Otomano no século XIX. Hoje, faz parte da União Europeia e possui uma economia cuja base inclui turismo, agricultura, navegação marítima e comércio. As tradições culturais permanecem vivas, manifestando-se em danças folclóricas, na música popular grega e na culinária, que valoriza azeite de oliva, queijo feta, vegetais frescos, peixes e especiarias. Lugares como Atenas, com sua mistura de modernidade e ruínas antigas, e ilhas como Santorini, com suas casas brancas à beira de penhascos vulcânicos, atraem milhões de visitantes todos os anos.
A Grécia, portanto, não é apenas um país de grande beleza natural; ela representa um dos pilares fundamentais da história humana. Seu legado permanece presente na linguagem, na arte, na política, na ciência, na filosofia e no pensamento crítico que sustentam as sociedades modernas. Estudar a Grécia é compreender um pouco de nós mesmos, pois muito do que pensamos, valorizamos e buscamos encontra raízes profundas naquela antiga civilização que floresceu há mais de dois milênios e continua viva no imaginário coletivo da humanidade.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,  # Tamanho máximo de cada chunk (em caracteres)
    chunk_overlap=70  # Quantidade de caracteres que se sobrepõem entre chunks consecutivos
)

parts = splitter.create_documents([long_text])  # Divide o texto longo em uma lista de documentos menores

# É importante para diminuirmos o numero de tokens assim diminuindo o custo
llm = ChatOpenAI(
    model="gpt-5-nano",  # Nome do modelo de linguagem a ser utilizado
    temperature=0  # Controla a aleatoriedade das respostas (0.0 = determinístico, 1.0 = muito criativo)
)
chain_summarize = load_summarize_chain(
    llm,  # Modelo de linguagem a ser usado para sumarização
    chain_type="stuff",  # Tipo de chain: "stuff" envia todos os documentos de uma vez
    verbose=False  # Se True, imprime informações detalhadas durante a execução
)
result = chain_summarize.invoke({"input_documents": parts})  # Invoca a chain de sumarização com os documentos divididos

print(result["output_text"])