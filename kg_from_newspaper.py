from imports import *

load_dotenv()

NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
EMBEDDING_MODEL = "nomic-embed-text"
GEMINI_MODEL = "gemini-2.0-flash"

NEWS_URL = 'https://www.nytimes.com/'

config = Config()
config.memoize_articles = False
config.fetch_images = False
config.language = 'en'

genai.configure(api_key=GEMINI_API_KEY)
llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, api_key=GEMINI_API_KEY)

text_splitter = SemanticChunker(
    OllamaEmbeddings(model=EMBEDDING_MODEL), 
    breakpoint_threshold_type="percentile",
)
embed = OllamaEmbeddings(model=EMBEDDING_MODEL)
ner = CustomNer()
vector_db = VectorDB()

neo4jConnect = Neo4jAuraDB(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)
graphRAG = GraphRagWorkflow(embedding_model=EMBEDDING_MODEL)

news_paper = newspaper.build(NEWS_URL, config=config)
articles = news_paper.articles

len(articles)

category = news_paper.category_urls()
category

neo4jConnect.store_articles_to_neo4j(articles[:5])

query = 'Fetch news related to Trump'
key_entities = graphRAG.graphrag_search(query)
related_nodes = neo4jConnect.fetch_related_nodes(key_entities)
graph_context = "\n".join(
    [
        f"{r['source']} {r['relationship']} {r['target']} ({r['target_labels']})" 
        for r in related_nodes
    ]
)

parsed_articles = neo4jConnect.get_parsed_articles()
for parsed_article in parsed_articles:
    chunk_embeddings = graphRAG.chunk_and_embed_news(parsed_article)
    vector_db.add_embeddings(chunk_embeddings=chunk_embeddings)
    vector_db.save_index()
del vector_db
gc.collect()

vector_db = VectorDB()
vector_db.load_index("faiss_index.pkl")

query_text = "What is the news about Trump?"
query_embedding = graphRAG.generate_embeddings(query_text)
query_embedding = np.array(query_embedding).astype(np.float32)

relevant_chunks = vector_db.retrieve_relevant_chunks(query_embedding=query_embedding, top_k=5)
print(relevant_chunks)
