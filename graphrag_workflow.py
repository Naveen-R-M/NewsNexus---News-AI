from imports import *


class GraphRagWorkflow:
    def __init__(self, embedding_model):
        self.EMBEDDING_MODEL = embedding_model
        self.text_splitter = SemanticChunker(
            OllamaEmbeddings(model=self.EMBEDDING_MODEL), 
            breakpoint_threshold_type="percentile",
        )
        self.embed = OllamaEmbeddings(model=self.EMBEDDING_MODEL)
        self.ner = CustomNer()

    def generate_embeddings(self, text):
        """Generate embeddings for the given text."""
        try:
            vector_embeddings = self.embed.embed_query(text)
            return vector_embeddings
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return None

    def chunk_and_embed_news(self, news_article_text):
        """
        Chunks a news article using semantic chunking and generates embeddings.

        Args:
            news_article_text: The text content of the news article.

        Returns:
            A list of dictionaries, where each dictionary contains the chunk text and its embedding.
        """

        try:
            # Create a Document object for LangChain
            document = Document(page_content=news_article_text)

            # Split the document into chunks
            chunks = self.text_splitter.split_documents([document])

            # Generate embeddings for each chunk
            chunk_embeddings = []
            for chunk in chunks:
                chunk_text = chunk.page_content
                response = ollama.embeddings(self.EMBEDDING_MODEL, chunk_text)
                embedding = response['embedding']
                chunk_embeddings.append({"text": chunk_text, "embedding": embedding})

            return chunk_embeddings

        except Exception as e:
            print(f"Error chunking and embedding: {e}")
            return None
        
    # Combine graph and vector data, generate response
    def graphrag_search(self, query):
        # Parse query for entity
        key_entities = self.ner.parse_query(query)
        return key_entities
        # Fetch graph data
        