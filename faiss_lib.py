from imports import *

class VectorDB:
    def __init__(self, embedding_dim=768):  # Ensure this matches your embedding model's output size
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.text_data = []

    def add_embeddings(self, chunk_embeddings):
        """Add chunk embeddings to the FAISS index."""
        embeddings = np.array(
            [
                entry["embedding"] for entry in chunk_embeddings
            ]
        ).astype(np.float32)
        self.index.add(embeddings)
        self.text_data.extend(
            [
                entry["text"] for entry in chunk_embeddings
            ]
        )

    def save_index(self, path="faiss_index.pkl"):
        """Save FAISS index and text data."""
        with open(path, "wb") as f:
            pickle.dump({"index": self.index, "texts": self.text_data}, f)

    def load_index(self, path="faiss_index.pkl"):
        """Load FAISS index and text data."""
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.index = data["index"]
            self.text_data = data["texts"]

    def search(self, query_embedding, top_k):
        """Search for the top K most relevant documents with error handling."""
        query_vector = np.array(query_embedding).astype(np.float32).reshape(1, -1)
        if query_vector.shape[1] != self.index.d:
            raise ValueError(f"Embedding size mismatch: Expected {self.index.d}, but got {query_vector.shape[1]}")

        top_k = min(top_k, self.index.ntotal)  # Avoid exceeding available data
        distances, indices = self.index.search(query_vector, top_k)
        if indices[0][0] == -1:
            return [("No relevant document found.", None)]

        return ((self.text_data[i], distances[0][j]) for j, i in enumerate(indices[0]))


    def retrieve_relevant_chunks(self, query_embedding, top_k=3):
        """Retrieve relevant document chunks based on a query."""
        top_k = min(top_k, self.index.ntotal)  # Avoid requesting more results than available
        results = list(self.search(query_embedding, top_k))
        relevant_chunks = [
            {"match_distance": distance, "text": text}
            for text, distance in results  # Unpack tuples from the list
        ]
        return relevant_chunks