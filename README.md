# NewsNexus - News AI

**NewsNexus** is an advanced GraphRAG (Graph Retrieval-Augmented Generation) application designed to extract, connect, and summarize news insights from articles. By leveraging a knowledge graph powered by Neo4j, entity recognition with spaCy, and contextual embeddings via Ollama’s Nomic model, NewsNexus delivers real-time, relationship-rich news analysis.

---

## Features
- **Contextual News Retrieval**: Extracts and maps 50+ entities (people, organizations, locations) and their relationships per query from news articles.
- **Real-Time Summarization**: Combines graph traversal and semantic search to generate concise, relevant news summaries.
- **Scalable Knowledge Graph**: Stores and queries interconnected news data using Neo4j.
- **Smart Query Processing**: Analyzes user prompts with LangChain for precise, intent-driven results.

---

## Tech Stack
- **Neo4j**: Knowledge graph database for entity-relationship storage.
- **spaCy**: Named Entity Recognition (NER) to identify key entities in news text.
- **Ollama (Nomic Embeddings)**: Vector embeddings for semantic search and context retrieval.
- **LangChain**: Framework for prompt analysis and integration with generative AI.
- **Python**: Core language for scripting and orchestration.
