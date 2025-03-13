from imports import *

load_dotenv()

NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

class Neo4jAuraDB:
    """
    A class to manage connections and queries to a Neo4j AuraDB instance.
    """

    def __init__(self, uri, user, password):
        """
        Initializes the Neo4jAuraDB instance.

        Args:
            uri: The URI of the AuraDB instance.
            user: The username for authentication.
            password: The password for authentication.
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.parsed_articles = []
        self.driver = self._connect()
        self.ner = CustomNer()

    def _connect(self):
        """
        Connects to the Neo4j AuraDB instance.

        Returns:
            A GraphDatabase driver object, or None if connection fails.
        """
        try:
            driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            driver.verify_connectivity()
            print("Connection successful!")
            return driver
        except Exception as e:
            print(f"Error connecting to AuraDB: {e}")
            return None
    
    def get_driver_info(self):
        return self.driver

    def execute_query(self, query, parameters=None):
        """
        Executes a Cypher query against the Neo4j database.

        Args:
            query: The Cypher query to execute.
            parameters: Optional parameters for the query.

        Returns:
            A list of result records, or None if an error occurs.
        """
        if self.driver is None:
            print("Driver is not initialized.")
            return None

        try:
            with self.driver.session() as session:
                if parameters:
                    results = session.run(query, parameters)
                else:
                    results = session.run(query)
                return [record for record in results]

        except Exception as e:
            print(f"Error executing query: {e}")
            return None

    def store_relationships_auradb(self, relationships):
        """Stores extracted relationships in Neo4j AuraDB, ensuring uniqueness."""
        for relationship in relationships:
            subject_text = relationship["entity1"]["text"]
            subject_label = relationship["entity1"]["label"]
            relationship_type = relationship["relation"]
            object_text = relationship["entity2"]["text"]
            object_label = relationship["entity2"]["label"]

            # Use the execute_query method to run Cypher queries with parameters
            self.driver.execute_query(
                f"MERGE (s:{subject_label} {{name: $subject_text}})",
                {"subject_text": subject_text}
            )
            self.driver.execute_query(
                f"MERGE (o:{object_label} {{name: $object_text}})",
                {"object_text": object_text}
            )
            self.driver.execute_query(
                f"MATCH (s:{subject_label} {{name: $subject_text}}) "
                f"MATCH (o:{object_label} {{name: $object_text}}) "
                f"MERGE (s)-[:{relationship_type}]->(o)",
                {"subject_text": subject_text, "object_text": object_text}
            )
        
    def store_articles_to_neo4j(self, articles):
        for article in articles:
            parsed_article = self.ner.parse_article(article)
            entities = self.ner.extract_entities(parsed_article)
            relationships = self.ner.extract_relationships(parsed_article, entities)
            # self.store_relationships_auradb(relationships)
            print(relationships)
            self.close()
            self.parsed_articles.append(parsed_article)
            
    def get_parsed_articles(self):
        return self.parsed_articles
            
    # Fetch related nodes and relationships from Neo4j
    def fetch_related_nodes(self, key_entities):
        """
        Fetches related nodes from Neo4j based on a list of key entities.

        Args:
            key_entities: A list of entity names to find related nodes for.

        Returns:
            A list of dictionaries, where each dictionary represents a relationship.
        """
        query = """
        MATCH (n)-[r]->(m)
        WHERE n.name IN $key_entities
        AND (n:GPE OR n:PERSON OR n:ORG)
        RETURN DISTINCT n.name AS source, type(r) AS relationship, m.name AS target, labels(m) AS target_labels 
        LIMIT 50;
        """
        try:
            with self.driver.session() as session:
                results = session.run(query, key_entities=key_entities)
                return [
                    {
                        "source": r["source"],
                        "relationship": r["relationship"],
                        "target": r["target"],
                        "target_labels": r["target_labels"]
                    }
                    for r in results
                ]
        except Exception as e:
            print(f"Error fetching related nodes: {e}")
            return [] # Return an empty list in case of error.
    
    def close(self):
        """
        Closes the Neo4j driver connection.
        """
        if self.driver:
            self.driver.close()
