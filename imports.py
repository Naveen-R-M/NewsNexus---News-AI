import gc
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import pickle
import re


import faiss
import google.generativeai as genai
import numpy as np
import spacy
import spacy_transformers
from dotenv import load_dotenv
from langchain.chains import GraphCypherQAChain, RetrievalQA
from langchain_community.vectorstores import Neo4jVector
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.docstore.document import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain.prompts import (
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_neo4j import Neo4jGraph
from langchain_ollama import OllamaEmbeddings
import ollama
from neo4j import GraphDatabase
import newspaper
from newspaper import Article, Config
from tqdm import tqdm

from custom_ner import CustomNer
from faiss_lib import VectorDB
from graphrag_workflow import GraphRagWorkflow
from neo4j_auradb import Neo4jAuraDB