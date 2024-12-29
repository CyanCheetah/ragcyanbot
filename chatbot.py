from langchain_community.llms import LlamaCpp
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
import os
import logging
import time
import chromadb
from chromadb.config import Settings
from opensearch_client import OpenSearchClient
from typing import List, Dict, Any
import re
import spacy
import networkx as nx

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
_llm_instance = None
_opensearch_client = None

def get_llm():
    global _llm_instance
    if _llm_instance is None:
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, "pyllama", "models", "llama-2-7b-chat.Q4_K_M.gguf")
            
            _llm_instance = LlamaCpp(
                model_path=model_path,
                temperature=0.7,
                max_tokens=256,
                top_p=1,
                n_ctx=2048,
                n_gpu_layers=32,
                n_batch=512,
                verbose=True,
                f16_kv=True,
                repeat_penalty=1.1,
                use_metal=True,
                seed=42,
                main_gpu=0,
                tensor_split=[0],
                rope_freq_scale=0.5
            )
            logger.info("LLM initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}", exc_info=True)
            raise
    return _llm_instance

def get_opensearch_client():
    global _opensearch_client
    if _opensearch_client is None:
        try:
            _opensearch_client = OpenSearchClient()
            logger.info("OpenSearch client initialized successfully")
        except Exception as e:
            logger.warning(f"OpenSearch not available: {str(e)}. Will continue without OpenSearch.")
            _opensearch_client = None
    return _opensearch_client

class CyanBot:
    def __init__(self):
        try:
            # Get the current directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Create necessary directories with proper permissions
            os.makedirs("chroma_db", exist_ok=True)
            os.makedirs("documents", exist_ok=True)
            
            # Initialize components
            self.llm = get_llm()
            
            # Try to initialize OpenSearch, but continue if it fails
            try:
                self.opensearch_client = get_opensearch_client()
            except Exception as e:
                logger.warning(f"OpenSearch initialization failed: {str(e)}. Will continue without OpenSearch.")
                self.opensearch_client = None
            
            # Initialize spaCy
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy initialized successfully")
            
            # Initialize embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            logger.info("Embeddings initialized successfully")
            
            # Initialize memory
            self.memory = ConversationBufferWindowMemory(
                k=3,
                return_messages=True,
                memory_key="chat_history",
                input_key="question",
                output_key="answer"
            )
            
            # Initialize ChromaDB client with new configuration
            self.chroma_client = chromadb.PersistentClient(
                path=os.path.join(current_dir, "chroma_db"),
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Create custom prompt template
            qa_prompt = PromptTemplate(
                template="""You are CyanBot, a knowledgeable AI assistant that helps users understand and work with documents. Your responses should be based ONLY on the information provided in the context below. Always speak in first person.

Context:
{context}

Current Question: {question}
Previous Conversation:
{chat_history}

Instructions:
1. Answer ONLY based on the information in the provided context
2. If the answer cannot be found in the context, say "I don't have enough information in the provided documents to answer that question"
3. When answering, cite specific documents and sections you're drawing information from
4. For questions about document content, quote relevant passages when appropriate
5. Be direct and specific in your answers
6. Always use first person (e.g., "I am" instead of "You are")
7. Never make assumptions about the user or their preferences

Answer: Let me help you based on the available documents.""",
                input_variables=["context", "chat_history", "question"]
            )

            # Initialize vector store
            self._init_vectorstore()
            
            # Create the chain
            logger.info("Creating conversation chain...")
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": 6,
                        "fetch_k": 10,
                        "lambda_mult": 0.7
                    }
                ),
                memory=self.memory,
                return_source_documents=True,
                verbose=True,
                chain_type="stuff",
                combine_docs_chain_kwargs={
                    "prompt": qa_prompt
                }
            )
            logger.info("Chatbot initialization complete")
            
        except Exception as e:
            logger.error(f"Error in initialization: {str(e)}", exc_info=True)
            raise

    def _init_vectorstore(self):
        """Initialize or reinitialize the vector store"""
        try:
            # Delete existing collection if it exists
            try:
                self.chroma_client.delete_collection("rag_collection")
                logger.info("Deleted existing collection")
            except Exception as e:
                logger.info("No existing collection to delete")

            # Create new collection
            collection = self.chroma_client.create_collection("rag_collection")
            
            # Initialize vector store
            self.vectorstore = Chroma(
                client=self.chroma_client,
                collection_name="rag_collection",
                embedding_function=self.embeddings,
            )
            logger.info("Vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise

    def load_documents(self, directory_path):
        """Load documents from a directory and update the vector store."""
        try:
            # Clear existing documents from OpenSearch
            if self.opensearch_client:
                self.opensearch_client.delete_all_documents()
            
            # Clear existing ChromaDB collection
            self._init_vectorstore()
            
            # Load and process new documents
            loader = DirectoryLoader(
                directory_path,
                glob="**/*.txt",
                loader_cls=TextLoader
            )
            documents = loader.load()
            
            if not documents:
                logger.warning("No documents found in the specified directory.")
                return False
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(documents)
            
            # Add documents to vector store
            self.vectorstore.add_documents(chunks)
            
            # Index documents in OpenSearch
            if self.opensearch_client:
                for chunk in chunks:
                    self.opensearch_client.index_document(chunk)
            
            logger.info(f"Successfully loaded and indexed {len(documents)} documents.")
            return True
            
        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
            return False

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text using spaCy"""
        doc = self.nlp(text)
        # Extract nouns, proper nouns, and compound nouns
        keywords = []
        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN'] and not token.is_stop:
                keywords.append(token.text.lower())
            # Get compound nouns
            if token.dep_ == 'compound' and token.head.pos_ in ['NOUN', 'PROPN']:
                keywords.append((token.text + ' ' + token.head.text).lower())
        return list(set(keywords))

    def get_response(self, query):
        """Get a response based on the query using RAG."""
        try:
            # Get relevant documents from vector store
            docs = self.vectorstore.similarity_search(query, k=3)
            
            # Get relevant documents from OpenSearch
            opensearch_docs = []
            if self.opensearch_client:
                opensearch_docs = self.opensearch_client.search_documents(query)
            
            # Combine and deduplicate documents
            all_docs = []
            seen_content = set()
            
            for doc in docs + opensearch_docs:
                content = doc.page_content if hasattr(doc, 'page_content') else doc['content']
                if content not in seen_content:
                    seen_content.add(content)
                    all_docs.append(doc)
            
            if not all_docs:
                return "I don't have enough context to answer that question. Please try asking something else or upload relevant documents."
            
            # Create context from documents
            context = "\n\n".join(
                doc.page_content if hasattr(doc, 'page_content') else doc['content']
                for doc in all_docs
            )
            
            # Create prompt
            prompt = f"""You are a helpful AI assistant. Answer the following question based ONLY on the provided context. 
            If you cannot find the answer in the context, say so clearly.
            Always respond in the first person using "I" statements.
            
            Context: {context}
            
            Question: {query}
            
            Answer: """
            
            # Get response from language model
            response = self.llm(prompt)
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error getting response: {str(e)}")
            return "I encountered an error while processing your question. Please try again."

    def get_keyword_graph(self) -> Dict[str, Any]:
        """Generate a keyword graph from the documents"""
        try:
            # Get all documents from ChromaDB
            docs = self.vectorstore.get()
            if not docs or not docs['documents']:
                logger.warning("No documents found in vector store")
                return {'nodes': [], 'edges': []}

            # Create graph
            G = nx.Graph()
            keyword_pairs = []

            # Extract keywords from each document
            for doc in docs['documents']:
                try:
                    # Extract keywords using spaCy
                    keywords = self._extract_keywords(doc)
                    # Create pairs of keywords that appear in the same document
                    for i in range(len(keywords)):
                        for j in range(i + 1, len(keywords)):
                            keyword_pairs.append((keywords[i], keywords[j]))
                except Exception as e:
                    logger.error(f"Error processing document for graph: {str(e)}")
                    continue

            # Add edges with weights
            for pair in keyword_pairs:
                if G.has_edge(pair[0], pair[1]):
                    G[pair[0]][pair[1]]['weight'] += 1
                else:
                    G.add_edge(pair[0], pair[1], weight=1)

            # Convert to vis.js format
            nodes = []
            edges = []
            
            # Add nodes
            for node in G.nodes():
                nodes.append({
                    'id': node,
                    'label': node,
                    'value': G.degree(node),  # Node size based on connections
                    'title': f"Keyword: {node}\nConnections: {G.degree(node)}"  # Tooltip
                })
            
            # Add edges
            for edge in G.edges(data=True):
                edges.append({
                    'from': edge[0],
                    'to': edge[1],
                    'value': edge[2].get('weight', 1),  # Edge thickness based on weight
                    'title': f"Weight: {edge[2].get('weight', 1)}"  # Tooltip
                })

            logger.info(f"Generated graph with {len(nodes)} nodes and {len(edges)} edges")
            return {
                'nodes': nodes,
                'edges': edges
            }
            
        except Exception as e:
            logger.error(f"Error generating keyword graph: {str(e)}")
            return {'nodes': [], 'edges': []} 