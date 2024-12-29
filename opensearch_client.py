from opensearchpy import OpenSearch
import spacy
import networkx as nx
import plotly.graph_objects as go
from typing import List, Dict, Any
import os
from dotenv import load_dotenv
import logging
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenSearchClient:
    def __init__(self):
        self.client = OpenSearch(
            hosts=[{'host': 'localhost', 'port': 9200}],
            use_ssl=False,
            verify_certs=False,
            ssl_show_warn=False
        )
        self.nlp = spacy.load('en_core_web_sm')
        self.index_name = 'documents'
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        self._create_index_if_not_exists()

    def _create_index_if_not_exists(self):
        if not self.client.indices.exists(self.index_name):
            index_body = {
                'settings': {
                    'index': {
                        'number_of_shards': 1,
                        'number_of_replicas': 0,
                        'knn': True,
                        'knn.algo_param.ef_search': 100
                    }
                },
                'mappings': {
                    'properties': {
                        'content': {'type': 'text'},
                        'title': {'type': 'text'},
                        'author': {'type': 'text'},
                        'keywords': {'type': 'keyword'},
                        'creation_date': {'type': 'date'},
                        'file_path': {'type': 'keyword'},
                        'file_type': {'type': 'keyword'},
                        'page_number': {'type': 'integer'},
                        'chapter': {'type': 'keyword'},
                        'embedding': {
                            'type': 'knn_vector',
                            'dimension': 384,
                            'method': {
                                'name': 'hnsw',
                                'space_type': 'l2',
                                'engine': 'nmslib',
                                'parameters': {
                                    'ef_construction': 128,
                                    'm': 24
                                }
                            }
                        }
                    }
                }
            }
            self.client.indices.create(index=self.index_name, body=index_body)

    def extract_keywords(self, text: str) -> List[str]:
        doc = self.nlp(text)
        keywords = []
        for token in doc:
            if not token.is_stop and not token.is_punct and token.pos_ in ['NOUN', 'PROPN']:
                keywords.append(token.text.lower())
        return list(set(keywords))

    def index_document(self, document):
        """Index a single document in OpenSearch."""
        try:
            # Extract content and metadata
            content = document.page_content if hasattr(document, 'page_content') else document['content']
            metadata = document.metadata if hasattr(document, 'metadata') else document
            
            # Extract keywords
            keywords = self.extract_keywords(content)
            
            # Get embedding
            embedding = self.embeddings.embed_query(content)
            
            # Create document body
            doc = {
                'content': content,
                'title': os.path.basename(metadata.get('source', '')),
                'file_path': metadata.get('source', ''),
                'file_type': os.path.splitext(metadata.get('source', ''))[1],
                'page_number': metadata.get('page', 0),
                'chunk_id': str(hash(content)),
                'embedding': embedding,
                'keywords': keywords
            }
            
            # Index the document
            self.client.index(
                index=self.index_name,
                body=doc,
                id=doc['chunk_id'],
                refresh=True
            )
            return True
            
        except Exception as e:
            logger.error(f"Error indexing document: {str(e)}")
            return False

    def search_documents(self, query: str, size: int = 10, filter_dict: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search documents using OpenSearch with optional filters"""
        try:
            logger.info(f"Searching for: {query}")
            
            # Build search query
            search_query = {
                "query": {
                    "bool": {
                        "should": [
                            {
                                "match": {
                                    "content": {
                                        "query": query,
                                        "boost": 2
                                    }
                                }
                            },
                            {
                                "match": {
                                    "title": {
                                        "query": query,
                                        "boost": 1.5
                                    }
                                }
                            },
                            {
                                "match": {
                                    "keywords": {
                                        "query": query,
                                        "boost": 1.5
                                    }
                                }
                            }
                        ],
                        "minimum_should_match": 1
                    }
                },
                "highlight": {
                    "fields": {
                        "content": {"number_of_fragments": 3},
                        "title": {},
                        "keywords": {}
                    }
                }
            }
            
            # Add filters if provided
            if filter_dict:
                for field, value in filter_dict.items():
                    if value is not None:
                        search_query["query"]["bool"]["filter"] = search_query["query"]["bool"].get("filter", [])
                        search_query["query"]["bool"]["filter"].append({"term": {field: value}})
            
            logger.info(f"Search query: {search_query}")
            response = self.client.search(
                index=self.index_name,
                body=search_query,
                size=size
            )
            logger.info(f"Found {len(response['hits']['hits'])} results")
            
            results = []
            for hit in response['hits']['hits']:
                source = hit['_source']
                result = {
                    'content': source.get('content', ''),
                    'title': source.get('title', ''),
                    'author': source.get('author', ''),
                    'file_path': source.get('file_path', ''),
                    'score': hit['_score'],
                    'highlights': hit.get('highlight', {}),
                    'keywords': source.get('keywords', [])
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return []

    def generate_keyword_graph(self) -> Dict[str, Any]:
        try:
            # Get all documents
            response = self.client.search(
                index=self.index_name,
                body={'query': {'match_all': {}}, 'size': 1000}
            )
            documents = [hit['_source'] for hit in response['hits']['hits']]

            # Create graph
            G = nx.Graph()
            keyword_pairs = []

            # Extract keywords and create pairs
            for doc in documents:
                keywords = doc.get('keywords', [])
                for i in range(len(keywords)):
                    for j in range(i + 1, len(keywords)):
                        keyword_pairs.append((keywords[i], keywords[j]))

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
                    'value': G.degree(node)  # Node size based on connections
                })
            
            # Add edges
            for edge in G.edges(data=True):
                edges.append({
                    'from': edge[0],
                    'to': edge[1],
                    'value': edge[2].get('weight', 1)  # Edge thickness based on weight
                })

            return {
                'nodes': nodes,
                'edges': edges
            }
            
        except Exception as e:
            logger.error(f"Error generating graph: {str(e)}")
            return {'nodes': [], 'edges': []}

    def bulk_index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Bulk index multiple documents in OpenSearch"""
        try:
            if not documents:
                return
                
            bulk_data = []
            for doc in documents:
                # Add index action
                bulk_data.append({"index": {"_index": "documents"}})
                # Add document
                bulk_data.append(doc)
            
            # Perform bulk indexing
            self.client.bulk(index="documents", body=bulk_data, refresh=True)
            
        except Exception as e:
            logger.error(f"Error bulk indexing documents: {str(e)}")
            raise 

    def delete_all_documents(self):
        """Delete all documents from the OpenSearch index."""
        try:
            # Delete all documents in the index
            self.client.delete_by_query(
                index=self.index_name,
                body={
                    "query": {
                        "match_all": {}
                    }
                }
            )
            print(f"Successfully deleted all documents from index {self.index_name}")
            return True
        except Exception as e:
            print(f"Error deleting documents: {str(e)}")
            return False

    def index_document(self, document):
        """Index a single document in OpenSearch."""
        try:
            # Extract content and metadata
            content = document.page_content if hasattr(document, 'page_content') else document['content']
            metadata = document.metadata if hasattr(document, 'metadata') else document
            
            # Extract keywords
            keywords = self.extract_keywords(content)
            
            # Get embedding
            embedding = self.embeddings.embed_query(content)
            
            # Create document body
            doc = {
                'content': content,
                'title': os.path.basename(metadata.get('source', '')),
                'file_path': metadata.get('source', ''),
                'file_type': os.path.splitext(metadata.get('source', ''))[1],
                'page_number': metadata.get('page', 0),
                'chunk_id': str(hash(content)),
                'embedding': embedding,
                'keywords': keywords
            }
            
            # Index the document
            self.client.index(
                index=self.index_name,
                body=doc,
                id=doc['chunk_id'],
                refresh=True
            )
            return True
            
        except Exception as e:
            logger.error(f"Error indexing document: {str(e)}")
            return False 