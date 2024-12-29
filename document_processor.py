import os
import logging
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from opensearchpy import OpenSearch
from typing import List, Dict, Any
from collections import Counter
import spacy
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        """Initialize document processor with OpenSearch and NLP capabilities"""
        try:
            # Initialize OpenSearch client
            self.opensearch = OpenSearch(
                hosts=[{'host': os.getenv('OPENSEARCH_HOST', 'localhost'), 
                       'port': int(os.getenv('OPENSEARCH_PORT', 9200))}],
                http_auth=(os.getenv('OPENSEARCH_USER', 'admin'), 
                          os.getenv('OPENSEARCH_PASSWORD', 'admin')),
                use_ssl=True,
                verify_certs=False,
                ssl_show_warn=False
            )
            
            # Initialize spaCy for NLP
            self.nlp = spacy.load("en_core_web_sm")
            
            # Create index if it doesn't exist
            self._create_index()
            
            logger.info("Document processor initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing document processor: {str(e)}")
            raise

    def _create_index(self):
        """Create OpenSearch index with appropriate mappings"""
        index_name = "documents"
        
        # Define index mapping
        mapping = {
            "mappings": {
                "properties": {
                    "content": {"type": "text"},
                    "title": {"type": "text"},
                    "author": {"type": "text"},
                    "keywords": {"type": "keyword"},
                    "creation_date": {"type": "date", "format": "strict_date_optional_time||epoch_millis"},
                    "file_path": {"type": "keyword"},
                    "file_type": {"type": "keyword"},
                    "page_number": {"type": "integer"},
                    "embedding": {"type": "dense_vector", "dims": 384}  # For semantic search
                }
            }
        }
        
        try:
            if not self.opensearch.indices.exists(index_name):
                self.opensearch.indices.create(index=index_name, body=mapping)
                logger.info(f"Created index: {index_name}")
        except Exception as e:
            logger.error(f"Error creating index: {str(e)}")

    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process a document and extract metadata, content, and keywords"""
        try:
            # Read the file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Process the content with spaCy
            doc = self.nlp(content)
            
            # Extract keywords (nouns and proper nouns)
            keywords = [token.text.lower() for token in doc 
                       if not token.is_stop and not token.is_punct 
                       and token.pos_ in ['NOUN', 'PROPN']]
            
            # Create document info
            document_info = {
                'content': content,
                'file_path': file_path,
                'file_type': os.path.splitext(file_path)[1].lower(),
                'keywords': list(set(keywords)),
                'title': os.path.basename(file_path),
                'creation_date': None
            }
            
            # Index document in OpenSearch
            self.opensearch.index(
                index="documents",
                body=document_info,
                refresh=True
            )
            
            return document_info
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise

    def _index_document(self, document_info: Dict[str, Any]):
        """Index document in OpenSearch"""
        try:
            # Prepare document for indexing
            for page in document_info['pages']:
                doc = {
                    'content': page['content'],
                    'title': document_info.get('title', ''),
                    'author': document_info.get('author', ''),
                    'keywords': page['keywords'],
                    'entities': page['entities'],
                    'file_path': document_info['file_path'],
                    'file_type': document_info['file_type'],
                    'page_number': page['page_number'],
                    'creation_date': document_info.get('creation_date', '')
                }
                
                self.opensearch.index(
                    index="documents",
                    body=doc,
                    id=f"{document_info['file_path']}_{page['page_number']}"
                )
            
            logger.info(f"Indexed document: {document_info['file_path']}")
            
        except Exception as e:
            logger.error(f"Error indexing document: {str(e)}")

    def generate_keyword_graph(self, file_path: str = None) -> Dict[str, Any]:
        """Generate keyword co-occurrence graph"""
        try:
            # Query OpenSearch for keywords
            query = {"match_all": {}} if not file_path else {
                "term": {"file_path": file_path}
            }
            
            response = self.opensearch.search(
                index="documents",
                body={"query": query},
                size=1000
            )
            
            # Create graph
            G = nx.Graph()
            keyword_pairs = []
            
            # Process keywords from search results
            for hit in response['hits']['hits']:
                keywords = hit['_source'].get('keywords', [])
                for i, kw1 in enumerate(keywords):
                    for kw2 in keywords[i+1:]:
                        keyword_pairs.append(tuple(sorted([kw1, kw2])))
            
            # Count co-occurrences
            edge_weights = Counter(keyword_pairs)
            
            # Add edges to graph
            for (kw1, kw2), weight in edge_weights.items():
                G.add_edge(kw1, kw2, weight=weight)
            
            # Convert to simple format
            nodes = [{"id": node, "label": node} for node in G.nodes()]
            edges = [{"source": source, "target": target, "weight": G[source][target]["weight"]} 
                    for source, target in G.edges()]

            return {
                "nodes": nodes,
                "edges": edges
            }
            
        except Exception as e:
            logger.error(f"Error generating keyword graph: {str(e)}")
            return {"nodes": [], "edges": []}

    def search_documents(self, query: str, file_path: str = None) -> List[Dict[str, Any]]:
        """Search documents using OpenSearch"""
        try:
            # Build search query
            search_query = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": ["content", "title", "keywords^2", "entities^2"]
                                }
                            }
                        ]
                    }
                },
                "highlight": {
                    "fields": {
                        "content": {},
                        "title": {},
                        "keywords": {}
                    }
                }
            }
            
            # Add file path filter if specified
            if file_path:
                search_query["query"]["bool"]["filter"] = [
                    {"term": {"file_path": file_path}}
                ]
            
            response = self.opensearch.search(
                index="documents",
                body=search_query,
                size=10
            )
            
            results = []
            for hit in response['hits']['hits']:
                source = hit['_source']
                result = {
                    'title': source.get('title', ''),
                    'author': source.get('author', ''),
                    'file_path': source.get('file_path', ''),
                    'page_number': source.get('page_number', ''),
                    'score': hit['_score'],
                    'highlights': hit.get('highlight', {}),
                    'keywords': source.get('keywords', [])
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return [] 