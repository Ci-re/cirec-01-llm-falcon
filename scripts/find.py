# Imports
import pandas as pd
from minsearch import Index
from scripts.gather import load_data
from elasticsearch import Elasticsearch
import logging
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# data = load_data()
def load_and_prepare_data():
    '''
    return: pd.DataFrame
    
    This function loads the data and returns a DataFrame.
    
    '''
    return load_data()


class BasicSearch:
    
    def __init__(self, data, query, num_results):
        self.data = data
        self.query = query
        self.num_results = num_results


    def _create_index(self):
        '''
        data: pd.DataFrame
        return: Index
        
        This function takes a DataFrame and returns an Index object.
        '''
        
        index = Index(
            text_fields = ["question", "answer", "sub.category"],
            keyword_fields = ["category"]
        )
        return index.fit(self.data)    

    def basic_search(self) -> list:
        '''
        query: str
        return: str
        
        This function takes a query and returns the same query.
        
        '''
        # filter_dict = {"category": "Questions about God"}
        boost_dict = {"question": 3, "answer": 1, "sub.category": 1}
        return self._create_index.search(self.query, boost_dict, num_results=self.num_results)

# ====================================================================================================

### Elastic Search and Semantic Search ###
class AdvancedElasticSearch:
    
    def __init__(
        self, 
        data, 
        index_name, 
        num_results, 
        search_client
    ):
        self.data = data
        self.index_name = index_name
        self.num_results = num_results
        self.search_client = search_client


    def start_elastic_search(self, *args, **kwargs):
        
        self._create_elastic_search_client()
        self._index_data()

    
    def _create_elastic_search_client(self) -> str:
        '''
        query: str
        return: str
        
        This function takes a query and returns the same query.
        
        '''
        index_settings = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0
            },
            "mappings": {
                "properties": {
                    "answer": {"type": "text"},
                    "sub.category": {"type": "text"},
                    "question": {"type": "text"},
                    "category": {"type": "keyword"} 
                }
            }
        }

        index_name = self.index_name
        if not self.search_client.indices.exists(index=index_name):
            self.search_client.indices.create(index=index_name, body=index_settings)
        self.search_client.indices.create(index=index_name, body=index_settings)


    def _index_data(self) -> str:
        '''
        query: str
        return: str
        
        This function takes a query and returns the same query.
        '''
        for doc in self.data:
            self.search_client.index(index=self.index_name, body=doc)
            

    def advanced_search(self, query: str) -> List[Dict[str]]:
        '''
        query: str
        return: List[Dict[str]]
        
        This function takes a query and returns the same query.
        
        '''
        results_docs = []
        search_query = {
            "size": self.num_results,
            "query": {
                "bool": {
                    "must": {
                        "multi_match": {
                            "query": query,
                            "fields": ["question^3", "answer", "sub.category"],
                            "type": "best_fields"
                        }
                    }
                }
            }
        }
        
        try:
            results = self.search_client.search(index=self.index_name, body=search_query, size=self.num_results)
            results_docs.extend(hits["_source"] for hits in results["hits"]["hits"])
            logging.info(f"Search results available are: {self.num_results}")
            return results_docs
        except Exception as e:
            logging.error(f"An error occured: {e}")
            return []


# ====================================================================================================
# Vector Search

class VectorSearch:
    
    def __init__(
        self, 
        data = None,
        search_client = None,
        num_results = 5, 
        model_name = "all-MiniLM-L6-v2", 
        embed_model_cache_dir = "./cache",
        index_name = "vector_search_index"
    ):
        self.data = data
        self.search_client = search_client
        self.num_results = num_results
        self.model_name = model_name
        self.embed_model_cache_dir = embed_model_cache_dir 
        self.index_name = index_name
        

    def start_vs(self):
        '''
        query: str
        return: str
        
        This function takes a query and returns the same query.
        
        '''
        self._vs_client()
        self._vs_index_data()
        
    def _vs_client(self) -> str:
        '''
        query: str
        return: str
        
        This function takes a query and returns the same query.
        
        '''
        index_settings = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0
            },
            "mappings": {
                "properties": {
                    "answer": {"type": "text"},
                    "sub.category": {"type": "text"},
                    "question": {"type": "text"},
                    "category": {"type": "keyword"} ,
                    "answer_embedding": {
                        "type": "dense_vector", 
                        "dims": 384, 
                        "index": True, 
                        "similarity": "cosine"
                    },
                }
            }
        }
        
        try:
            if self.search_client.exists(self.index_name):
                self.search_client.delete(self.index_name)
                logging.info("Existing index '%s' deleted.", self.index_name)
            self.search_client.create(self.index_name, index_settings)
            logging.info("Index '%s' created.", self.index_name)
        except Exception as e:
            logging.error(f"An error occured: {e}")
            
    
    def _load_model(self) -> SentenceTransformer:
        '''
        query: str
        return: str
        
        This function takes a query and returns the same query.
        
        '''
        return SentenceTransformer(self.model_name, cache_folder=self.embed_model_cache_dir)
    
    
    def _vs_index_data(self) -> str:
        '''
        query: str
        return: str
        
        This function takes a query and returns the same query.
        
        '''
        model = self._load_model()
        
        for doc in self.data:
            doc["answer_embedding"] = model.encode(doc["answer"])
            self.search_client.index(self.index_name, doc)
            
        
    
    def vector_search(self, query: str) -> List[Dict[str]]:
        '''
        query: str
        return: List[Dict[str]]
        
        This function takes a query and returns the same query.
        
        '''
        model = self._load_model()
        query_embedding = model.encode(query)
        
        query = {
            "field": "answer_embedding",
            "query_vector": query_embedding.tolist(),
            "k": 5,
            "num_candidates": 10000
        }
        
        try:
            results = self.search_client.search(self.index_name, search_query)
            return results
        except Exception as e:
            logging.error(f"An error occured: {e}")
            return []
