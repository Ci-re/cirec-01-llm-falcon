# Imports
import pandas as pd
from minsearch import Index
from gather import load_data

data = load_data()

index = Index(
    text_fields = ["question", "answer", "sub.category"],
    keyword_fields = ["category"]
)

index.fit(data)    



def basic_search(query) -> list:
    '''
    query: str
    return: str
    
    This function takes a query and returns the same query.
    
    '''
    # filter_dict = {"category": "Questions about God"}
    boost_dict = {"question": 3, "answer": 1, "sub.category": 1}
    return index.search(query, boost_dict, num_results=5)


def semantic_search(query) -> str:
    '''
    query: str
    return: str
    
    This function takes a query and returns the same query.
    
    '''
    
    return query


def hybrid_search(query) -> str:
    '''
    query: str
    return: str
    
    This function takes a query and returns the same query.
    
    '''
    
    return query


