import math
from itertools import chain
import time
import numpy as np
from QueryProcessor import QueryProcessor
import time
from BM25 import BM25

def merge_results(title_scores, body_scores, page_rank, page_view, title_weight=0.5, text_weight=0.5,
                  page_rank_weight=0.5, page_view_weight=0.5, N=3):
    """
    This function merge and sort documents retrieved by its weighte score (e.g., title and body).

    Parameters:
    -----------
    title_scores: a dictionary build upon the title index of queries and tuples representing scores as follows:
                                                                            key: query_id
                                                                            value: list of pairs in the following format:(doc_id,score)

    body_scores: a dictionary build upon the body/text index of queries and tuples representing scores as follows:
                                                                            key: query_id
                                                                            value: list of pairs in the following format:(doc_id,score)
    title_weight: float, for weigted average utilizing title and body scores
    text_weight: float, for weigted average utilizing title and body scores
    N: Integer. How many document to retrieve. This argument is passed to topN function. By default N = 3, for the topN function.

    Returns:
    -----------
    dictionary of querires and topN pairs as follows:
                                                        key: query_id
                                                        value: list of pairs in the following format:(doc_id,score).
    """
    # getting all ditinct keys
    all_doc_id = list(set(list(title_scores.keys()) + list(body_scores.keys())))
    doc_dict = {}
    for doc_id in all_doc_id:
        isIn = False
        # checking if the query_id is in the titles dict
        if title_scores.get(doc_id) is not None:
            isIn = True
            score = title_scores[doc_id]
            w_score = score * title_weight
            doc_dict[doc_id] = w_score
        # checking if the query_id is in the body dict
        if body_scores.get(doc_id) is not None:
            isIn = True
            score = body_scores[doc_id]
            w_score = score * text_weight
            doc_dict.setdefault(doc_id, 0) 
            doc_dict[doc_id] += w_score
        if isIn is True:
            # Adding pagerank and pageview scores
            pagerank_score = page_rank.get(doc_id)
            page_view_score = page_view.get(doc_id)
            if pagerank_score:
                doc_dict[doc_id] += (pagerank_score * page_rank_weight)
            if page_view_score:
                doc_dict[doc_id] += (page_view_score * page_view_weight)
    
    # Sorting the documents by score 
    doc_score_list = list(zip(doc_dict.keys(), doc_dict.values()))
    sorted_docs_scores = sorted(doc_score_list, key=lambda x: x[1], reverse=True)[:N]
    docs = [doc for doc, score in sorted_docs_scores]
    
    return docs

def beckend_search(query, inv_index_text, inv_index_title, inv_index_anchors, page_rank, page_view, id_to_title, doc_to_len):
    query_processor = QueryProcessor()
    token_query = query_processor.tokenize(query)
        
    bm25_title = BM25(inv_index_title, doc_to_len, "inverted_index_title", k1=2, b=0.05) 
    BM25_title_score = dict(bm25_title.search(token_query))
    
    bm25 = BM25(inv_index_text, doc_to_len, "inverted_index_text", k1=5, b=0.2) 
    BM25_body_score = dict(bm25.search(token_query))    
    
    docs = merge_results(title_scores=BM25_title_score, body_scores=BM25_body_score,
                         page_rank=page_rank, page_view=page_view, text_weight=0.7, title_weight=0.3,
                         page_rank_weight=0.15,page_view_weight=0.15, N=40)
    
    doc2titles = query_processor.id_to_title(id_to_title, docs)
    
    return doc2titles



