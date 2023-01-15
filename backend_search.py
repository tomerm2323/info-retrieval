import math
from itertools import chain
import time
import numpy as np
from IndexUtils.IndexReader import IndexReader
from IndexUtils.QueryProcessor import QueryProcessor
from IndexUtils.Metrics import Metric
# When preprocessing the data have a dictionary of document length for each document saved in a variable called `DL`.
class BM25_from_index(Metric.Metric):
    """
    Best Match 25.
    ----------
    k1 : float, default 1.5

    b : float, default 0.75

    index: inverted index
    """

    def __init__(self, index, k1=1.5, b=0.75):
        self.b = b
        self.k1 = k1
        self.index = index
        self.N = len(doc_to_len)
        self.AVGDL = sum(doc_to_len.values()) / self.N



    def search(self, query, N=3):
        """
        This function calculate the bm25 score for given query and document.
        We need to check only documents which are 'candidates' for a given query.
        This function return a dictionary of scores as the following:
                                                                    key: query_id
                                                                    value: a ranked list of pairs (doc_id, score) in the length of N.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']
        doc_id: integer, document id.

        Returns:
        -----------
        score: float, bm25 score.
        """
        m = Metric(self.index)
        candidates = m.get_candidate_docs(query)
        docs_scores = list(set([(doc_id, self._score(query, doc_id)) for doc_id in candidates.keys()]))
        docs_scores_sorted = dict(sorted(docs_scores, key=lambda x: x[1], reverse=True)[:N])
        return docs_scores_sorted

    def _score(self, query, doc_id):
        """
        This function calculate the bm25 score for given query and document.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']
        doc_id: integer, document id.

        Returns:
        -----------
        score: float, bm25 score.
        """
        score = 0.0
        doc_len = doc_to_len[doc_id]
        for term in query:
            if term in self.index.term_total.keys():
                ####################### folder name####################################
                term_frequencies = dict(IndexReader.load_posting_lists_for_token(term, index=self.index,folder_name= 'postings_gcp'))
                if doc_id in term_frequencies.keys():
                    freq = term_frequencies[doc_id]
                    term_idf = np.log( self.N / self.index.df[term])
                    numerator = term_idf * freq * (self.k1 + 1)
                    denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.AVGDL)
                    score += (numerator / denominator)
        return score

def beckend_search_title(query_as_tokens):
    if len(query_as_tokens) == 0:
        return 0
    res = {}
    inv_index = inv_index_title
    for token in query_as_tokens:
        pl = IndexReader.load_posting_lists_for_token(token, inv_index,'postings_gcp_title')
        for doc_id, tf in pl:
            instances_in_doc_title = res.setdefault(doc_id, 0) + tf
            res[doc_id] = instances_in_doc_title
    sorted_res = {k: v for k, v in sorted(res.items(), key=lambda item: item[1])}
    return sorted_res


def merge_results(title_scores, body_scores, title_weight=0.5, text_weight=0.5, N=3):
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
    for doc_id in all_doc_id:
        doc_dict = {}
        # checking if the query_id is in the titles dict
        if title_scores.get(doc_id):
            # title_score_lst = title_scores[doc_id]
            # for doc_id, score in title_score_lst:
            score = title_scores[doc_id]
            w_score = score * title_weight
            doc_dict[doc_id] = w_score
        # checking if the query_id is in the body dict
        if body_scores.get(doc_id):
            score = body_scores[doc_id]
            w_score = score * text_weight
            if doc_dict.get(doc_id):
                doc_dict[doc_id] += w_score
            else:
                doc_dict[doc_id] = w_score
        doc_dict.setdefault(doc_id, 0)
        doc_id[doc_id] += pr[doc_id]

    doc_score_list = list(zip(doc_dict.keys(), doc_dict.values()))
    sorted_docs_scores = sorted(doc_score_list, key=lambda x: x[1], reverse=True)[:N]
    docs = [doc for doc, score in sorted_docs_scores]
    return docs
def beckend_search(query):

    query_processor = QueryProcessor()
    token_query = query_processor.tokenize(query)
    doc_title_match = beckend_search_title(token_query)
    BM25_body_score = BM25_from_index(index=inv_index_text).search(token_query)
    docs = merge_results(title_scores=doc_title_match,
                             body_scores=BM25_body_score,text_weight=10, title_weight=1, N=40)
    query_processor = QueryProcessor()
    doc2titles = query_processor.id_to_title(id_to_title, docs)
    return doc2titles



