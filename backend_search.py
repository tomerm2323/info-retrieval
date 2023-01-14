import math
from itertools import chain
import time
from IndexUtils.Metric import Metric

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
        self.words, self.pls = zip(*self.index.posting_lists_iter())


    def search(self, queries, N=3):
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
        return_dict = dict.fromkeys(queries.keys())
        for query_key, query in queries.items():
            candidates = self.get_candidate_documents(query)
            doc_score = list(set([(doc_id, self._score(query, doc_id)) for doc_id in candidates.keys()]))
            return_dict[query_key] = sorted(doc_score, key=lambda x: x[1], reverse=True)[:N]
        return return_dict

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
        self.idf = self.calc_idf(query)
        for term in query:
            if term in self.index.term_total.keys():
                term_frequencies = dict(IndexReader.read_pl(term))
                if doc_id in term_frequencies.keys():
                    freq = term_frequencies[doc_id]
                    numerator = self.idf[term] * freq * (self.k1 + 1)
                    denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.AVGDL)
                    score += (numerator / denominator)
        return score