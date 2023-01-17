from IndexReader import IndexReader
from InvertedIndex import InvertedIndex
import numpy as np


class BM25:

    def __init__(self, index: InvertedIndex, DL, folder_name, k1=1.5, b=0.75):
        self.b = b
        self.k1 = k1 
        self.index = index
        self.N = len(DL)
        self.DL = DL
        self.AVGDL = sum(DL.values()) / self.N
        self.idf = None
        self.freqs = {}
        self.folder_name = folder_name

    def calc_idf(self, list_of_tokens):
        idf = {}
        for term in list_of_tokens:
            if term in self.index.df.keys():
                n_ti = self.index.df.get(term, 0.5-self.N)
                idf[term] = np.log(1 + (self.N - n_ti + 0.5) / (n_ti + 0.5))
            else:
                pass
        return idf

    def get_candidate_docs(self, query, index):
        res = set()
        for token in np.unique(query):
            if token in index.df.keys():
                reader = IndexReader(index)
                pl = reader.token_posting_list(token, index, self.folder_name)
                self.freqs[token] = {}
                for doc_id, tf in pl:
                    self.freqs[token][doc_id] = tf
                    res.add(doc_id)
        return res

    def search(self, tokenized_query, N=100):
        self.idf = self.calc_idf(tokenized_query)
        candid_list = self.get_candidate_docs(tokenized_query, self.index)
        res = sorted([(doc_id, self._score(tokenized_query, doc_id)) for doc_id in candid_list], key=lambda x: x[1],
                     reverse=True)[:N]
        return res

    def _score(self, query, doc_id):
        score = 0.0
        doc_len = self.DL.get(doc_id, 0)

        for term in query:
            if term in self.freqs.keys():
                if doc_id in self.freqs[term]:
                    freq = self.freqs[term][doc_id]
                    numerator = self.idf.get(term, 0) * freq * (self.k1 + 1)
                    denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.AVGDL)
                    score += (numerator / denominator)
        return score