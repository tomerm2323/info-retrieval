from collections import Counter
import numpy as np
import scipy
from Processor import Processor

class QueryProcessor(Processor):
  def __init__(self):
    pass

  def calc_tfidf_query(self,query,inverted_index):
    """
    returns the query as a vector of tfidf for each term.
    
    parameters
    ----------
      query: list, tokens in the query
      inverted_index: InvertedIndex
    
    returns
    -------
      np.array, size of 1 X  (voabulary size)
    """
    eps = 10 ** -9
    total_vocab_size = len(inverted_index.term_total)
    Q = scipy.sparse.csr_matrix((1,total_vocab_size))
    term_vector = list(inverted_index.term_total.keys())    
    term_stats_dict = Counter(query)
    N = inverted_index.N
    for term, tf in term_stats_dict.items():
      if term in inverted_index.term_total.keys():
        term_df = inverted_index.df[term]
        idf = np.log(N/ term_df + eps)
        tfidf = tf * idf
        try:
            ind = term_vector.index(term)
            Q[0,ind] = tfidf                    
        except:
            pass 
    return Q