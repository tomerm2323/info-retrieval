from collections import Counter
import numpy as np
import scipy
from Processor import Processor

import scipy


class QueryProcessor(Processor):
  def __init__(self):
    pass
    # self.panc = set(string.punctuation)
    # self.stopwords = set(stopwords.words('english'))

  def spars_matrix_to_list(self, matrix):
    indices = matrix.nonzero()
    # Get the values of the non-zero elements
    values = matrix.data

    # Zip the indices and values into a list of tuples
    non_zero_elements = list(zip(indices[1], values))
    return non_zero_elements

  def calc_tfidf_query(self, query, inverted_index):
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
    # total_vocab_size = len(inverted_index.df)
    # Q = scipy.sparse.csr_matrix((1, total_vocab_size))
    Q = {}
    # term_vector = list(inverted_index.df.keys())
    term_stats_dict = Counter(query)
    N = inverted_index.N
    qeury_len = len(query)
    for term, tf in term_stats_dict.items():
      if term in inverted_index.df.keys():
        norm_tf = tf / qeury_len
        term_df = inverted_index.df[term]
        idf = np.log(N / term_df + eps)
        tfidf = norm_tf * idf
        try:
          # ind = term_vector.index(term)
          Q[term] = tfidf
        except:
          pass
    # non_zero_elements = self.spars_matrix_to_list(matrix=Q)
    # return non_zero_elements
    return Q
  def id_to_title(self, id_title_dict, ids_to_titles):
    """
      Returns the titles of all ids_to_titles doc ids
    """
    result_titles = []
    for id in ids_to_titles:
      result_titles.append((id, id_title_dict[id]))
    return result_titles