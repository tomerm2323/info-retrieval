import scipy
import numpy as np
from IndexReader import IndexReader

class Metric:
  def __init__(self, inverted_index):
    # self.index = index
    self.inverted_index = inverted_index

  def get_candidate_docs(self, query):
    """
    This method return all the relevant docs to calculate cosine similiry with.
    The relevant docs are the ones who in the posting list of each term in the query. 

    parameters
    ----------
    query: list, each element is a word in the query

    returns
    -------
    dict, key --> (doc_id, term), value --> tfidf
    """
    term_vector = list(self.inverted_index.df.keys())

    eps = 10 ** -9
    docs = {}
    tokens = list(set(query))
    for token in tokens:
      reader = IndexReader() 
      pl = reader.load_posting_lists_for_token(token, self.inverted_index, 'postings_gcp')
      for doc_id, tf in pl:
        # key = (doc_id,token)

        norm_tf = tf / doc_to_len[doc_id]
        idf = np.log(self.inverted_index.N / self.inverted_index.df[token] + eps)
        tfidf = norm_tf * idf

        # docs[key] = tfidf
        # token_index = term_vector.index(token)
        docs.setdefault(doc_id, [])
        # docs[doc_id].append((token_index, tfidf))
        docs[doc_id].append((token, tfidf))
    return docs

  def spars_matrix_to_dict(self, matrix):
    
    indices = matrix.nonzero() 
    # Get the values of the non-zero elements
    values = matrix.data
    # Zip the indices and values into a list of tuples
    non_zero_elements = list(zip(indices[0], indices[1], values))
    docs = set([non_zero_elements[i][0] for i in range(len(non_zero_elements))])
    docs_tfidf = dict.fromkeys(docs,None)
    for doc,term_index, tfidf in non_zero_elements:
      if docs_tfidf[doc] is None: 
        docs_tfidf[doc] = [(term_index,tfidf)]
      else:
        docs_tfidf[doc].append((term_index,tfidf))
    return docs_tfidf

  def tfidf_vec_size(self,doc_id):
    doc_stasts = self.inverted_index.doc_stats[doc_id]
    vec_size = 0 
    for term,tf in doc_stasts.items():
      term_df = self.inverted_index.df[term]
      idf = np.log(self.inverted_index.N / term_df)
      tfidf = tf * idf
      vec_size += tfidf ** 2
    vec_size = np.sqrt(vec_size)
    return vec_size

  def get_top_n(self, score_dict, N=3):
    """ 
    Sort and return the highest N documents according to the cosine similarity score.
    Generate a dictionary of cosine similarity scores 
   
    Parameters:
    -----------
    score_dict: a dictionary of similarity score as follows:
                                                                key: document id (e.g., doc_id)
                                                                value: similarity score. We keep up to 5 digits after the decimal point. (e.g., round(score,5))

    N: Integer (how many documents to retrieve). By default N = 3
    
    Returns:
    -----------
    a ranked list of pairs (doc_id, score) in the length of N.
    """
    
    return sorted([(doc_id,np.round(score,5)) for doc_id, score in score_dict.items()], key = lambda x: x[1],reverse=True)[:N]
