import numpy as np
import scipy


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
      byte_pl = self.inverted_index.get_byte_pl(word=token)
      pl = self.inverted_index.byte_pl_to_list(byte_pl)
      for doc_id, tf in pl:
        # key = (doc_id,token)
        norm_tf = tf / doc2len[doc_id]
        idf = np.log(self.inverted_index.N / self.inverted_index.df[token] + eps)
        tfidf = norm_tf * idf
        # docs[key] = tfidf
        token_index = term_vector.index(token)
        docs.setdefault(doc_id, [])
        docs[doc_id].append((token_index, tfidf))
    return docs




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
