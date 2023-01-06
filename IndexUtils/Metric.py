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
    eps = 10 ** -9
    docs = {}
    tokens = list(set(query))
    for token in tokens:
      byte_pl = self.inverted_index.get_byte_pl(word=token)
      
      pl = self.inverted_index.byte_pl_to_list(byte_pl)
      for doc_id,tf in pl:
        key = (doc_id,token)
        idf = np.log(self.inverted_index.N / self.inverted_index.df[token] + eps)
        tfidf = tf * idf 
        docs[key] = tfidf
    return docs


  def get_tfidf_matrix(self,query):
    """
    returns a matrix for a docs X terms where each entry is the tfidf scroe of the term in the doc.

    parameters
    ----------
      query: list, tokens in the query
    
    returns
    -------
      pd.DataFrame, size of (canidate docs) X  (voabulary size)
    


    """
    total_vocab_size = len(self.inverted_index.term_total)
    term_vector = list(self.inverted_index.term_total.keys())  
    candidates_scores = self.get_candidate_docs(query) 
    print(f"candidates_scores = {candidates_scores}")
    unique_candidates = np.unique([doc_id for doc_id, freq in candidates_scores.keys()])
    print(f"unique_candidates = {unique_candidates}")
    # D = np.zeros((len(unique_candidates), total_vocab_size))
    # D = pd.DataFrame(D)
    
    # D.index = unique_candidates
    # D.columns = self.inverted_index.term_total.keys()
    total_candidates = len(unique_candidates)
    matrix = scipy.sparse.lil_matrix((total_candidates, total_vocab_size))
    for key, tfidf in candidates_scores.items():
      doc_id, term = key 
      try:
        term_index = term_vector.index(term) 
        matrix[doc_id, term_index] = tfidf                    
      except:
          pass 
    matrix = matrix.tocsr()
    return matrix
  def get_top_n(self, score_dict,N=3):
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
    
    return sorted([(doc_id,round(score,5)) for doc_id, score in score_dict.items()], key = lambda x: x[1],reverse=True)[:N]
