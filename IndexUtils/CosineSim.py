import numpy as np
import pandas as pd
from Metric import Metric
from sklearn.metrics.pairwise import cosine_similarity

class CosineSim(Metric):

  def __init__(self, inverted_index):
    super().__init__(inverted_index)

  #cosine-sim sklearn
  def cosine_sim_using_sklearn(self, queries,tfidf):
      """
      In this function you need to utilize the cosine_similarity function from sklearn.
      You need to compute the similarity between the queries and the given documents.
      This function will return a DataFrame in the following shape: (# of queries, # of documents).
      Each value in the DataFrame will represent the cosine_similarity between given query and document.
      
      Parameters:
      -----------
        queries: sparse matrix represent the queries after transformation of tfidfvectorizer.
        documents: sparse matrix represent the documents.
        
      Returns:
      --------
        DataFrame: This function will return a DataFrame in the following shape: (# of queries, # of documents).
        Each value in the DataFrame will represent the cosine_similarity between given query and document.
      """
      cos_sim = cosine_similarity(queries,tfidf)
      df = pd.DataFrame(cos_sim)
      return df
  # cosine self implementation

  def cosine_similarity(self, D,Q):
      """
      Calculate the cosine similarity for each candidate document in D and a given query (e.g., Q).
      Generate a dictionary of cosine similarity scores 
      key: doc_id
      value: cosine similarity score
      
      Parameters:
      -----------
      D: DataFrame of tfidf scores.

      Q: vectorized query with tfidf scores
      
      Returns:
      -----------
      dictionary of cosine similarity score as follows:
                                                                  key: document id (e.g., doc_id)
                                                                  value: cosine similarty score.
      """
      q_abs_value = np.linalg.norm(Q)
      original_index = D.index.values
      Dtrans = D.transpose()
      df_abs = Dtrans.apply(lambda x : np.linalg.norm(x) * q_abs_value)
      dot_product_df = pd.DataFrame(np.dot(D,Q),index=original_index)
      cosine_sim_df = dot_product_df.div(df_abs.values,axis=0)[0]
      return cosine_sim_df.to_dict()
    
  