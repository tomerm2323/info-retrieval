from CosineSim import CosineSim
from IndexReader import IndexReader
from QueryProcessor import QueryProcessor
from pyspark import SparkContext
from pyspark.conf import SparkConf
conf = SparkConf()
conf.setMaster('local').setAppName('myapp')
sc = SparkContext(conf=conf)
from flask import Flask, request, jsonify
from InvertedIndex import InvertedIndex
import requests
import datetime
from mwviews.api import PageviewsClient
import backend_search
import time
from MultiFileReader import MultiFileReader

class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

invertedIndex = InvertedIndex()
reader = IndexReader(invertedIndex)

global inv_index_text
inv_index_text = reader.read_index('', 'inverted_index_text')
global inv_index_title
inv_index_title = reader.read_index('', 'inverted_index_title')
global inv_index_anchor
inv_index_anchor = reader.read_index('','inverted_index_anchor')
global id_to_title
id_to_title = reader.read_index('', 'doc_to_title')
global doc_to_len
doc_to_len = reader.read_index('', 'docs_len')
global tokens
tokens = reader.read_index('', 'tokens')
global page_view
page_view = reader.read_index('', 'page_views')
global page_rank
page_rank_df = reader.read_pagerank('page_rank')
page_rank = page_rank_df.set_index('id').to_dict()['pr']


@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    res = backend_search.beckend_search(query, inv_index_text, inv_index_title, inv_index_anchor,
                                        page_rank, page_view, id_to_title, doc_to_len)
    return jsonify(res)

@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. 

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)

    inv_index = inv_index_text
    ids_and_titles = id_to_title
    query_processor = QueryProcessor()
    query_as_tokens = query_processor.tokenize(query)
    tfidf_query_vec = query_processor.calc_tfidf_query(query=query_as_tokens, inverted_index=inv_index)
    cosine = CosineSim(inverted_index=inv_index, doc_to_len=doc_to_len)
    doc_term_tfidf_matrics = cosine.get_candidate_docs(query=query_as_tokens, folder_name="inverted_index_text")
    cosine_sim_dict = cosine.cos_sim(query=tfidf_query_vec,docs=doc_term_tfidf_matrics)
    top100 = cosine.get_top_n(score_dict=cosine_sim_dict, N=100)  # return as doc_id, score
    docs_title_pair = query_processor.id_to_title(ids_and_titles, [i[0] for i in top100])
    return jsonify(docs_title_pair)


@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO 
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. For example, a document 
        with a title that matches two distinct query words will be ranked before a 
        document with a title that matches only one distinct query word, 
        regardless of the number of times the term appeared in the title (or 
        query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    query = request.args.get('query', '')
    res = {}
    if len(query) == 0:
        return jsonify(res)
    inv_index = inv_index_title
    query_processor = QueryProcessor()
    query_as_tokens = query_processor.tokenize(query)
    for token in query_as_tokens:
        # byte_pl = inv_index.get_byte_pl(token)
        # pl = inv_index.byte_pl_to_list(byte_pl)
        # pl = reader.load_posting_lists_for_token(token, inv_index, 'postings_gcp_title')

        # reader = IndexReader(inv_index)
        # TUPLE_SIZE = 6       
        # bucket_name = "208378042-irproject"
        folder_name = "inverted_index_title"
        # n_bytes = inv_index.df[token] * TUPLE_SIZE
        # locs = inv_index.posting_locs[token]
        # pl = reader.read_title_pl( locs, n_bytes, bucket_name, folder_name, token, inv_index)
        
        pl = reader.token_posting_list(token, inv_index_title, folder_name)
        
        for doc_id, tf in pl:
            instances_in_doc_title = res.setdefault(doc_id, 0) + tf
            res[doc_id] = instances_in_doc_title
    sorted_res = {k: v for k, v in sorted(res.items(), key=lambda item: item[1], reverse=True)}
    docs_title_pair = query_processor.id_to_title(id_to_title, list(sorted_res.keys()))[:100]
    return jsonify(docs_title_pair)

@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment 
        3 (GCP part) to do the tokenization and remove stopwords. For example, 
        a document with a anchor text that matches two distinct query words will 
        be ranked before a document with anchor text that matches only one 
        distinct query word, regardless of the number of times the term appeared 
        in the anchor text (or query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    query = request.args.get('query', '')
    res = {}
    if len(query) == 0:
        return jsonify(res)
    query_processor = QueryProcessor()
    query_as_tokens = query_processor.tokenize(query)
    reader = IndexReader(inv_index_anchor)
    for token in query_as_tokens:
        posting_list = reader.token_posting_list(token, inv_index_anchor, "inverted_index_anchor")
        for doc_id, tf in posting_list:
            instances_in_doc_title = res.setdefault(doc_id, 0) + tf
            res[doc_id] = instances_in_doc_title
    sorted_res = {k: v for k, v in sorted(res.items(), key=lambda item: item[1], reverse=True)}
    docs_title_pair = query_processor.id_to_title(id_to_title, list(sorted_res.keys()))[:100]
    return jsonify(docs_title_pair)


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''

    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    for id in wiki_ids:
        pagerank_score = page_rank.get(id)
        if pagerank_score:
            res.append(pagerank_score)
    return jsonify(res)

@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    print("LOGGER: get_pageview(): Started")
    res = []
    wiki_ids = request.get_json()
    print(f"LOGGER: get_pageview(): wiki_ids = {wiki_ids}")
    if len(wiki_ids) == 0:
        return jsonify(res)
    elif len(wiki_ids) < 30:
        jsonRes = calc_page_view([id_to_title[wiki_id] for wiki_id in wiki_ids])
        res = list(jsonRes.values())
        if not res[0]:
            res = [page_view[wiki_id] for wiki_id in wiki_ids]
        return jsonify(res)
    else:
        res = [page_view[wiki_id] for wiki_id in wiki_ids]
        return jsonify(res)


def calc_page_view(titles: list):
  contatct_info = "tomerm3399@gmail.com"
  p = PageviewsClient(user_agent="Python query script by " + contatct_info)
  start_date = datetime.date(2021, 8, 1)
  end_date = datetime.date(2021, 9, 1)
  senate_views = p.article_views(project='en.wikipedia', 
                              articles=titles, 
                              granularity='monthly', 
                              start=start_date, 
                              end=end_date)
  page_view_dict = list(senate_views.values())[0]

  return page_view_dict

if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)

