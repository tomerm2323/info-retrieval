from IndexUtils.CosineSim import CosineSim
from IndexUtils.IndexReader import IndexReader
from IndexUtils.QueryProcessor import QueryProcessor
from pyspark import SparkContext
from pyspark.conf import SparkConf
conf = SparkConf()
conf.setMaster('local').setAppName('myapp')
sc = SparkContext(conf=conf)
from flask import Flask, request, jsonify

class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

# sc = SparkContext("local", "Simple App")
    
global inv_index_text
inv_index_text = IndexReader.read_index('home/shemi_p17/inv_index_text.pkl')
global inv_index_title
inv_index_title = IndexReader.read_index('home/shemi_p17/inv_index_title.pkl')
global inv_index_anchor
inv_index_anchor = IndexReader.read_rdd_from_binary_files('/home/shemi_p17/inv_index_anchors', sc)
global id_to_title
id_to_title = IndexReader.read_rdd_from_binary_files('/home/shemi_p17/doc_id_to_titles', sc)
global doc2tfidf_size
doc2tfidf_size = IndexReader.read_rdd_from_binary_files('/home/shemi_p17/tfIdfSizes.pkl', sc)
global doc_to_len
doc_to_len = IndexReader.read_index('/home/shemi_p17/doc_len.pkl')

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
    score =  + 1000
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
    ids_and_titles = ids_to_titles
    query_processor = QueryProcessor()
    query_as_tokens = query_processor.tokenize(query)
    tfidf_qurery_vec =  query_processor.calc_tfidf_query(query=query_as_tokens,inverted_index=inv_index)
    cosine = CosineSim(inverted_index=inv_index)
    doc_term_tfidf_matrics = cosine.get_tfidf_matrix(query=query_as_tokens)
    cosine_sim_dict = cosine.cos_sim(query=tfidf_qurery_vec,docs=doc_term_tfidf_matrics)

    top100 = cosine.get_top_n(score_dict=cosine_sim_dict, N=100) # return as doc_id, score
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
    # ids_and_titles = ids_to_titles
    query_processor = QueryProcessor()
    query_as_tokens = query_processor.tokenize(query)
    for token in query_as_tokens:
        byte_pl = inv_index.get_byte_pl(token)
        pl = inv_index.byte_pl_to_list(byte_pl)
        for doc_id, tf in pl:
            instances_in_doc_title = res.setdefault(doc_id, 0) + tf
            res[doc_id] = instances_in_doc_title
    sorted_res = {k: v for k, v in sorted(res.items(), key=lambda item: item[1])}
    # docs_title_pair = query_processor.id_to_title(ids_and_titles, list(sorted_res.keys()))[:100]
    return jsonify(sorted_res)

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
    rdd_anchor_stats = anchor_index
    ids_and_titles = ids_to_titles
    query_processor = QueryProcessor()
    query_as_tokens = query_processor.tokenize(query)
    for token in query_as_tokens:
        token_stats_dict = rdd_anchor_stats.filter(lambda x: list(x.keys())[0] == token).first()
        doc_tf_stats = token_stats_dict[token]
        for src_id, src_tf in doc_tf_stats.itmes():
            res.setdefault(src_id, 0)
            res[src_id] += src_tf
    sorted_res = {k: v for k, v in sorted(res.items(), key=lambda item: item[1])}
    docs_anchor_pair = query_processor.id_to_title(ids_and_titles, list(sorted_res.keys()))
    return jsonify(docs_anchor_pair)


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
    pr_filtered = pr.filter(pr["id"].isin(wiki_ids))
    pagerank_list = list(pr_filtered.select("pagerank").toPandas()['pagerank'])
    return jsonify(pagerank_list)

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
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    res = [pageview[f'{wiki_id}'] for wiki_id in wiki_ids]
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)

