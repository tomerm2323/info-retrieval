from IndexUtils.CosineSim import CosineSim
from IndexUtils.IndexReader import IndexReader
from IndexUtils.QueryProcessor import QueryProcessor
from flask import Flask, request, jsonify

class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


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
    # BEGIN SOLUTION

    # END SOLUTION
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

    inv_index = IndexReader().read_index(base_dir='inv_index_text',name='inv_index_text')
    ids_and_titles = IndexReader().read_index(base_dir='titles', name='ids_titles')
    query_processor = QueryProcessor()
    query_as_tokens = query_processor.tokenize(query)
    print(f"query_as_tokens = {query_as_tokens}")
    tfidf_qurery_vec =  query_processor.calc_tfidf_query(query=query_as_tokens,inverted_index=inv_index)
    cosine = CosineSim(inverted_index=inv_index)
    print(f"cosine.get_tfidf_matrix started")
    doc_term_tfidf_matrics = cosine.get_tfidf_matrix(query=query_as_tokens)
    print(f"cosine.get_tfidf_matrix ended")
    print(f"cosine.cosine_similarity started, D = {doc_term_tfidf_matrics},   Q = {tfidf_qurery_vec}")
    print()
    print(f"D.keys = {doc_term_tfidf_matrics.keys()}, D.values {doc_term_tfidf_matrics.values()}")
    print()
    # cosine_sim_dict = cosine.cosine_similarity(D=doc_term_tfidf_matrics, Q=tfidf_qurery_vec)
    # cosine_sim_dict = cosine.cosine_sim_using_sklearn(queries=tfidf_qurery_vec,tfidf=doc_term_tfidf_matrics).to_dict()
    cosine_sim_dict = cosine.cos_sim(query=tfidf_qurery_vec,docs=doc_term_tfidf_matrics)
    print(f"cosine.cosine_similarity ended")
    print()
    print(f"cosine_sim_dict = {cosine_sim_dict}")
    print()
    print(f"cosine.get_top_n started")
    top100 = cosine.get_top_n(score_dict=cosine_sim_dict, N=100) # return as doc_id, score
    print(f"cosine.get_top_n ended")
    docs_title_pair = query_processor.id_to_title(ids_and_titles, top100)[:100]
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
    inv_index = IndexReader().read_index(base_dir='inv_index_title', name='inv_index_title')
    ids_and_titles = IndexReader().read_index(base_dir='titles', name='ids_titles')
    query_processor = QueryProcessor()
    query_as_tokens = query_processor.tokenize(query)
    for token in query_as_tokens:
        byte_pl = inv_index.get_byte_pl(token)
        pl = inv_index.byte_pl_to_list(byte_pl)
        for doc_id, tf in pl:
            instances_in_doc_title = res.setdefault(doc_id, 0) + tf
            res[doc_id] = instances_in_doc_title
    sorted_res = {k: v for k, v in sorted(res.items(), key=lambda item: item[1])}
    docs_title_pair = query_processor.id_to_title(ids_and_titles, list(sorted_res.keys()))[:100]
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
    rdd_anchor_stats = IndexReader().read_index(base_dir='inv_index_anchor', name='inv_index_anchor')
    ids_and_titles = IndexReader().read_index(base_dir='anchors', name='ids_anchors')
    query_processor = QueryProcessor()
    query_as_tokens = query_processor.tokenize(query)
    for token in query_as_tokens:
        token_stats_dict = rdd_anchor_stats.filter(lambda x: list(x.keys())[0] == tolekn).collect()
        stats = token_stats_dict[token]
        for src_id, src_stats in stats.itmes():
            count_tok_in_src = src_stats['count']
            dest_set = src_stats['dest']
            res[src_id] = count_tok_in_src
    sorted_res = {k: v for k, v in sorted(res.items(), key=lambda item: item[1])}
    docs_title_pair = query_processor.id_to_title(ids_and_titles, list(sorted_res.keys()))
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
    ids_and_titles = IndexReader().read_index(base_dir='anchors', name='ids_anchors')
    query_processor = QueryProcessor()
    docs_title_pair = query_processor.id_to_title(ids_and_titles, list(wiki_ids))
    titles = [title for doc_id, title in docs_title_pair]
    contatct_info = "tomerm3399@gmail.com.com"
    p = PageviewsClient(user_agent="Python query script by " + contatct_info)
    start_date = datetime.date(2021, 8, 1)
    end_date = datetime.date(2021, 8, 31)
    senate_views = p.article_views(project='en.wikipedia',
                                   articles=titles,
                                   granularity='monthly',
                                   start=start_date,
                                   end=end_date)
    return jsonify(senate_views)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
