from collections import Counter, defaultdict

class Index:
  def __init__(self,base_dir,name,num_of_docs,df,file={}):
    self.docs = {}
    self.term_total = Counter()
    # stores posting list per term while building the index (internally), 
    # otherwise too big to store in memory.
    self._index_posting_list = defaultdict(list)
    # mapping a term to posting file locations, which is a list of 
    # (file_name, offset) pairs. Since posting lists are big we are going to
    # write them to disk and just save their location in this list. We are 
    # using the MultiFileWriter helper class to write fixed-size files and store
    # for each term/posting list its list of locations. The offset represents 
    # the number of bytes from the beginning of the file where the posting list
    # starts. 
    self.posting_locs = defaultdict(list)
    # self.writer = IndexWriter(base_dir,name)
    # self.reader = MultiFileReader()
    self.base_dir = base_dir
    self.name = name
    self.TUPLE_SIZE = 6
    self.TF_MASK = 2 ** 16 - 1
    self.N = num_of_docs # N is the amount of terms
    self.df = df


  def add_doc(self, doc_id, tokens):
    print(f"Index -> add_doc : Started on doc: {doc_id}")
    """ Adds a document to the index with a given `doc_id` and tokens. It counts
        the tf of tokens, then update the index (in memory, no storage 
        side-effects).

    parameters
    ----------
      tokens: list of non uniqe toekns form a doc.
    """
    w2cnt = Counter(tokens)
    self.term_total.update(w2cnt)
    self.docs[doc_id] = []
    num_of_terms = len(w2cnt.keys())
    for w, tf in w2cnt.items():
      self.docs.setdefault(doc_id,{"terms": []})
      norm_tf = tf / num_of_terms
      tfidf = self.calc_tfidf(word=w, tf=norm_tf)
      self.docs[doc_id]['terms'] += (w,tfidf)

  def calc_tfidf(self,word, tf):
    idf = np.log(self.N / self.df[word])
    tfidf = idf * tf
    return tfidf
      

  def get_byte_pl(self,doc_id):
    loc = self.posting_locs[doc_id]
    # read a certain number of bytes into variable b
    with closing(MultiFileReader()) as reader:
      b = reader.read(loc, self.df[doc_id] * self.TUPLE_SIZE)
      return b

  def byte_pl_to_list(self,pl_as_byte):
    pl = []
    
    for i in range(4,len(pl_as_byte), 6):
      term = int.from_bytes(pl_as_byte[i-4:i],'big')
      tfidf = int.from_bytes(pl_as_byte[i:i + 2],'big')
      pl.append((term,tfidf))
      
    return pl


  def add_file(self,json_file):
    print(f"Index -> add_file : Started ")
    """
    This method will get a json and add it to the Inverted index.
    The json file structure is {doc_id, list_of_tokens}
    
    parameters
    ----------
      json_file: dict
    """
    for doc_id, list_of_tokens in json_file.items():
      self.add_doc(doc_id,list_of_tokens)
      self.N += 1 # under the assumption that we only see every doc id onece.
    self.write()
  
  def update_docs(self, updates,update_name):
    """
    This method is updateing the index after it's already builet.
    By update we mean only add on top other parameters e.g pageRank

    parameters
    ----------
    updates: dict, key is a doc id and the value is an update
    update_name: str, how to call this update

    Example for after update:

    docs = {docId1: {"terms": [(word1,count1),(word2,count2)], "update_name": value}
            docId2: {"terms": [(word1,count1),(word3,count3)], "update_name": value}
              }
    """
    for doc_id, update_val in updates.items():
      self.docs.setdefault(doc_id,{"terms": [], f"{update_name}": 0})
      self.docs[doc_id][update_name] = update_val


  def write(self):
    print(f"Index -> write : Started ")
    """ Write the in-memory index to disk and populate the `posting_locs`
        variables with information about file location and offset of posting
        lists. Results in at least two files: 
        (1) posting files `name`XXX.bin containing the posting lists.
        (2) `name`.pkl containing the global term stats (e.g. df).
    """
    #### POSTINGS ####
    self.posting_locs = defaultdict(list)
    with closing(MultiFileWriter(self.base_dir, self.name)) as writer:
      for doc_id in sorted(self._index_posting_list.keys()):
        self.write_a_index_pl(doc_id, writer=writer, sort=True)
    #### GLOBAL DICTIONARIES ####
    self._write_globals()

  def write_a_index_pl(self, doc_id, writer, sort=False):
    print(f"Index -> write_a_index_pl : Started ")
    # sort the posting list by doc_id
    index_pl = self._index_posting_list[doc_id]
    if sort:
      index_pl = sorted(index_pl, key=itemgetter(0))
    # convert to bytes
    b = b''.join([(term << 16 | (tf & self.TF_MASK)).to_bytes(self.TUPLE_SIZE, 'big')
                  for term, tf in index_pl])
    # write to file(s)
    locs = self.writer.write(b)
    # save file locations to index
    self.posting_locs[doc_id].extend(locs) 


  def _write_globals(self):
    print(f"Index -> _write_globals : Started ")
    with open(Path(self.base_dir) / f'{self.name}.pkl', 'wb') as f:
      pickle.dump(self, f)

  def __getstate__(self):
    """ Modify how the object is pickled by removing the internal posting lists
        from the object's state dictionary. 
    """
    state = self.__dict__.copy()
    del state['_index_posting_list']
    return state

  def posting_list_iter(self):
    TUPLE_SIZE = 6 
    with closing(MultiFileReader()) as reader:
      for doc_id, locs in self.posting_locs.items():
        num_of_terms_in_doc = len(self.docs[doc_id]['terms'])
        # read a certain number of bytes into variable b
        b = reader.read(locs, num_of_terms_in_doc * TUPLE_SIZE)
        index_posting_list = []
        # convert the bytes read into `b` to a proper posting list.
        for i in range(4,len(b), 6):
          word = int.from_bytes(b[i-4:i],'big')
          tfidf = int.from_bytes(b[i:i + 2],'big')
          index_posting_list.append((word,tfidf))
        yield doc_id, index_posting_list

  @staticmethod
  def read_index(base_dir, name):
    with open(Path(base_dir) / f'{name}.pkl', 'rb') as f:
      return pickle.load(f)

  @staticmethod
  def delete_index(base_dir, name):
    path_globals = Path(base_dir) / f'{name}.pkl'
    path_globals.unlink()
    for p in Path(base_dir).rglob(f'{name}_*.bin'):
      p.unlink()