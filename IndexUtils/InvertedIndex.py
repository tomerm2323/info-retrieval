from collections import Counter, defaultdict
from contextlib import closing
from operator import itemgetter
from pathlib import Path
import pickle
import numpy as np
from MultiFileWriter import MultiFileWriter
from MultiFileReader import MultiFileReader

class InvertedIndex:  
  def __init__(self,base_dir,name,docs={}):
    """ Initializes the inverted index and add documents to it (if provided).
    Parameters:
    -----------
      docs: dict mapping doc_id to list of tokens
    """
    # stores document frequency per term
    self.df = Counter()
    # stores total frequency per term
    self.term_total = Counter()
    # stores posting list per term while building the index (internally), 
    # otherwise too big to store in memory.
    self._posting_list = defaultdict(list)
    # mapping a term to posting file locations, which is a list of 
    # (file_name, offset) pairs. Since posting lists are big we are going to
    # write them to disk and just save their location in this list. We are 
    # using the MultiFileWriter helper class to write fixed-size files and store
    # for each term/posting list its list of locations. The offset represents 
    # the number of bytes from the beginning of the file where the posting list
    # starts. 
    self.posting_locs = defaultdict(list)
    # self.writer = InvertedIndexWriter(base_dir,name)
    # self.reader = MultiFileReader()
    self.base_dir = base_dir
    self.name = name
    self.add_file(docs)
    self.TUPLE_SIZE = 6
    self.TF_MASK = 2 ** 16 - 1
    self.N = 0 # N is the amount of terms

  def add_doc(self, doc_id, tokens):
    print(f"InvertedIndex -> add_doc : Started on doc: {doc_id}")
    """ Adds a document to the index with a given `doc_id` and tokens. It counts
        the tf of tokens, then update the index (in memory, no storage 
        side-effects).
    """
    w2cnt = Counter(tokens)
    self.term_total.update(w2cnt)
    for w, cnt in w2cnt.items():
      self.df[w] = self.df.get(w, 0) + 1
      self._posting_list[w].append((doc_id, cnt))

  def add_file(self,json_file):
    print(f"InvertedIndex -> add_file : Started")
    """
    This method will get a json and add it to the Inverted index.
    The json file structure is {doc_id, list_of_tokens}
    
    parameters
    ----------
      json_file: dict
    """
    for doc_id, list_of_tokens in json_file.items():
      self.add_doc(doc_id,list_of_tokens)
      self.N += 1

    self.write()
    self._posting_list = defaultdict(list)
  
  def get_byte_pl(self,word):
    loc = self.posting_locs[word]
    # read a certain number of bytes into variable b
    with closing(MultiFileReader()) as reader:
      b = reader.read(loc, self.df[word] * self.TUPLE_SIZE)
      return b
  def byte_pl_to_list(self,pl_as_byte):
    pl = []
    
    for i in range(4,len(pl_as_byte), 6):
      doc_id = int.from_bytes(pl_as_byte[i-4:i],'big')
      tf = int.from_bytes(pl_as_byte[i:i + 2],'big')
      pl.append((doc_id,tf))

    return pl

  def calc_tfidf(self,word,doc_id):
    """
    This method was built for if we want to go over the inverted index after and calca tfidf
    """
    idf = np.log(self.N / self.df[word])
    tfidf = 0
    pl_as_byte = self.get_byte_pl(word)
    for i in range(4,len(pl_as_byte), 6):
      temp_doc_id = int.from_bytes(pl_as_byte[i-4:i],'big')
      if temp_doc_id == doc_id:
        tf = int.from_bytes(pl_as_byte[i:i + 2],'big')
        tfidf = tf * idf
        break
    return tfidf
            


  def write(self):
    print(f"InvertedIndex -> write : Started")
    """ Write the in-memory index to disk and populate the `posting_locs`
        variables with information about file location and offset of posting
        lists. Results in at least two files: 
        (1) posting files `name`XXX.bin containing the posting lists.
        (2) `name`.pkl containing the global term stats (e.g. df).
    """
    #### POSTINGS ####
    self.posting_locs = defaultdict(list)
    with closing(MultiFileWriter(self.base_dir, self.name)) as writer:
      # iterate over posting lists in lexicographic order
      for w in sorted(self._posting_list.keys()):
        self._write_a_posting_list(w, writer, sort=True)
    #### GLOBAL DICTIONARIES ####
    self._write_globals()

  def _write_globals(self):
    print(f"InvertedIndex -> _write_globals : Started")
    with open(Path(self.base_dir) / f'{self.name}.pkl', 'wb') as f:
      pickle.dump(self, f)

  def _write_a_posting_list(self, w, writer, sort=False):
    print(f"InvertedIndex -> _write_a_posting_list : Started")
    # sort the posting list by doc_id
    pl = self._posting_list[w]
    if sort:
      pl = sorted(pl, key=itemgetter(0))
    # convert to bytes
    b = b''.join([(doc_id << 16 | (tf & self.TF_MASK)).to_bytes(self.TUPLE_SIZE, 'big')
                  for doc_id, tf in pl])
    # write to file(s)
    locs = writer.write(b)
    # save file locations to index
    self.posting_locs[w].extend(locs) 

  def __getstate__(self):
    """ Modify how the object is pickled by removing the internal posting lists
        from the object's state dictionary. 
    """
    state = self.__dict__.copy()
    del state['_posting_list']
    return state

  def posting_list_iter(self):
    with closing(MultiFileReader()) as reader:
      for w, locs in self.posting_locs.items():
        # read a certain number of bytes into variable b
        b = reader.read(locs, self.df[w] * self.TUPLE_SIZE)
        posting_list = []
        # convert the bytes read into `b` to a proper posting list.
        for i in range(4,len(b), 6):
          doc_id = int.from_bytes(b[i-4:i],'big')
          tf = int.from_bytes(b[i:i + 2],'big')
          posting_list.append((doc_id,tf))
        yield w, posting_list
