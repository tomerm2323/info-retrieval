from collections import Counter, defaultdict
from contextlib import closing
from operator import itemgetter
from pathlib import Path
import pickle
import numpy as np
from MultiFileWriter import MultiFileWriter
from MultiFileReader import MultiFileReader

class InvertedIndex:  
    def __init__(self, docs={}):
        """ Initializes the inverted index and add documents to it (if provided).
        Parameters:
        -----------
          docs: dict mapping doc_id to list of tokens
        """
        self.df = Counter()
        self.term_total = Counter()
        self._posting_list = defaultdict(list)
        self.posting_locs = defaultdict(list)
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
        self.doc_stats[doc_id] = w2cnt
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

    def add_docs_stats_from_file(self,json_file):
        print(f"InvertedIndex -> add_docs_stats_from_file : Started")
        """
        This method will get a json and add it to the Inverted index.
        The json file structure is {doc_id, list_of_tokens}

        parameters
        ----------
          json_file: dict
        """
        for doc_id, list_of_tokens in json_file.items():
            self.add_to_doc_stats(doc_id,list_of_tokens)

    def calc_tfidf(self,word, tf):
        idf = np.log(self.N / self.df[word])
        tfidf = idf * tf
        return tfidf
      
    def add_to_doc_stats(self, doc_id,list_of_tokens):
        w2cnt = Counter(list_of_tokens)
        vec_size = 0 
        for word, tf in w2cnt.items():
            vec_size = self.calc_tfidf(word,tf) ** 2
        vec_size = np.sqrt(vec_size)
        self.doc_stats[doc_id] = vec_size



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
        self.write_globals()

        
    def write_globals(self, base_dir, name):
        print(f"InvertedIndex -> _write_globals : Started")
        with open(Path(base_dir) / f'{name}.pkl', 'wb') as f:
            pickle.dump(self, f)

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


    @staticmethod
    def write_a_posting_list(b_w_pl, bucket_name):
        posting_locs = defaultdict(list)
        bucket_id, list_w_pl = b_w_pl
        
        with closing(MultiFileWriter(".", bucket_id, bucket_name)) as writer:
            for w, pl in list_w_pl: 
                # convert to bytes
                b = b''.join([(doc_id << 16 | (tf & TF_MASK)).to_bytes(TUPLE_SIZE, 'big')
                              for doc_id, tf in pl])
                # write to file(s)
                locs = writer.write(b)
                # save file locations to index
                posting_locs[w].extend(locs)
            writer.upload_to_gcp() 
            InvertedIndex._upload_posting_locs(bucket_id, posting_locs, bucket_name)
        return bucket_id
    
    @staticmethod
    def _upload_posting_locs(bucket_id, posting_locs, bucket_name):
        with open(f"{bucket_id}_posting_locs.pickle", "wb") as f:
            pickle.dump(posting_locs, f)
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob_posting_locs = bucket.blob(f"postings_gcp/{bucket_id}_posting_locs.pickle")
        blob_posting_locs.upload_from_filename(f"{bucket_id}_posting_locs.pickle")
