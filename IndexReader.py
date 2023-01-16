from contextlib import closing
import os
from pathlib import Path
import pickle
from google.cloud import storage
from MultiFileReader import MultiFileReader
from InvertedIndex import InvertedIndex
import pandas as pd
import gcsfs 

TUPLE_SIZE = 6
BLOCK_SIZE = 1999998


class CustomUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        try:
            return super().find_class(__name__, name)
        except AttributeError:
            return super().find_class(module, name)

class IndexReader:
  def __init__(self, invertedIndex):
    self.inv_index = invertedIndex

  def read_index(self, base_dir, name):
    with open(f'{name}.pkl', 'rb') as f:
      return CustomUnpickler(f).load()
    
  
  def read_rdd_from_binary_files(self, path, sc):
      return sc.pickleFile(f"{path}")

  def read_pagerank(self, path):
    df = pd.read_csv(path, names=["id", "pr"])
    return df


  def delete_index(self, base_dir, name):
    path_globals = Path(base_dir) / f'{name}.pkl'
    path_globals.unlink()
    for p in Path(base_dir).rglob(f'{name}_*.bin'):
      p.unlink()


  def token_posting_list(self, token, index: InvertedIndex, folder_name):
      locs = index.posting_locs[token]
      for f_name, pos in locs:
          name = f"{f_name}"
          if not os.path.exists(name):
              bucket_name = "208378042-irproject"
              client = storage.Client()
              bucket = client.get_bucket(bucket_name)
              bucket.get_blob(f"{folder_name}/{f_name}").download_to_filename(name)
              
      with closing(MultiFileReader()) as reader:
          if token in index.df.keys(): 
            df = index.df[token]
          else:
            return []
          b = reader.read(locs, df * TUPLE_SIZE)
          posting_list = []
          for i in range(df):
              doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
              tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
              posting_list.append((doc_id, tf))
          return posting_list
        
  def read_title_pl(self, locs, n_bytes, bucket_name, folder_name, w, inv_index):
      b = []
      for loc in locs:
          f_name = loc[0]
          offset = loc[1]
          fs = gcsfs.GCSFileSystem(project='ifass3')
          with fs.open(f'{bucket_name}/{folder_name}/{f_name}') as f:
              f.seek(offset)
              n_read = min(n_bytes, BLOCK_SIZE - offset)
              b.append(f.read(n_read))
              n_bytes -= n_read
      bArr = b''.join(b)
      posting_list = []
      df = inv_index.df[w]
      for i in range(df):
          doc_id = int.from_bytes(bArr[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
          tf = int.from_bytes(bArr[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
          posting_list.append((doc_id, tf))
      return posting_list
    

