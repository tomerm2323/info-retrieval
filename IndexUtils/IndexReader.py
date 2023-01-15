from contextlib import closing
import os
from pathlib import Path
import pickle
from google.cloud import storage
from MultiFileReader import MultiFileReader
from InvertedIndex import InvertedIndex

TUPLE_SIZE = 6

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
      return sc.pickleFile(f"file:///{path}")

  def delete_index(self, base_dir, name):
    path_globals = Path(base_dir) / f'{name}.pkl'
    path_globals.unlink()
    for p in Path(base_dir).rglob(f'{name}_*.bin'):
      p.unlink()

  
  def load_posting_lists_for_token(self, token, index: InvertedIndex, folder_name):
      locs = index.posting_locs[token]
      for f_name, pos in locs:
          name = f"{folder_name}_{f_name}"
          if not os.path.exists(name):
              bucket_name = "shemiperetz"
              client = storage.Client()
              bucket = client.get_bucket(bucket_name)
              bucket.get_blob(f"{folder_name}/{f_name}").download_to_filename(name)
      with closing(MultiFileReader(folder_name)) as reader:
          b = reader.read(locs, index.df[token] * TUPLE_SIZE)
          posting_list = []
          for i in range(index.df[token]):
              doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
              tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
              posting_list.append((doc_id, tf))
          return posting_list

