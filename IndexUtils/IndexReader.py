import pickle


class IndexReader:
  @staticmethod
  def read_index(base_dir, name):
    with open(Path(base_dir) / f'{name}.pkl', 'rb') as f:
      return pickle.load(f)
    
  
  @staticmethod
  def read_rdd_from_binary_files(path):
      return spark.sparkContext.pickleFile(f"file:///{path}")

  @staticmethod
  def delete_index(base_dir, name):
    path_globals = Path(base_dir) / f'{name}.pkl'
    path_globals.unlink()
    for p in Path(base_dir).rglob(f'{name}_*.bin'):
      p.unlink()

