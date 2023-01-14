# Let's start with a small block size of 30 bytes just to test things out. 
BLOCK_SIZE = 1999998

class MultiFileWriter:
    """ Sequential binary writer to multiple files of up to BLOCK_SIZE each. """
    def __init__(self, base_dir, name, bucket_name):
        self._base_dir = Path(base_dir)
        self._name = name
        self._file_gen = (open(self._base_dir / f'{name}_{i:03}.bin', 'wb') 
                          for i in itertools.count())
        self._f = next(self._file_gen)
        # Connecting to google storage bucket. 
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        
    
    def write(self, b):
        locs = []
        while len(b) > 0:
            pos = self._f.tell()
            remaining = BLOCK_SIZE - pos
        # if the current file is full, close and open a new one.
            if remaining == 0:  
                self._f.close()
                self.upload_to_gcp()                
                self._f = next(self._file_gen)
                pos, remaining = 0, BLOCK_SIZE
            self._f.write(b[:remaining])
            locs.append((self._f.name, pos))
            b = b[remaining:]
        return locs

    def close(self):
        self._f.close()
    
    def upload_to_gcp(self):
        '''
            The function saves the posting files into the right bucket in google storage.
        '''
        file_name = self._f.name
        blob = self.bucket.blob(f"postings_gcp/{file_name}")
        blob.upload_from_filename(file_name)

