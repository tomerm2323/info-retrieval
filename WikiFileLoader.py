class WikiFileLoader:

    def __init__(self):
        self.wiki_files = []
        self.wiki_parquet_files = []


    def get_all_files_from_bucket(self):
        from google.cloud import storage
        # Put your bucket name below and make sure you can access it without an error
        bucket_name = 'shemiperetz-irproject' 
        full_path = f"gs://{bucket_name}/"
        paths=[]

        client = storage.Client()
        blobs = client.list_blobs(bucket_name)
        for b in blobs:
            if ".parquet" in b.name:
                paths.append(full_path+b.name)

        parquetFile = spark.read.parquet(*paths)
        return parquetFile

            

    def read_doc_list_from_files(self, doc_id_list):
        selected_rows = []

        # Iterate through the file names and read each file using pandas
        for df in self.wiki_parquet_files:
          # Select the rows with document IDs that are in the list
          selected_rows.appens(df.loc[df['id'].isin(doc_id_list)])

        # Save the selected rows to a new parquet file
        selected_rows.to_parquet('selected_rows.parquet')
        return selected_rows

    def read_test_queries_doc_id(self):
        with open('queries_train.json', 'r') as f:
            data = json.load(f)

        # Extract the values and remove duplicates
        values = data.values()
        return values
