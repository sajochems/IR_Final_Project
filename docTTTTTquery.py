import torch
import pandas as pd
import time
from tqdm import tqdm
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration
import pyterrier as pt
import os
import re
import ir_datasets

class DocT5Query:

    def __init__(self):
        print("Initializing DocT5Query")
        self.marco_documents_path = "datasets/MSMARCO/collection.tsv"
        self.marco_documents_appended_path = "datasets/MSMARCO/collection_appended.tsv"
        self.antique_documents_appended_path = "datasets/antique/collection_appended.tsv"
        self.marco_queries_path = "datasets/MSMARCO/queries.dev.tsv"
        self.marco_qrels_path = "datasets/MSMARCO/qrels.dev.tsv"
        self.index_dir = r"C:\Users\Wouter\Documents\Projects\IR_Final_Project\index"
        self.max_docs = 1000
        self.num_queries_to_append = 5
        self.batch_size = 16

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Using device:", self.device)

        self.tokenizer = T5Tokenizer.from_pretrained('castorini/doc2query-t5-base-msmarco')
        self.model = T5ForConditionalGeneration.from_pretrained('castorini/doc2query-t5-base-msmarco')
        self.model.to(self.device)

    def load_antique(self):
        print("Loading antique documents")
        antique = ir_datasets.load("antique/test")
        print("Loading antique finished")
        return antique

    def load_data(self):
        # Load documents, queries, and qrels from TSV files.
        # Documents: assumed columns: [docno, text]
        # Queries: assumed columns: [qid, query]
        # Qrels: assumed columns: [qid, docno, rel]
        print("loading documents")
        documents = pd.read_csv(self.marco_documents_path, sep='\t', header=None, names=['docno', 'text'])
        queries = pd.read_csv(self.marco_queries_path, sep='\t', header=None, names=['qid', 'query'])
        qrels = pd.read_csv(self.marco_qrels_path, sep='\t', header=None, usecols=[0,2], names=['qid', 'docno'])
        qrels['relevance'] = 1
        print("loaded data with shapes:")
        print("documents:", documents.shape)
        print("queries:", queries.shape)
        print("qrels:", qrels.shape)

        documents['docno'] = documents['docno'].astype(str)
        queries['qid'] = queries['qid'].astype(str)
        qrels['qid'] = qrels['qid'].astype(str)
        qrels['docno'] = qrels['docno'].astype(str)

        # Pre-process queries to remove or escape special characters (e.g., "/")
        queries['query'] = queries['query'].apply(self.clean_query)

        return documents, queries, qrels

    def build_index(self, documents):
        # PyTerrier expects an iterator of dictionaries.
        # Build an index from the documents dataframe using IterDictIndexer.
        print("building index")
        documents['docno'] = documents['docno'].astype(str)

        os.makedirs(self.index_dir, exist_ok=True)
        print("Index directory created at:", self.index_dir)
        indexer = pt.IterDictIndexer(self.index_dir)
        index_ref = indexer.index(documents.to_dict('records'), fields=['docno', 'text'])
        print("index built")
        return pt.IndexFactory.of(index_ref)

    def load_index(self, documents):
        if os.path.exists(self.index_dir) and os.path.isdir(self.index_dir):
            print("Loading index from disk.")
            return pt.IndexFactory.of(self.index_dir)
        else:
            print("Index directory does not exist. Building index from scratch.")
            return self.build_index(documents)

    def clean_query(self, query):
        # Remove punctuation that may cause Terrier's parser to choke.
        # This replaces any non-alphanumeric character (except whitespace) with a space.
        return re.sub(r'[^\w\s]', ' ', query)

    def evaluate_bm25(self):
        # Load data from files
        documents, queries, qrels = self.load_data()
        # Build the PyTerrier index from the document collection.
        index = self.load_index(documents)


        print("Index loaded")
        # Create a BM25 retrieval model using PyTerrier's BatchRetrieve.
        bm25 = pt.terrier.Retriever(index, wmodel="BM25")

        print("Running BM25 retrieval")
        # Retrieve results for the queries.
        results = bm25.transform(queries)

        # Evaluate the BM25 results.
        # The qrels dataframe should have columns: qid, docno, rel.
        print("Evaluating results")
        eval_results = pt.Utils.evaluate(results, qrels, metrics=["map", "ndcg", "recip_rank"])
        print("Evaluation Results:")
        print(eval_results)



    def load_marco(self):
        print("Loading marco dataset...")
        df = pd.read_csv(self.marco_documents_path, sep='\t', header=None, nrows=self.max_docs)
        print("Loaded marco dataset with shape:", df.shape)
        return df

    def generate_queries(self, df):
        # record time
        start = time.time()

        # Limit the number of documents to process
        df = df[:self.max_docs]
        docs = df.iloc[:, 1].tolist()  # Assuming the text is in the second column

        all_queries = []
        # Process documents in mini-batches
        for i in tqdm(range(0, len(docs), self.batch_size), desc="Generating queries"):
            batch_docs = docs[i : i + self.batch_size]
            # Batch tokenize with padding/truncation
            inputs = self.tokenizer(
                batch_docs,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)

            # Generate queries in batch
            outputs = self.model.generate(
                **inputs,
                max_length=64,
                do_sample=True,
                top_k=10,
                num_return_sequences=self.num_queries_to_append
            )

            # Reshape outputs: shape becomes (batch_size, num_return_sequences, sequence_length)
            outputs = outputs.view(len(batch_docs), self.num_queries_to_append, -1)

            # Decode generated outputs
            for batch_out in outputs:
                batch_queries = [
                    self.tokenizer.decode(seq, skip_special_tokens=True)
                    for seq in batch_out
                ]
                all_queries.append(batch_queries)

        # Convert the list of lists to a numpy array if needed
        queries = np.array(all_queries, dtype=object)

        # Print the time taken for query generation
        end = time.time()
        print("Generated", len(queries), "queries in", end - start, "seconds")

        return queries


    def example(self):
        doc_text = 'The presence of communication amid scientific minds was equally important to the success of the Manhattan Project as scientific intellect was. The only cloud hanging over the impressive achievement of the atomic researchers and engineers is what their success truly meant; hundreds of thousands of innocent lives obliterated.'

        input_ids = self.tokenizer.encode(doc_text, return_tensors='pt').to(self.device)
        outputs = self.model.generate(
            input_ids=input_ids,
            max_length=64,
            do_sample=True,
            top_k=10,
            num_return_sequences=3)

        for i in range(3):
            print(f'sample {i + 1}: {self.tokenizer.decode(outputs[i], skip_special_tokens=True)}')

    def create_appended(self):
        # Load and convert the iterator to a list for reuse.
        antique = self.load_antique()

        # self.max_docs = antique.docs_count()

        # doc_list = list(antique.docs_iter())

        # Create a DataFrame and generate queries.
        print("creating df")
        df = pd.DataFrame.from_records(self.safe_docs_iter(antique), columns=['docno', 'text'])
        queries = self.generate_queries(df)

        if len(queries) != len(df):
            print("Warning: The number of generated queries does not match the number of documents.")


    # Build the appended_docs list with a nested list comprehension, using tqdm for the outer loop.
        appended_docs = [
            {
                'docno': row.docno,
                'text': f"{row.text} " + " ".join(q)
            }
            for row, q in tqdm(zip(df.itertuples(index=False), queries),
                               total=min(len(df), len(queries)),
                               desc="Appending queries")
        ]

        # Save the appended documents to a TSV file.
        appended_df = pd.DataFrame(appended_docs)
        appended_df.to_csv(self.antique_documents_appended_path, sep='\t', index=False, header=False)
        print("Appended documents saved to:", self.antique_documents_appended_path)

        pass

    def safe_docs_iter(self, dataset):
        """
        Yields documents from dataset.docs_iter() while handling UnicodeDecodeError.
        If a document raises a UnicodeDecodeError, it is skipped.
        """
        it = dataset.docs_iter()
        while True:
            try:
                yield next(it)
            except (UnicodeDecodeError, RuntimeError) as e:
                print("Error encountered, skipping doc:", e)
                continue
            except StopIteration:
                break

if __name__ == "__main__":
    print("Running DocT5Query script")
    docT5Query = DocT5Query()
    # df = docT5Query.load_marco()
    # queries = docT5Query.generate_queries(df)
    # print("Queries:")
    # print(queries.shape)
    # print(queries[-10:])

    docT5Query.create_appended()

    # docT5Query.evaluate_bm25()