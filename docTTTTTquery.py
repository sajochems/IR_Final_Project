import gc

import torch
import pandas as pd
import time

from ir_measures import RR, nDCG, MAP
from torch.ao.quantization import prepare_qat
from tqdm import tqdm
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration
import pyterrier as pt
import os
import re
import ir_datasets
import ftfy

import utils
from utils import evaluate_in_stream, doc_generator


class DocT5Query:

    def __init__(self):
        print("Initializing DocT5Query")
        self.marco_documents_path = "datasets/MSMARCO/collection.tsv"
        self.marco_documents_appended_path = "datasets/MSMARCO/collection_appended.tsv"
        self.antique_documents_appended_path = "datasets/antique/collection_appended_enc.tsv"
        self.marco_queries_path = "datasets/MSMARCO/queries.dev.tsv"
        self.marco_qrels_path = "datasets/MSMARCO/qrels.dev.tsv"
        self.index_dir = r"C:\Users\Wouter\Documents\Projects\IR_Final_Project\index"
        self.output_dir = r"C:\Users\Wouter\Documents\Projects\IR_Final_Project\output"
        self.final_save_path = "datasets/MSMARCO/collection_appended_enc_fix_full.tsv"
        self.bm25_out_path = "bm25_results.csv"
        self.max_docs = 1000
        self.num_queries_to_append = 5
        self.batch_size = 16
        self.batch_size_bm25 = 1000

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Using device:", self.device)

        self.tokenizer = T5Tokenizer.from_pretrained('castorini/doc2query-t5-base-msmarco')
        self.model = T5ForConditionalGeneration.from_pretrained('castorini/doc2query-t5-base-msmarco')
        self.model.to(self.device)

    def load_antique(self):
        print("Loading antique documents")
        antique = ir_datasets.load("antique/train")
        print("Loading antique finished")
        return antique


    def load_marco_ir(self):
        # Load the MARCO dataset using ir_datasets
        print("Loading MARCO dataset")
        marco = ir_datasets.load("msmarco-passage/dev")
        print("Loading MARCO finished")
        print("Documents count:", marco.docs_count())
        print("Queries count:", marco.queries_count())
        print("Qrels count:", marco.qrels_count())
        # print("Documents metadata:", marco.docs_metadata())
        # print("Queries metadata:", marco.queries_metadata())
        # print("Qrels metadata:", marco.qrels_metadata())
        return marco

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



    def build_index(self, documents, dirname):
        chunk_size = 100000
        # PyTerrier expects an iterator of dictionaries.
        # Build an index from the documents dataframe using IterDictIndexer.
        start = time.time()
        print("building index")
        documents['docno'] = documents['docno'].astype(str)

        path = os.path.join(self.index_dir, dirname)
        os.makedirs(path, exist_ok=True)
        print("Index directory created at:", path)
        indexer = pt.IterDictIndexer(path, verbose=False)
        # index_ref = indexer.index(documents.to_dict('records'), fields=['docno', 'text'])
        index_ref = indexer.index(doc_generator(documents, chunk_size), fields=['docno', 'text'])

        end = time.time()
        print("Index built in", end - start, "seconds")
        return pt.IndexFactory.of(index_ref)

    def load_index(self, documents, dirname):
        path = os.path.join(self.index_dir, dirname)
        if os.path.exists(path) and os.path.isdir(path):
            print("Loading index from disk.")
            return pt.IndexFactory.of(path)
        else:
            print("Index directory does not exist. Building index from scratch.")
            return self.build_index(documents, dirname)

    # def clean_query(self, query):
    #     # Remove punctuation that may cause Terrier's parser to choke.
    #     # This replaces any non-alphanumeric character (except whitespace) with a space.
    #     return re.sub(r'[^\w\s]', ' ', query)

    def clean_query(self, query):
        query = query.replace("\n", " ").replace("\r", " ").replace("?", "")
        query = query.encode("ascii", "ignore").decode()
        query = re.sub(r"[\"`]", "", query)
        query = re.sub(r"\s+", " ", query).strip()
        query = re.sub(r"[^\w\s]", "", query)
        return query

    def inference_bm25(self, dataset_name):
        # Load data from files
        # documents, queries, qrels = self.load_data()
        doc_path = None
        if dataset_name == 'msmarco':
            dataset = self.load_marco_ir()
            dirname_index = "marco_index"
            dirname = "marco"
        elif dataset_name == 'msmarco_appended':
            dataset = self.load_marco_ir()
            dirname_index = "marco_appended_index"
            dirname = "marco_appended"
            # doc_path = self.marco_documents_appended_path
            doc_path = self.final_save_path
        elif dataset_name == 'antique_appended':
            dataset = self.load_antique()
            dirname_index = "antique_appended_index"
            dirname = "antique_appended"
            doc_path = self.antique_documents_appended_path
        else:
            dataset = self.load_antique()
            dirname_index = "antique_index"
            dirname = "antique"
            dataset_name = 'antique'

        print("loaded dataset:", dataset_name)
        print("Documents count:", dataset.docs_count())
        print("Queries count:", dataset.queries_count())
        print("Qrels count:", dataset.qrels_count())


        print("Loading qrels and queries")
        # max(len(text.encode("utf-8")) for _, text in dataset.docs)
        # queries = pd.DataFrame.from_records(self.safe_iter(dataset.queries_iter()), columns=['qid', 'query'])
        queries = pd.DataFrame.from_records(dataset.queries_iter(), columns=['qid', 'query'])
        queries['query'] = queries['query'].apply(self.clean_query)
        qrels = pd.DataFrame.from_records(dataset.qrels_iter(), columns=['qid', 'docno', 'label', 'iteration'])
        qrels = qrels.drop(columns=['iteration'])
        qrels['qid'] = qrels['qid'].astype(str)
        # qrels = qrels.rename(columns={"label": "relevance"})

        # print("Documents shape:", documents.shape)
        # print("Queries shape:", queries.shape)
        # print("Qrels shape:", qrels.shape)


        path = os.path.join("output", dirname, self.bm25_out_path)
        if not os.path.exists(path):
            print("converting documents to dataframe")
            if dataset_name == 'msmarco' or dataset_name == 'antique':
                # documents = pd.DataFrame.from_records(self.safe_iter(dataset.docs_iter()), columns=['docno', 'text'])
                documents = pd.DataFrame.from_records(dataset.docs_iter(), columns=['docno', 'text'])
            else:
                documents = pd.read_csv(doc_path, sep='\t', header=None, dtype={'docno': str, 'text': str}, names=['docno', 'text'])
                print("Columns:", documents.columns)


            index = self.load_index(documents, dirname_index)
            print("Index loaded")
            # Create a BM25 retrieval model using PyTerrier's BatchRetrieve.
            bm25 = pt.terrier.Retriever(index, wmodel="BM25", verbose=False, threads=1)

            print("Running BM25 retrieval")
            print("Amount of queries:", len(queries))
            print("Batch size:", self.batch_size_bm25)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            header_written = False
            for i in tqdm(range(0, len(queries), self.batch_size_bm25), desc="BM25 retrieval"):
                batch = queries.iloc[i:i+self.batch_size_bm25]
                batch_result = bm25.transform(batch)
                batch_result.to_csv(path, mode='a', index=False, header=not header_written)
                # print("Finished batch ", i, "of", int(len(queries) / self.batch_size_bm25), "and wrote results to", path)
                header_written = True
                del batch_result
            print("Finished BM25 retrieval and wrote results to", path)
                # batch_results.append(bm25.transform(batch))
            gc.collect()


    def evaluate_bm25_chunk(self, dataset_name):
        doc_path = None
        if dataset_name == 'msmarco':
            dataset = self.load_marco_ir()
            dirname_index = "marco_index"
            dirname = "marco"
        elif dataset_name == 'msmarco_appended':
            dataset = self.load_marco_ir()
            dirname_index = "marco_appended_index"
            dirname = "marco_appended"
            # doc_path = self.marco_documents_appended_path
            doc_path = self.final_save_path
        elif dataset_name == 'antique_appended':
            dataset = self.load_antique()
            dirname_index = "antique_appended_index"
            dirname = "antique_appended"
            doc_path = self.antique_documents_appended_path
        else:
            dataset = self.load_antique()
            dirname_index = "antique_index"
            dirname = "antique"
            dataset_name = 'antique'

        path = os.path.join("output", dirname, self.bm25_out_path)
        assert os.path.exists(path)

        queries = pd.DataFrame.from_records(dataset.queries_iter(), columns=['qid', 'query'])
        queries['query'] = queries['query'].apply(self.clean_query)
        qrels = pd.DataFrame.from_records(dataset.qrels_iter(), columns=['qid', 'docno', 'label', 'iteration'])
        qrels = qrels.drop(columns=['iteration'])
        qrels['qid'] = qrels['qid'].astype(str)

        chunksize = 100_000

        print("evaluating in chunks")

        eval_results = evaluate_in_stream(path, qrels, chunksize=chunksize)


        # results = pd.read_csv(path, sep=',', dtype={'qid': str, 'docno': str, 'score': float})
        # documents = pd.DataFrame.from_records(dataset.docs_iter(), columns=['docno', 'text'])
        # index = self.load_index(documents, dirname_index)
        # bm25 = pt.terrier.Retriever(index, wmodel="BM25", verbose=False, threads=1)

        # eval_results = pt.Experiment(
        #     [bm25],
        #     queries,
        #     qrels,  # Use rewritten queries
        #     eval_metrics=[RR @ 10, nDCG @ 20, MAP],
        # )

        # eval_results = pt.Evaluate(results, qrels, metrics=[RR @ 10, nDCG @ 20, MAP])

        date_time = time.strftime("%Y%m%d-%H%M%S")
        output_path = os.path.join(self.output_dir, f"{dataset_name}_eval_results_{date_time}.csv")
        os.makedirs(self.output_dir, exist_ok=True)
        df = pd.DataFrame([eval_results])
        # df = eval_results.drop(columns=["name"])

        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")

        return eval_results


    def load_marco(self):
        print("Loading marco dataset...")
        df = pd.read_csv(self.marco_documents_path, sep='\t', header=None, nrows=self.max_docs)
        print("Loaded marco dataset with shape:", df.shape)
        return df

    def generate_queries(self, df):
        # record time
        start = time.time()

        # Limit the number of documents to process
        # df = df[:self.max_docs]
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

    def create_appended(self, dataset, save_path, limit_docs=True):
        if not limit_docs:
            self.max_docs = dataset.docs_count()

        # Create a DataFrame and generate queries.
        print("creating df")
        df = pd.DataFrame.from_records(dataset.docs_iter(), columns=['docno', 'text'])
        print("created df with shape:", df.shape)
        queries = self.generate_queries(df)

        if len(queries) != len(df):
            print("Warning: The number of generated queries does not match the number of documents.")
            print("  Generated queries shape:", queries.shape)
            print("  Documents shape:", df.shape)


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
        appended_df.to_csv(save_path, sep='\t', index=False, header=False)
        print("Appended documents saved to:", save_path)

    def hotfix_appended(self):
        marco_documents_appended_path = "datasets/MSMARCO/collection_appended_enc.tsv"
        # antique_documents_appended_path = "datasets/antique/collection_appended_enc.tsv"

        new_marco_documents_appended_path = "datasets/MSMARCO/collection_appended_enc_fix.tsv"
        # new_antique_documents_appended_path = "datasets/antique/collection_appended_enc_fix.tsv"

        print("Loading MARCO documents")
        df = pd.read_csv(marco_documents_appended_path, sep='\t', header=None, keep_default_na=False)
        print("Loaded MARCO documents with shape:", df.shape)
        df[1] = df[1].apply(lambda x: ftfy.fix_text(x) if isinstance(x, str) else x)

        df.to_csv(new_marco_documents_appended_path, sep='\t', index=False, header=False)
        print("Appended documents saved to:", new_marco_documents_appended_path)
        pass

    def calc_missing(self):
        new_marco_documents_appended_path = "datasets/MSMARCO/collection_appended_enc_fix.tsv"
        print("Loading MARCO documents")
        df = pd.read_csv(new_marco_documents_appended_path, sep='\t', header=None, keep_default_na=False)
        print("Loaded MARCO documents with shape:", df.shape)
        #
        current_doc_ids = df[0].tolist()
        # del df
        #
        dataset = self.load_marco_ir()
        #
        all_doc_ids = []
        for doc in tqdm(dataset.docs_iter(), desc="Loading all doc ids", total=dataset.docs_count()):
            all_doc_ids.append(doc.doc_id)
        #
        missing_doc_ids = set(all_doc_ids) - set(current_doc_ids)
        print("Missing doc ids:", len(missing_doc_ids))
        print("Current doc ids:", len(current_doc_ids))
        print("All doc ids:", len(all_doc_ids))
        #
        ## save the missing doc ids to csv
        missing_doc_ids_df = pd.DataFrame(missing_doc_ids, columns=['docno'])
        missing_doc_ids_df.to_csv("missing_doc_ids.csv", index=False)

    def calc_missing_antique(self):
        new_antique_documents_appended_path = "datasets/antique/collection_appended_enc.tsv"
        print("Loading antique documents")
        df = pd.read_csv(new_antique_documents_appended_path, sep='\t', header=None, keep_default_na=False)
        print("Loaded antique documents with shape:", df.shape)
        #
        current_doc_ids = df[0].tolist()
        # del df
        #
        dataset = self.load_antique()
        #
        all_doc_ids = []
        for doc in tqdm(dataset.docs_iter(), desc="Loading all doc ids", total=dataset.docs_count()):
            all_doc_ids.append(doc.doc_id)
        #
        missing_doc_ids = set(all_doc_ids) - set(current_doc_ids)
        print("Missing doc ids:", len(missing_doc_ids))
        print("Current doc ids:", len(current_doc_ids))
        print("All doc ids:", len(all_doc_ids))
        #
        ## save the missing doc ids to csv
        missing_doc_ids_df = pd.DataFrame(missing_doc_ids, columns=['docno'])
        missing_doc_ids_df.to_csv("missing_antique_doc_ids.csv", index=False)

    def interp_appended(self):
        new_marco_documents_appended_path = "datasets/MSMARCO/collection_appended_enc_fix.tsv"
        current_df = pd.read_csv(new_marco_documents_appended_path, sep='\t', header=None, keep_default_na=False)
        print("Loaded collection_appended_enc_fix documents with shape:", current_df.shape)
        # # load the missing doc ids
        # missing_doc_ids_df = pd.read_csv("missing_doc_ids.csv", sep=',', header=0)
        # # sort
        # missing_doc_ids_df = missing_doc_ids_df.sort_values(by='docno')
        # # print(missing_doc_ids_df[:100])
        # missing_doc_ids_df = missing_doc_ids_df['docno']

        # print(len(missing_doc_ids_df))
        # self.max_docs = len(missing_doc_ids_df)

        dataset = self.load_marco_ir()
        df_full = pd.DataFrame.from_records(dataset.docs_iter(), columns=['docno', 'text'])
        print("Loaded MARCO documents with shape:", df_full.shape)

        df = df_full[~df_full['docno'].isin(current_df[0])]
        print("new_df shape:", df.shape)
        # missing_iter = dataset.docs_store().get_many_iter(missing_doc_ids_df.tolist())

        # self.create_appended(missing_iter, save_path="datasets/MSMARCO/collection_appended_fix_full.tsv")
        save_path = "datasets/MSMARCO/collection_appended_fix_part2.tsv"


        # Create a DataFrame and generate queries.
        # print("creating df")
        #
        # print("created df with shape:", df.shape)
        # df = df[df['docno'].isin(missing_doc_ids_df.tolist())]

        # print("!!created df with shape:", df.shape)
        queries = self.generate_queries(df)

        if len(queries) != len(df):
            print("Warning: The number of generated queries does not match the number of documents.")
            print("  Generated queries shape:", queries.shape)
            print("  Documents shape:", df.shape)


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
        appended_df.to_csv(save_path, sep='\t', index=False, header=False)
        print("Appended documents saved to:", save_path)

        ## combine the two files
        combined_df = pd.concat([current_df, appended_df], ignore_index=True)
        combined_df.sort_values(by=['docno'], inplace=True)
        combined_df.to_csv(self.final_save_path, sep='\t', index=False, header=False)
        
        del combined_df, appended_df, current_df, df_full

    def test(self):
        print("Testing")
        dataset = self.load_marco_ir()
        doc_count = dataset.docs_count()
        # part1 = "datasets/MSMARCO/collection_appended_enc_fix.tsv"
        # part2 = "datasets/MSMARCO/collection_appended_fix_part2.tsv"
        # df1 = pd.read_csv(part1, sep='\t', header=None, keep_default_na=False)
        # print("Loaded collection_appended_enc_fix documents with shape:", df1.shape)
        # df2 = pd.read_csv(part2, sep='\t', header=None, keep_default_na=False)
        # print("Loaded collection_appended_fix_part2 documents with shape:", df2.shape)
        # print("Nan values in part1:", df1.isnull().sum().sum())
        # print("Nan values in part2:", df2.isnull().sum().sum())

        # check if the number of documents in part1 + part2 is equal to the number of documents in the dataset
        # print("Number of documents in part1 + part2:", df1.shape[0] + df2.shape[0])
        # print("Number of documents in dataset:", doc_count)

        df3 = pd.read_csv(self.final_save_path, sep='\t', header=None, keep_default_na=False)
        print("Loaded collection_appended_enc_fix_full documents with shape:", df3.shape)
        print("Nan values in part3:", df3.isnull().sum().sum())
        print("empty values in part3:", df3.isna().sum().sum())
        print("Number of documents in part3:", df3.shape[0])
        print("Number of documents in dataset:", doc_count)



if __name__ == "__main__":
    print("Running DocT5Query script")
    docT5Query = DocT5Query()
    # docT5Query.test()
    # dataset = docT5Query.load_marco_ir()
    # print("Creating appended document Antique")
    # docT5Query.create_appended(docT5Query.load_antique(), docT5Query.antique_documents_appended_path, limit_docs=False)
    # print("Creating appended document MARCO")
    # docT5Query.create_appended(docT5Query.load_marco_ir(), docT5Query.marco_documents_appended_path, limit_docs=False)
    # print("Queries:")
    # print(queries.shape)
    # print(queries[-10:])
    # dataset = docT5Query.load_antique()

    # queries = pd.DataFrame.from_records(dataset.queries_iter())
    # qrels = pd.DataFrame.from_records(dataset.qrels_iter(), columns=['qid', 'docno', 'label', 'iteration'])

    # print(qrels.head())
    # print(qrels['label'].unique())
    # print("Queries:")
    # print(dataset.queries_metadata())
    # print(dataset.qrels_metadata())

    # docT5Query.create_appended(docT5Query.load_marco_ir(), docT5Query.marco_documents_appended_path, limit_docs=False)

    # df = pd.read_json("hf://datasets/macavaney/codec/documents.jsonl.gz", lines=True)

    # docT5Query.create_appended(ir_datasets.load("codec"), "datasets/codec/collection_appended.tsv")

    # docT5Query.interp_appended()

    print("Evaluating BM25")
    docT5Query.inference_bm25("antique")
    docT5Query.inference_bm25("antique_appended")
    docT5Query.inference_bm25("msmarco")
    docT5Query.inference_bm25("msmarco_appended")

    docT5Query.evaluate_bm25_chunk("antique")
    docT5Query.evaluate_bm25_chunk("antique_appended")
    docT5Query.evaluate_bm25_chunk("msmarco")
    docT5Query.evaluate_bm25_chunk("msmarco_appended")




    # docT5Query.hotfix_appended()
    # docT5Query.interp_appended()
    # docT5Query.calc_missing()

    # utils.plot_results()


