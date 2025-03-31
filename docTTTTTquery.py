import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration

class DocT5Query:

    def __init__(self):
        self.marco_path = "datasets/MSMARCO.tar"
        self.max_docs = 10000
        self.num_queries_to_append = 3

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Using device:", self.device)

        self.tokenizer = T5Tokenizer.from_pretrained('castorini/doc2query-t5-base-msmarco')
        self.model = T5ForConditionalGeneration.from_pretrained('castorini/doc2query-t5-base-msmarco')
        self.model.to(self.device)


    def load_marco(self):
        print("Loading marco dataset...")
        df = pd.read_csv(self.marco_path, sep='\t', header=None)
        print("Loaded marco dataset with shape:", df.shape)
        return df

    def append_queries(self, df, batch_size=64):
        # Limit the number of documents to process
        df = df[:self.max_docs]
        docs = df.iloc[:, 1].tolist()  # Assuming the text is in the second column

        all_queries = []
        # Process documents in mini-batches
        for i in tqdm(range(0, len(docs), batch_size)):
            batch_docs = docs[i : i + batch_size]
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



if __name__ == "__main__":
    docT5Query = DocT5Query()
    df = docT5Query.load_marco()
    queries = docT5Query.append_queries(df)
    print("Queries:")
    print(queries[:-10])