import gc
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


def average_precision(ranked_list, relevant_docnos):
    """
    ranked_list: list of docnos in descending score order
    relevant_docnos: set of docnos considered 'relevant'

    Returns average precision for that ranking.
    """
    # “ranked_list” is e.g. [docno1, docno2, docno3 ... docnoN]
    # “relevant_docnos” is a set, e.g. {'D1','D7','D22'}
    num_relevant = 0
    precision_sum = 0.0
    for i, docno in enumerate(ranked_list, start=1):
        if docno in relevant_docnos:
            num_relevant += 1
            precision_sum += num_relevant / i
    if len(relevant_docnos) == 0:
        return 0.0
    return precision_sum / len(relevant_docnos)

def reciprocal_rank(ranked_list, relevant_docnos, k=None):
    """
    Returns reciprocal rank (1 / rank of first relevant doc),
    considering only the top k results if k is provided.
    """
    if k is not None:
        ranked_list = ranked_list[:k]
    for i, docno in enumerate(ranked_list, start=1):
        if docno in relevant_docnos:
            return 1.0 / i
    return 0.0

def ndcg(ranked_list, relevant_docnos, k=None):
    """
    Returns nDCG (with binary relevance) evaluated over the top k results.
    If k is not provided, the entire ranked_list is used.
    """
    if k is not None:
        ranked_list = ranked_list[:k]

    # Compute gains: 1 if the document is relevant, else 0
    gains = [1.0 if d in relevant_docnos else 0.0 for d in ranked_list]

    # Compute Discounted Cumulative Gain (DCG)
    dcg = 0.0
    for i, g in enumerate(gains, start=1):
        dcg += g / np.log2(i + 1)

    # Compute Ideal DCG (IDCG)
    # For binary relevance, the ideal ranking puts all relevant docs at the top.
    # The number of positions to fill is the length of the truncated list.
    ideal_count = min(len(relevant_docnos), len(ranked_list))
    ideal_gains = [1.0] * ideal_count + [0.0] * (len(ranked_list) - ideal_count)

    idcg = 0.0
    for i, g in enumerate(ideal_gains, start=1):
        idcg += g / np.log2(i + 1)

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


# def evaluate_in_stream(path_to_results, qrels_df, chunksize=500_000):
#     """
#     Read the BM25 CSV in chunks, compute per-query metrics on the fly.
#     qrels_df: must map (qid, docno) => relevance. Ideally 1 or 0.
#               If you have multiple relevance levels, treat >0 as relevant
#               or adapt the code to use graded nDCG, etc.
#     """
#     # Convert your qrels to a dictionary of sets
#     #    qrels_dict[qid] = {docno1, docno2, ...} for relevant docs
#     qrels_df['label'] = qrels_df['label'].astype(int)
#     grouped = qrels_df.groupby('qid')
#     qrels_dict = {
#         qid: set(group.loc[group['label'] > 0, 'docno'].tolist())
#         for qid, group in grouped
#     }
#
#     # We assume the results CSV has columns: qid, docno, score, (optionally rank).
#     # Must be sorted by qid, THEN by descending score or at least we must sort by score
#     # within each qid. If not sorted, we’ll sort each qid’s rows once we have them.
#     # Suppose the CSV is already sorted by qid desc or so, but typically you might need
#     # to do chunk.sort_values('score', ascending=False) after reading the chunk in memory.
#     # We also keep a buffer for leftover rows.
#     buffer_df = pd.DataFrame()
#     last_qid_in_chunk = None
#
#     # We'll store metric sums and a per-query count
#     sum_ap = 0.0
#     sum_ndcg = 0.0
#     sum_rr = 0.0
#     query_count = 0
#
#     # We’ll use an iterator to read the CSV in chunks
#     with pd.read_csv(path_to_results, chunksize=chunksize, dtype={'qid': str, 'docno': str, 'score': float}) as reader:
#         for chunk in tqdm(reader, desc="Reading results in chunks"):
#             # Prepend leftover buffer
#             if not buffer_df.empty:
#                 chunk = pd.concat([buffer_df, chunk], ignore_index=True)
#                 buffer_df = pd.DataFrame()
#
#             # Because it’s sorted by qid, the last qid might be incomplete
#             last_qid = chunk.iloc[-1]['qid']
#
#             # Find the first occurrence of that last qid
#             start_idx = chunk[chunk['qid'] == last_qid].index[0]
#             complete_part = chunk.iloc[:start_idx]     # guaranteed complete qids
#             buffer_part  = chunk.iloc[start_idx:]      # leftover partial qid
#
#             # Evaluate the complete qids in `complete_part`
#             # Group by qid
#             for qid, group_rows in complete_part.groupby('qid'):
#                 # Sort by descending score if not guaranteed sorted
#                 # group_rows = group_rows.sort_values('score', ascending=False)
#                 ranked_docs = group_rows['docno'].tolist()
#                 relevant_docs = qrels_dict.get(qid, set())
#
#                 # Compute metrics
#                 ap = average_precision(ranked_docs, relevant_docs)
#                 rr = reciprocal_rank(ranked_docs, relevant_docs)
#                 nd = ndcg(ranked_docs, relevant_docs, k=None)
#
#                 # Accumulate
#                 sum_ap += ap
#                 sum_ndcg += nd
#                 sum_rr += rr
#                 query_count += 1
#
#             # Keep the leftover partial qid
#             buffer_df = buffer_part
#
#     # After reading all chunks, buffer_df has the final leftover qid
#     if not buffer_df.empty:
#         for qid, group_rows in buffer_df.groupby('qid'):
#             # group_rows = group_rows.sort_values('score', ascending=False)
#             ranked_docs = group_rows['docno'].tolist()
#             relevant_docs = qrels_dict.get(qid, set())
#
#             ap = average_precision(ranked_docs, relevant_docs)
#             rr = reciprocal_rank(ranked_docs, relevant_docs)
#             nd = ndcg(ranked_docs, relevant_docs, k=None)
#
#             sum_ap += ap
#             sum_ndcg += nd
#             sum_rr += rr
#             query_count += 1
#
#     # Now compute aggregates
#     map_ = sum_ap / query_count if query_count > 0 else 0.0
#     mean_ndcg = sum_ndcg / query_count if query_count > 0 else 0.0
#     mean_rr = sum_rr / query_count if query_count > 0 else 0.0
#
#     metrics = {
#         "map": map_,
#         "ndcg": mean_ndcg,
#         "recip_rank": mean_rr,
#         "queries_evaluated": query_count,
#     }
#     return metrics

# def evaluate(path_to_results, qrles_df):
#     ## load results from file
#     results = pd.read_csv(path_to_results, sep=',', dtype={'qid': str, 'docno': str, 'score': float})
#
#     pt.Experiment(
#         [results],
#         qrles_df,  # Use rewritten queries
#         eval_metrics=[RR @ 10, nDCG @ 20, MAP],
#     )

def evaluate_in_stream(path_to_results, qrels_df, chunksize=100_000):
    """
    Streamed evaluation that:
      - Reads the BM25 CSV file in chunks
      - Uses total file size for a tqdm progress bar
      - Groups by qid so that we never lose continuity
      - Evaluates (AP, nDCG, RR) on-the-fly, without storing all results in memory

    The results file must be sorted by qid (and ideally by descending score within each qid).
    Columns expected in path_to_results: qid, docno, score, (optionally rank).
    """

    # 1) Convert qrels to dictionary-of-sets for quick relevant-doc lookups
    qrels_df['label'] = qrels_df['label'].astype(int)
    qrels_dict = {}
    for qid, group in tqdm(qrels_df.groupby('qid'), desc="Processing qrels"):
        # treat anything > 0 as relevant (typical binary approach)
        rel_docs = group.loc[group['label'] > 0, 'docno'].tolist()
        qrels_dict[qid] = set(rel_docs)

    # 2) We'll keep partial qid rows in this buffer
    buffer_df = pd.DataFrame()

    # 3) For aggregating metrics
    sum_ap = 0.0
    sum_ndcg_ = 0.0
    sum_rr = 0.0
    query_count = 0

    # 4) Set up the file-size-based progress bar
    filesize = os.path.getsize(path_to_results)
    with open(path_to_results, 'rb') as f, tqdm(
            total=filesize, unit='B', unit_scale=True, desc="Reading CSV"
    ) as pbar:
        # Use pd.read_csv on the binary file handle
        reader = pd.read_csv(
            f,
            sep=',',
            chunksize=chunksize,
            dtype={'qid': str, 'docno': str, 'score': float}
        )

        last_pos = 0
        for chunk_df in reader:
            # Update the progress bar by how many bytes we've advanced
            current_pos = f.tell()
            bytes_read = current_pos - last_pos
            pbar.update(bytes_read)
            last_pos = current_pos

            # Prepend leftover buffer from previous chunk
            if not buffer_df.empty:
                chunk_df = pd.concat([buffer_df, chunk_df], ignore_index=True)
                buffer_df = pd.DataFrame()

            # Identify the last qid in this chunk, which might be incomplete
            last_qid = chunk_df.iloc[-1]['qid']
            # Start index for that last qid
            start_idx = chunk_df[chunk_df['qid'] == last_qid].index[0]

            # The rows up to start_idx (exclusive) are complete qids
            complete_part = chunk_df.iloc[:start_idx]
            # The leftover part is partial (the last qid)
            buffer_part = chunk_df.iloc[start_idx:]

            # Evaluate all complete qids
            for qid, group_rows in complete_part.groupby('qid'):
                # sort by descending score if not already guaranteed sorted
                group_rows = group_rows.sort_values('score', ascending=False)
                ranked_docs = group_rows['docno'].tolist()
                relevant_docs = qrels_dict.get(qid, set())

                ap = average_precision(ranked_docs, relevant_docs)
                rr = reciprocal_rank(ranked_docs, relevant_docs, k=10)
                nd = ndcg(ranked_docs, relevant_docs, k=20)

                sum_ap += ap
                sum_ndcg_ += nd
                sum_rr += rr
                query_count += 1

            # Move the partial qid into buffer
            buffer_df = buffer_part

    # 5) After finishing all chunks, buffer_df still holds the last qid's rows
    if not buffer_df.empty:
        for qid, group_rows in buffer_df.groupby('qid'):
            group_rows = group_rows.sort_values('score', ascending=False)
            ranked_docs = group_rows['docno'].tolist()
            relevant_docs = qrels_dict.get(qid, set())

            ap = average_precision(ranked_docs, relevant_docs)
            rr = reciprocal_rank(ranked_docs, relevant_docs)
            nd = ndcg(ranked_docs, relevant_docs, k=None)

            sum_ap += ap
            sum_ndcg_ += nd
            sum_rr += rr
            query_count += 1

    # 6) Compute overall metrics
    map_ = sum_ap / query_count if query_count > 0 else 0.0
    mean_ndcg = sum_ndcg_ / query_count if query_count > 0 else 0.0
    mean_rr = sum_rr / query_count if query_count > 0 else 0.0

    return {
        'map': map_,
        'ndcg': mean_ndcg,
        'recip_rank': mean_rr,
        'queries_evaluated': query_count
    }

def doc_generator(df, chunk_size=None):
    error_count = 0
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Building index"):
        # Skip documents with null text
        if pd.isna(row.text):
            error_count += 1
            continue
        yield {'docno': row.docno, 'text': row.text}
    if error_count > 0:
        print(f"Skipped {error_count} documents with null text.")


def plot_results():
    # Data from CSV
    metrics = ['MAP', 'NDCG', 'Reciprocal Rank']
    # antique
    # doc_values = [0.33403741824881583,0.5488317950973486,0.8566959064327484]
    # bm25_values = [0.44460569950244383,0.5983482852072701,0.912570767195767]

    # ms marco
    doc_values = [0.15460591447933805,0.22356504664093496,0.15734730804094244]
    bm25_values = [0.17527114204981067,0.29806899894774774,0.17846779766971171]

    # Set the positions of the bars on the x-axis
    x = np.arange(len(metrics))
    width = 0.35  # width of the bars

    fig, ax = plt.subplots(figsize=(8, 6))

    # Create bars for each method
    rects1 = ax.bar(x - width/2, doc_values, width, label='docTTTTTquery')
    rects2 = ax.bar(x + width/2, bm25_values, width, label='BM25')

    # Add labels, title, and custom x-axis tick labels
    ax.set_ylabel('Scores')
    ax.set_title('Results evaluated on the MSMARCO Passage dataset')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    # Function to add labels on top of each bar
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.show()