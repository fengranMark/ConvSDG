from re import T
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import sys
sys.path.append('..')
sys.path.append('.')
import argparse
import os
#from utils import check_dir_exist_or_build
from os import path
from os.path import join as oj
import numpy as np
import json
from pyserini.search.lucene import LuceneSearcher
#import pytrec_eval

def main():
    args = get_args()
    
    query_list = []
    qid_list = []
    output_file = oj(args.output_dir_path, args.query_type + "_topiocqa_topic_with_relp_bm25.jsonl")
    with open(args.input_query_path, "r") as fin:
        data = fin.readlines()

    n = len(data)
    
    for i in range(n):
        data_sample = json.loads(data[i])
        cur_query = data_sample["query"]
        cur_answer = data_sample["answer"]
        context = data_sample["context"]
        conv_query = ""
        for key in context:
            conv_query += key + ' '
        topic = data_sample["topic"]
        qid = str(data_sample["conversation_id"]) + '-' + str(data_sample["turn"])
        if args.query_type == "q":
            query = cur_query
        elif args.query_type == "q+a":
            query = cur_query + ' ' + cur_answer
        elif args.query_type == "q+a+topic":
            query = cur_query + ' ' + cur_answer + ' ' + topic
        elif args.query_type == "q+topic":
            query = cur_query + ' ' + topic
        elif args.query_type == "convq+topic":
            query = conv_query + topic

        query_list.append(query)
        qid_list.append(qid)
        
    # pyserini search
    logger.info("Start search relevant passages ...")
    searcher = LuceneSearcher(args.index_dir_path)
    searcher.set_bm25(args.bm25_k1, args.bm25_b)
    hits = searcher.batch_search(query_list, qid_list, k = args.top_k, threads = 20)

    with open(output_file, "w") as fout:
        for i in range(len(qid_list)):
            qid = qid_list[i]
            assert len(hits[qid]) == args.top_k
            pos_docs_pids = []
            pos_docs = []
            for idx in range(args.top_k):
                pos_docs_pids.append(hits[qid][idx].docid)
                pos_docs.append(json.loads(hits[qid][idx].raw)['contents'])
            record = json.loads(data[i])
            record['bm25_pos_docs'] = pos_docs
            record['bm25_pos_docs_pids'] = pos_docs_pids
            fout.write(json.dumps(record))
            fout.write('\n')

    logger.info("Finish")

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_query_path", type=str, default="../datasets/topic_topiocqa_dev.jsonl")
    parser.add_argument('--output_dir_path', type=str, default="../datasets/")
    parser.add_argument('--index_dir_path', type=str, default="../../ConvDR-main/datasets/topiocqa/indexes/bm25")
    #parser.add_argument('--index_dir_path', type=str, default="../../ConvDR-main/datasets/cast21/bm25_index")
    #parser.add_argument('--index_dir_path', type=str, default="../../ConvDR-main/datasets/qrecc/indexes/bm25")
    parser.add_argument("--top_k", type=int,  default=3)
    #parser.add_argument("--rel_threshold", type=int,  default=1)
    parser.add_argument("--bm25_k1", type=int,  default=1.4)
    parser.add_argument("--bm25_b", type=int,  default=0.8)
    parser.add_argument('--query_type', type=str, default="q+a+topic")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()
