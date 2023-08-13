from IPython import embed
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import sys
sys.path.append('..')
sys.path.append('.')

import csv
import argparse
from models import ANCE
from utils import check_dir_exist_or_build, pstore, pload, set_seed, get_optimizer
from data import padding_seq_to_same_length, Retrieval_topiocqa, Search_Rel_Passage
import os
from os import path
from os.path import join as oj
import json
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
import faiss
import time
import copy
import pickle
import torch
import numpy as np
import pytrec_eval
from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer

'''
Test process, perform dense retrieval on collection (e.g., MS MARCO):
1. get args
2. establish index with Faiss on GPU for fast dense retrieval
3. load the model, build the test query dataset/dataloader, and get the query embeddings. 
4. iteratively searched on each passage block one by one to got the retrieved scores and passge ids for each query.
5. merge the results on all pasage blocks
6. output the result
'''

def load_collection(collection_file, title = False):
    all_passages = {}
    ext = collection_file[collection_file.rfind(".") + 1:]
    if ext not in ["jsonl", "tsv"]:
        raise TypeError("Unrecognized file type")
    print("begin load")
    with open(collection_file, "r") as f:
        if ext == "jsonl":
            for line in f:
                line = line.strip()
                obj = json.loads(line)
                pid = int(obj["id"][3:])
                #passage = obj["title"] + "[SEP]" + obj["text"]
                passage = obj["title"] + obj["text"]
                all_passages[pid] = passage
        else:
            #first_line = True
            for line in tqdm(f):
            #    if first_line:
            #        first_line = False
            #        continue
                line = line.strip()
                try:
                    line_arr = line.split("\t")
                    pid = line_arr[0]
                    if title == True:
                        passage = line_arr[2].rstrip() + ' ' + line_arr[1].rstrip()
                    else:
                        passage = line_arr[1].rstrip()
                    all_passages[pid] = passage
                except IndexError:
                    print("bad passage")
                except ValueError:
                    print("bad pid")
    return all_passages

def build_faiss_index(args):
    logger.info("Building index...")
    # ngpu = faiss.get_num_gpus()
    ngpu = args.n_gpu
    gpu_resources = []
    tempmem = -1

    for i in range(ngpu):
        res = faiss.StandardGpuResources()
        if tempmem >= 0:
            res.setTempMemory(tempmem)
        gpu_resources.append(res)

    cpu_index = faiss.IndexFlatIP(768)  
    index = None
    if args.use_gpu:
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.usePrecomputed = False
        # gpu_vector_resources, gpu_devices_vector
        vres = faiss.GpuResourcesVector()
        vdev = faiss.Int32Vector()
        for i in range(0, ngpu):
            vdev.push_back(i)
            vres.push_back(gpu_resources[i])
        gpu_index = faiss.index_cpu_to_gpu_multiple(vres,
                                                    vdev,
                                                    cpu_index, co)
        index = gpu_index
    else:
        index = cpu_index

    return index


def search_one_by_one_with_faiss(args, passge_embeddings_dir, index, query_embeddings, topN):
    merged_candidate_matrix = None
    if args.passage_block_num < 0:
        # automaticall get the number of passage blocks
        for filename in os.listdir(passge_embeddings_dir):
            try:
                args.passage_block_num = max(args.passage_block_num, int(filename.split(".")[1]) + 1)
            except:
                continue
        print("Automatically detect that the number of doc blocks is: {}".format(args.passage_block_num))
    for block_id in range(args.passage_block_num):
        logger.info("Loading passage block " + str(block_id))
        passage_embedding = None
        passage_embedding2id = None
        try:
            with open(oj(passge_embeddings_dir, "doc_emb_block.{}.pb".format(block_id)), 'rb') as handle:
                passage_embedding = pickle.load(handle)
            with open(oj(passge_embeddings_dir, "doc_embid_block.{}.pb".format(block_id)), 'rb') as handle:
                passage_embedding2id = pickle.load(handle)
                if isinstance(passage_embedding2id, list):
                    passage_embedding2id = np.array(passage_embedding2id)
        except:
            raise LoadError    
        
        logger.info('passage embedding shape: ' + str(passage_embedding.shape))
        logger.info("query embedding shape: " + str(query_embeddings.shape))

        passage_embeddings = np.array_split(passage_embedding, args.num_split_block)
        passage_embedding2ids = np.array_split(passage_embedding2id, args.num_split_block)
        for split_idx in range(len(passage_embeddings)):
            passage_embedding = passage_embeddings[split_idx]
            passage_embedding2id = passage_embedding2ids[split_idx]
            
            logger.info("Adding block {} split {} into index...".format(block_id, split_idx))
            index.add(passage_embedding)
            
            # ann search
            tb = time.time()
            D, I = index.search(query_embeddings, topN)
            elapse = time.time() - tb
            logger.info({
                'time cost': elapse,
                'query num': query_embeddings.shape[0],
                'time cost per query': elapse / query_embeddings.shape[0]
            })

            candidate_id_matrix = passage_embedding2id[I] # passage_idx -> passage_id
            D = D.tolist()
            candidate_id_matrix = candidate_id_matrix.tolist()
            candidate_matrix = []

            for score_list, passage_list in zip(D, candidate_id_matrix):
                candidate_matrix.append([])
                for score, passage in zip(score_list, passage_list):
                    candidate_matrix[-1].append((score, passage))
                assert len(candidate_matrix[-1]) == len(passage_list)
            assert len(candidate_matrix) == I.shape[0]

            index.reset()
            del passage_embedding
            del passage_embedding2id

            if merged_candidate_matrix == None:
                merged_candidate_matrix = candidate_matrix
                continue
            
            # merge
            merged_candidate_matrix_tmp = copy.deepcopy(merged_candidate_matrix)
            merged_candidate_matrix = []
            for merged_list, cur_list in zip(merged_candidate_matrix_tmp,
                                            candidate_matrix):
                p1, p2 = 0, 0
                merged_candidate_matrix.append([])
                while p1 < topN and p2 < topN:
                    if merged_list[p1][0] >= cur_list[p2][0]:
                        merged_candidate_matrix[-1].append(merged_list[p1])
                        p1 += 1
                    else:
                        merged_candidate_matrix[-1].append(cur_list[p2])
                        p2 += 1
                while p1 < topN:
                    merged_candidate_matrix[-1].append(merged_list[p1])
                    p1 += 1
                while p2 < topN:
                    merged_candidate_matrix[-1].append(cur_list[p2])
                    p2 += 1

    merged_D, merged_I = [], []

    for merged_list in merged_candidate_matrix:
        merged_D.append([])
        merged_I.append([])
        for candidate in merged_list:
            merged_D[-1].append(candidate[0])
            merged_I[-1].append(candidate[1])
    merged_D, merged_I = np.array(merged_D), np.array(merged_I)

    logger.info(merged_D.shape)
    logger.info(merged_I.shape)

    return merged_D, merged_I


def get_test_query_embedding(args):
    set_seed(args)

    config = RobertaConfig.from_pretrained(args.pretrained_encoder_path)
    tokenizer = RobertaTokenizer.from_pretrained(args.pretrained_encoder_path, do_lower_case=True)
    model = ANCE.from_pretrained(args.pretrained_encoder_path, config=config).to(args.device)

    # test dataset/dataloader
    args.batch_size = args.per_gpu_test_batch_size * max(1, args.n_gpu)
    logger.info("Buidling test dataset...")
    test_dataset = Search_Rel_Passage(args, tokenizer, args.test_file_path)
    test_loader = DataLoader(test_dataset, 
                                batch_size = args.batch_size, 
                                shuffle=False, 
                                collate_fn=test_dataset.get_collate_fn(args))
    

    logger.info("Generating query embeddings for testing...")
    model.zero_grad()

    embeddings = []
    embedding2id = []

    with torch.no_grad():
        for batch in tqdm(test_loader, disable=args.disable_tqdm):
            model.eval()
            bt_sample_ids = batch["bt_sample_ids"] # question id
            # test type
            input_ids = batch["bt_search_query"].to(args.device)
            input_masks = batch["bt_search_query_mask"].to(args.device)
            
            query_embs = model(input_ids, input_masks)
            query_embs = query_embs.detach().cpu().numpy()
            embeddings.append(query_embs)
            embedding2id.extend(bt_sample_ids)

    embeddings = np.concatenate(embeddings, axis = 0)
    torch.cuda.empty_cache()

    return embeddings, embedding2id


def output_test_res(query_embedding2id,
                    retrieved_scores_mat, # score_mat: score matrix, test_query_num * (top_k * block_num)
                    retrieved_pid_mat, # pid_mat: corresponding passage ids
                    #offset2pid,
                    args):
    

    qids_to_ranked_candidate_passages = {}
    topN = args.top_k

    for query_idx in range(len(retrieved_pid_mat)):
        seen_pid = set()
        query_id = query_embedding2id[query_idx]

        top_ann_pid = retrieved_pid_mat[query_idx].copy()
        top_ann_score = retrieved_scores_mat[query_idx].copy()
        selected_ann_idx = top_ann_pid[:topN]
        selected_ann_score = top_ann_score[:topN].tolist()
        rank = 0

        if query_id in qids_to_ranked_candidate_passages:
            pass
        else:
            tmp = [(0, 0)] * topN
            tmp_ori = [0] * topN
            qids_to_ranked_candidate_passages[query_id] = tmp

        for pred_pid, score in zip(selected_ann_idx, selected_ann_score):
            if not pred_pid in seen_pid:
                qids_to_ranked_candidate_passages[query_id][rank] = (pred_pid, score)
                rank += 1
                seen_pid.add(pred_pid)

        '''
        for idx, score in zip(selected_ann_idx, selected_ann_score):
            pred_pid = offset2pid[idx]

            if not pred_pid in seen_pid:
                qids_to_ranked_candidate_passages[query_id][rank] = (pred_pid, score)
                rank += 1
                seen_pid.add(pred_pid)
        '''

    # write to file
    logger.info('begin to write the output...')
    with open(args.test_file_path, "r") as fin:
        data = fin.readlines()

    all_passages = load_collection(args.passage_collection_path) # dict - pid: passage

    output_file = oj(args.output_dir_path, args.query_type + "_cast21_topic_with_relp.jsonl")
    idx = 0
    with open(output_file, "w") as fout:
        for qid, passages in qids_to_ranked_candidate_passages.items():
            ance_pos_docs = []
            ance_pos_docs_pids = []
            for i in range(topN):
                pid, score = passages[i]
                ance_pos_docs_pids.append(pid)
                try:
                    ance_pos_docs.append(all_passages[pid])
                except:
                    breakpoint()
            record = json.loads(data[idx])
            record['ance_pos_docs'] = ance_pos_docs
            record['ance_pos_docs_pids'] = ance_pos_docs_pids
            fout.write(json.dumps(record))
            fout.write('\n')
            idx += 1

    logger.info("output file write ok at {}".format(output_file))

def gen_metric_score_and_save(args, index, query_embeddings, query_embedding2id):
    # score_mat: score matrix, test_query_num * (top_n * block_num)
    # pid_mat: corresponding passage ids
    retrieved_scores_mat, retrieved_pid_mat = search_one_by_one_with_faiss(
                                                     args,
                                                     args.passage_embeddings_dir_path, 
                                                     index, 
                                                     query_embeddings, 
                                                     args.top_k) 

    #with open(args.passage_offset2pid_path, "rb") as f:
    #    offset2pid = pickle.load(f)
    
    output_test_res(query_embedding2id,
                    retrieved_scores_mat,
                    retrieved_pid_mat,
                    #offset2pid,
                    args)


def main():
    args = get_args()
    set_seed(args) 
    
    index = build_faiss_index(args)
    query_embeddings, query_embedding2id = get_test_query_embedding(args)
    gen_metric_score_and_save(args, index, query_embeddings, query_embedding2id)

    logger.info("Test finish!")
    

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_file_path", type=str, default="../datasets/cast19_topic.jsonl")
    parser.add_argument("--passage_collection_path", type=str, default="../../ConvDR-main/datasets/cast21/cast21_collection.tsv")
    parser.add_argument("--passage_embeddings_dir_path", type=str, default="../../ConvDR-main/datasets/cast21/embeds")
    #parser.add_argument("--passage_collection_path", type=str, default="../../ConvDR-main/datasets/qrecc/qrecc_collection.tsv")
    #parser.add_argument("--passage_embeddings_dir_path", type=str, default="../../ConvDR-main/datasets/qrecc/embeds")
    #parser.add_argument("--passage_offset2pid_path", type=str, default="../../ConvDR-main/datasets/qrecc/tokenized/offset2pid.pickle")
    parser.add_argument("--pretrained_encoder_path", type=str, default="../../ConvDR-main/checkpoints/ad-hoc-ance-msmarco")
    parser.add_argument("--output_dir_path", type=str, default="../datasets")

    parser.add_argument("--query_type", type=str, default="convq+conva+topic")
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per_gpu_test_batch_size", type=int, default=4)
    parser.add_argument("--passage_block_num", type=int, default=17) # 22 for qrecc and 26 for topiocqa 16 for cast
    parser.add_argument("--disable_tqdm", type=bool, default=False)
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--num_split_block", type=int, default=1, help="further split each block into several sub-blocks to reduce gpu memory use.")    


    parser.add_argument("--max_query_length", type=int, default=64)
    parser.add_argument("--max_response_length", type=int, default=256)
    parser.add_argument("--max_concat_length", type=int, default=510)


    args = parser.parse_args()
    if args.use_gpu:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    args.device = device

    logger.info("---------------------The arguments are:---------------------")
    logger.info(args)
    return args

if __name__ == '__main__':
    main()

