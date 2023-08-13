import sys
sys.path.append('..')
sys.path.append('.')

import json,string, re
import pytrec_eval
from collections import Counter

import argparse
from collections import defaultdict
import time
import numpy as np
import os

def print_trec_res(run_file, qrel_file, rel_threshold):
    with open(run_file, 'r' )as f:
        run_data = f.readlines()
    with open(qrel_file, 'r') as f:
        qrel_data = f.readlines()
    
    qrels = {}
    qrels_ndcg = {}
    runs = {}
    
    for line in qrel_data:
        if args.dataset == "topiocqa":
            line = line.split(" ")
        else:
            line = line.split()
        query = line[0]
        passage = line[2]
        rel = int(line[3])
        if query not in qrels:
            qrels[query] = {}
        if query not in qrels_ndcg:
            qrels_ndcg[query] = {}

        # for NDCG
        qrels_ndcg[query][passage] = rel
        # for MAP, MRR, Recall
        if rel >= rel_threshold:
            rel = 1
        else:
            rel = 0
        qrels[query][passage] = rel
    
    for line in run_data:
        line = line.split(" ")
        query = line[0]
        passage = line[2]
        rel = int(line[4])
        if query not in runs:
            runs[query] = {}
        runs[query][passage] = rel

    # pytrec_eval eval
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"map", "recip_rank", "recall.5", "recall.10", "recall.20", "recall.100"})
    res = evaluator.evaluate(runs)
    map_list = [v['map'] for v in res.values()]
    mrr_list = [v['recip_rank'] for v in res.values()]
    recall_100_list = [v['recall_100'] for v in res.values()]
    recall_20_list = [v['recall_20'] for v in res.values()]
    recall_10_list = [v['recall_10'] for v in res.values()]
    recall_5_list = [v['recall_5'] for v in res.values()]

    evaluator = pytrec_eval.RelevanceEvaluator(qrels_ndcg, {"ndcg_cut.3"})
    res = evaluator.evaluate(runs)
    ndcg_3_list = [v['ndcg_cut_3'] for v in res.values()]

    res = {
            "MAP": np.average(map_list),
            "MRR": np.average(mrr_list),
            "NDCG@3": np.average(ndcg_3_list), 
            "Recall@5": np.average(recall_5_list),
            "Recall@10": np.average(recall_10_list),
            "Recall@20": np.average(recall_20_list),
            "Recall@100": np.average(recall_100_list),
        }

    
    print("---------------------Evaluation results:---------------------")    
    print(res)
    return res

def agg_res_with_maxp(run_trec_file):
    res_file = os.path.join(run_trec_file)
    with open(run_trec_file, 'r' ) as f:
        run_data = f.readlines()
    
    agg_run = {}
    for line in run_data:
        line = line.strip().split(" ")
        sample_id = line[0]
        if sample_id not in agg_run:
            agg_run[sample_id] = {}
        doc_id = "_".join(line[2].split('_')[:2])
        try:
            score = float(line[5])
        except:
            breakpoint()
        if doc_id not in agg_run[sample_id]:
            agg_run[sample_id][doc_id] = 0
        agg_run[sample_id][doc_id] = max(agg_run[sample_id][doc_id], score)
    
    agg_run = {k: sorted(v.items(), key=lambda item: item[1], reverse=True) for k, v in agg_run.items()}
    with open(os.path.join(run_trec_file + ".agg"), "w") as f:
        for sample_id in agg_run:
            doc_scores = agg_run[sample_id]
            rank = 1
            for doc_id, real_score in doc_scores:
                rank_score = 2000 - rank
                f.write("{} Q0 {} {} {} {}\n".format(sample_id, doc_id, rank, rank_score, real_score, "ance"))
                rank += 1


# (qid, Q0, docid, rank, rel, score, bm25)
def read_rank_list(args, file):
    qid_docid_list = defaultdict(list)
    qid_score_list = defaultdict(list)
    with open(file, 'r') as f:
        for line in f:
            qid, _, docid, rank, _, score, _ = line.strip().split(' ')
            '''
            if args.dataset == "cast20":
                if len(qid.split('_')) != 3:
                    qid_list = qid.split('-')
                    qid = "CAsT20-test" + '_' + qid_list[0] + '_' + qid_list[1]
            elif args.dataset == "cast19":
                qid = qid.replace('-', '_')
            '''
            qid_docid_list[qid].append(docid)
            qid_score_list[qid].append(float(score))
    return qid_docid_list, qid_score_list # qid: [docid] [score]

def fuse(args, docid_list0, docid_list1, doc_score_list0, doc_score_list1, alpha):
    if args.fusion_method == "CQE_hybrid":
        score = defaultdict(float)
        score0 = defaultdict(float)
        for i, docid in enumerate(docid_list0):
            score0[docid]+=doc_score_list0[i]
        min_val0 = min(doc_score_list0)
        min_val1 = min(doc_score_list1)
        for i, docid in enumerate(docid_list1):
            if score0[docid]==0:
                score[docid]+=min_val0 + doc_score_list1[i]*alpha
            else:
                score[docid]+=doc_score_list1[i]*alpha
        for i, docid in enumerate(docid_list0):
            if score[docid]==0:
                score[docid]+=min_val1*alpha
            score[docid]+=doc_score_list0[i]
        score= {k: v for k, v in sorted(score.items(), key=lambda item: item[1], reverse=True)}
        return score
    elif args.fusion_method == "combine_max":
        score = {}
        for i, docid in enumerate(docid_list0):
            score[docid] = doc_score_list0[i]
        for i, docid in enumerate(docid_list1):
            if docid not in score:
                score[docid] = doc_score_list1[i]
            else:
                score[docid] = max(score[docid], doc_score_list1[i])
        score= {k: v for k, v in sorted(score.items(), key=lambda item: item[1], reverse=True)}
        return score
    elif args.fusion_method == "combine_sum":
        score = {}
        for i, docid in enumerate(docid_list0):
            score[docid] = doc_score_list0[i]
        for i, docid in enumerate(docid_list1):
            if docid not in score:
                score[docid] = doc_score_list1[i]
            else:
                score[docid] = score[docid] + doc_score_list1[i]
        score= {k: v for k, v in sorted(score.items(), key=lambda item: item[1], reverse=True)}
        return score

    elif args.fusion_method == "RRF": # bug
        score0, score1, score = {}, {}, {}
        for i, docid in enumerate(docid_list0):
            score0[docid] = doc_score_list0[i]
        for i, docid in enumerate(docid_list1):
            score1[docid] = doc_score_list1[i] 
        
        for key, value in score0.items():
            if key in score1:
                score[key] = 1 / (score0[key] + score1[key] + 60)
            else:
                score[key] = 1 / (score0[key] + 60) 
        for key, value in score1.items():
            if key not in score0:
                score[key] = 1 / (score1[key] + 60) 
        score= {k: v for k, v in sorted(score.items(), key=lambda item: item[1], reverse=True)}
        return score


def hybrid_fusion(args, rank_file0, rank_file1, fusion_output, trec_gold):
    print('Read ranked list0...')
    qid_docid_list0, qid_score_list0 = read_rank_list(args, rank_file0)
    print('Read ranked list1...')
    qid_docid_list1, qid_score_list1 = read_rank_list(args, rank_file1)

    qids = qid_docid_list0.keys()
    #print(qids)
    fout = open(fusion_output, 'w')
    for j, qid in enumerate(qids):
        #  pid : score   for each qid
        rank_doc_score = fuse(args, qid_docid_list0[qid], qid_docid_list1[qid], qid_score_list0[qid], qid_score_list1[qid], args.alpha)
        for rank, doc in enumerate(rank_doc_score):
            if rank==args.topk:
                break
            score = rank_doc_score[doc]
            fout.write('{} Q0 {} {} {} {} {}\n'.format(qid, doc, rank + 1, str(-rank - 1 + 200), score, "fusion"))
    print('fusion finish')
        #pbar.update(j + 1)
    #time_per_query = (time.time() - start_time)/len(qids)
    #print('Fusing {} queries ({:0.3f} s/query)'.format(len(qids), time_per_query))
    if args.dataset == "cast21":
        agg_res_with_maxp(fusion_output)
        trec_res = print_trec_res(fusion_output + '.agg', trec_gold, args.rel_threshold)
    else:
        trec_res = print_trec_res(fusion_output, trec_gold, args.rel_threshold)
    fout.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rank list fusion')
    parser.add_argument('--topk', default=100, type=int, help='number of hits to retrieve')
    parser.add_argument("--alpha", type=float, default=1) # cast19 0.1 cast20 3.5 topiocqa 1.2
    parser.add_argument("--rel_threshold", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="cast21")
    parser.add_argument("--fusion_method", type=str, default="CQE_hybrid")
    args = parser.parse_args()

    if args.dataset == "topiocqa":
        rank_file1 = "../output/topiocqa/file/ancepos_convqp_prompt_res.trec"
        rank_file0 = "../output/topiocqa/file/bm25pos_convqp_prompt_res.trec"
        hybrid_fusion_output = "../output/topiocqa/file/fusion_topiocqa_res.trec"
        trec_gold = "../../ConvDR-main/datasets/topiocqa/dev_gold.trec"
        hybrid_fusion(args, rank_file0, rank_file1, hybrid_fusion_output, trec_gold)
    elif args.dataset == "cast19":
        rank_file0 = "../output/cast/file/ancepos_convq_cast19topic_res.trec"
        rank_file1 = "../output/cast/file/bm25pos_convq_cast19topic_res.trec"
        hybrid_fusion_output = "../output/cast/file/fusion_cast19_res.trec"
        trec_gold = "../../ConvDR-main/datasets/cast19/qrels.tsv"
        hybrid_fusion(args, rank_file0, rank_file1, hybrid_fusion_output, trec_gold)
    elif args.dataset == "cast20":
        args.rel_threshold = 2
        rank_file1 = "../output/cast/file/ancepos_convq_cast20topic_res.trec"
        rank_file0 = "../output/cast/file/bm25pos_convq_cast20topic_res.trec"
        hybrid_fusion_output = "../output/cast/file/fusion_cast20_res.trec"
        trec_gold = "../../ConvDR-main/datasets/cast20/qrels.tsv"
        hybrid_fusion(args, rank_file0, rank_file1, hybrid_fusion_output, trec_gold)
    elif args.dataset == "cast21":
        args.rel_threshold = 2
        rank_file1 = "../output/cast/file/ancepos_convq_cast21topic_res.trec"
        rank_file0 = "../output/cast/file/bm25pos_convq_cast21topic_res.trec"
        hybrid_fusion_output = "../output/cast/file/fusion_cast21_res.trec"
        trec_gold = "../../ConvDR-main/datasets/cast21/trec-cast-qrels-docs.2021.qrel"
        hybrid_fusion(args, rank_file0, rank_file1, hybrid_fusion_output, trec_gold)
