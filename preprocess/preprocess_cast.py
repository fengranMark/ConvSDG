import argparse
from tqdm import tqdm
import pickle
import os
import json

def load_collection(collection_file, title = False):
    all_passages = ["[INVALID DOC ID]"] * 5000_0000
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
                passage = obj["title"] + obj["text"]
                all_passages[pid] = passage
        else:
            for line in tqdm(f):
                line = line.strip()
                try:
                    line_arr = line.split("\t")
                    pid = int(line_arr[0])
                    if title == True:
                        passage = line_arr[2].rstrip().replace(' [SEP] ', ' ') + ' ' + line_arr[1].rstrip()
                    else:
                        passage = line_arr[1].rstrip()
                    all_passages[pid] = passage
                except IndexError:
                    print("bad passage")
                except ValueError:
                    print("bad pid")
    return all_passages

def merge_relevant_passage(query_file_1, query_file_2, query_file_3, qrel_file, collection_file, new_file):
    with open(qrel_file, 'r') as f:
        qrel_data = f.readlines()
    
    qid2pospid = {}
    for line in qrel_data:
        line = line.split()
        qid, pid, rel = line[0], line[2], int(line[3])
        if rel > 0:
            if qid not in qid2pospid:
                qid2pospid[qid] = [pid]
            else:
                qid2pospid[qid].append(pid)
    
    
    pid2text = {}
    with open(collection_file, "r") as f:
        for line in tqdm(f, disable=False):
            try:
                pid, text = line.strip().split("\t")
                pid2text[pid] = text
            except IndexError:
                print("bad passage")
            except ValueError:
                print("bad pid")
    
    #pid2text = load_collection(collection_file)
    
    with open(query_file_1, 'r') as f1, open(query_file_2, 'r') as f2, open(query_file_3, 'r') as f3:
        data_1, data_2, data_3 = f1.readlines(), f2.readlines(), f3.readlines()
    assert len(data_1) == len(data_2)
    assert len(data_2) == len(data_3)

    context_qs_1, context_qs_2 = [], []
    with open(new_file, 'w') as g:
        for i in range(len(data_1)):
            record_1, record_2, record_3 = json.loads(data_1[i]), json.loads(data_2[i]), json.loads(data_3[i])
            ori_qid = record_1["id"]
            conv_id, turn_id = int(record_1["id"].split('-')[0]), int(record_1["id"].split('-')[1])
            cur_query_1 = record_2["query"]
            cur_query_2 = record_3["query"]
            if ori_qid not in qid2pospid:
                continue
            if turn_id == 1:
                context_qs_1, context_qs_2 = [], []
            else:
                context_qs_1.append(cur_query_1)
                context_qs_2.append(cur_query_2)
            pos_docs_pids = qid2pospid[ori_qid]
            pos_docs = []
            for pos_pid in pos_docs_pids:
                pos_docs.append(pid2text[pos_pid])
            record_2["context_qs"] = context_qs_1
            record_3["context_qs"] = context_qs_2
            record_1["pos_docs_pids"] = pos_docs_pids
            record_1["pos_docs"] = pos_docs
            record_2["pos_docs_pids"] = pos_docs_pids
            record_2["pos_docs"] = pos_docs
            record_3["pos_docs_pids"] = pos_docs_pids
            record_3["pos_docs"] = pos_docs
            g.write(json.dumps(record_1) + "\n")
            g.write(json.dumps(record_2) + "\n")
            g.write(json.dumps(record_3) + "\n")
    print("finish")

def merge_bm25_neg_info(bm25_run_file, orig_file, new_file):
    qid2bm25_pid = {}
    with open(bm25_run_file, 'r') as f:
        data = f.readlines()

    for line in data:
        line = line.strip().split()
        qid, pid = line[0], int(line[2])
        if qid not in qid2bm25_pid:
            qid2bm25_pid[qid] = [pid]
        else:
            qid2bm25_pid[qid].append(pid)

    with open(orig_file, 'r') as f:
        ori_data = f.readlines()

    with open(new_file, 'w') as g:
        for line in ori_data:
            record = json.loads(line)
            qid = record["sample_id"]
            pos_docs_pids = record["pos_docs_pids"]
            bm25_hard_neg_docs_pids = []
            for pid in qid2bm25_pid[qid]:
                if pid not in pos_docs_pids:
                    bm25_hard_neg_docs_pids.append(pid)
            record["bm25_hard_neg_docs_pids"] = bm25_hard_neg_docs_pids
            g.write(json.dumps(record))
            g.write('\n')
        
            
def extract_doc_content_of_bm25_hard_negs_for_train_file(collection_path, 
                                                         train_inputfile, 
                                                         train_outputfile_with_doc,
                                                         neg_ratio=5):
    '''
    - collection_path = "collection.tsv"
    - train_inputfile = "train.json"
    - train_outputfile_with_doc = "train_with_neg.json"       
    '''
    with open(train_inputfile, 'r') as f:
        data = f.readlines()
    
    pid2passage = load_collection(collection_file)
    
    # Merge doc content to the train file
    with open(train_outputfile_with_doc, 'w') as fw:
        for line in data:
            line = json.loads(line)
            pos_docs_pids = line["pos_docs_pids"]
            neg_docs_text = []
            for pid in line["bm25_hard_neg_docs_pids"][:neg_ratio]:
                if pid in pid2passage and pid not in pos_docs_pids:
                    neg_docs_text.append(pid2passage[pid])
            
            line["bm25_hard_neg_docs"] = neg_docs_text
            
            fw.write(json.dumps(line))
            fw.write('\n')


if __name__ == "__main__":
    query_file_1 = "train_cast.jsonl"
    query_file_2 = "test_rewrite1.jsonl"
    query_file_3 = "test_rewrite2.jsonl"
    qrel_file = "train_qrels.tsv"
    collection_file = "../../ConvDR-main/datasets/cast20/collection.tsv"
    new_file = "augmented_train_cast.jsonl"
    merge_relevant_passage(query_file_1, query_file_2, query_file_3, qrel_file, collection_file, new_file)
