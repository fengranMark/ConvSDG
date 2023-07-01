import torch
from torch.utils.data import Dataset
import json
from tqdm import tqdm, trange
import csv
import random

def padding_seq_to_same_length(input_ids, max_pad_length, pad_token = 0):
    padding_length = max_pad_length - len(input_ids)
    padding_ids = [pad_token] * padding_length
    attention_mask = []

    if padding_length <= 0:
        attention_mask = [1] * max_pad_length
        input_ids = input_ids[:max_pad_length]
    else:
        attention_mask = [1] * len(input_ids) + [0] * padding_length
        input_ids = input_ids + padding_ids
            
    assert len(input_ids) == max_pad_length
    assert len(attention_mask) == max_pad_length
  
    return input_ids, attention_mask

class Retrieval_topiocqa(Dataset):
    def __init__(self, args, tokenizer, filename, collection=None):
        self.examples = []
        
        with open(filename, encoding="utf-8") as f:
            data = f.readlines()
        if args.is_extend_train:
            with open(args.extend_train_file_path, encoding="utf-8") as f_extend:
                extend_data = f_extend.readlines()
                extend_data = random.sample(extend_data, int(len(extend_data) * args.sample_ratio))
                data.extend(extend_data)
        n = len(data)

        for i in tqdm(trange(n)):
            record = json.loads(data[i])
            # <CUR> oracle_query <CTX> cq_{k-1} <SEP> ... cq_{1} (<BOS>)
            sample_id = str(record["conversation_id"]) + '-' + str(record["turn"])
            flat_q_concat = []
            flat_qa_concat = []
            flat_qp_concat = []
            cur_query = record["query"]
            cur_answer = record["answer"]
            query_context = record["context"]
            answer_context = record["answer_context"]
            cur_topic = record["topic"]
            if record["turn"] == 1:
                last_response = ""
            else:
                if args.pos_type == "bm25":
                    last_response = json.loads(data[i - 1])['bm25_pos_docs'][0]
                else:
                    last_response = json.loads(data[i - 1])['ance_pos_docs']
            if args.pos_type == "bm25":       
                pos_docs_text = record['bm25_pos_docs'][0]
                pos_docs_pids = record['bm25_pos_docs_pids'][0]
            else:
                pos_docs_text = record['ance_pos_docs']
                pos_docs_pids = record['ance_pos_docs_pids']
            #neg_docs_text = record['neg_docs'][0]
            #neg_docs_pids = record['neg_docs_pids'][0]
            #prepos_neg_docs_pids = record['prepos_neg_docs_pids']
            #rel_label = record['rel_label']
            #cur_response_text = record["answer"]
            #oracle_utt_text = record["rewrite"]

            cur_utt = tokenizer.encode(cur_query, add_special_tokens = True, max_length = args.max_query_length)

            flat_q_concat.extend(cur_utt)
            flat_qa_concat.extend(cur_utt)
            flat_qp_concat.extend(cur_utt)
            
            if len(last_response) > 0:
                lp = []
                lp.append(tokenizer.cls_token_id)
                lp.extend(tokenizer.convert_tokens_to_ids(["<response>"]))
                lp.extend(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(last_response)))
                lp = lp[:args.max_doc_length]
                lp.append(tokenizer.sep_token_id)
                flat_qp_concat.extend(lp)
            
            
            for j in range(len(query_context) - 1, -1, -1):
                utt_a = tokenizer.encode(answer_context[j], add_special_tokens=True, max_length=args.max_response_length, truncation=True) # not remove [CLS]
                utt_q = tokenizer.encode(query_context[j], add_special_tokens=True, max_length=args.max_query_length, truncation=True) # not remove [CLS]
                if len(flat_qa_concat) + len(utt_a) > args.max_concat_length:
                    flat_qa_concat += utt_a[:args.max_concat_length - len(flat_qa_concat) - 1] + [utt_a[-1]]    # must ended with [SEP]
                else:
                    flat_qa_concat.extend(utt_a)
                flat_qa_concat.extend(utt_q)  
                if len(flat_qp_concat) + len(utt_q) > args.max_concat_length:
                    flat_qp_concat += utt_q[:args.max_concat_length - len(flat_qp_concat) - 1] + [utt_q[-1]]    # must ended with [SEP]
                else:
                    flat_qp_concat.extend(utt_q)

            cur_utt, cur_utt_mask = padding_seq_to_same_length(cur_utt, max_pad_length = args.max_query_length)
            flat_q_concat, flat_q_concat_mask = padding_seq_to_same_length(flat_q_concat, max_pad_length = args.max_concat_length)
            flat_qa_concat, flat_qa_concat_mask = padding_seq_to_same_length(flat_qa_concat, max_pad_length = args.max_concat_length)
            flat_qp_concat, flat_qp_concat_mask = padding_seq_to_same_length(flat_qp_concat, max_pad_length = args.max_concat_length)

            # doc 
            pos_docs = []
            neg_docs = []
            pos_docs_mask = []
            neg_docs_mask = []
            pos_docs_id = []
            neg_docs_id = []
            if args.is_train:
                pos_docs.extend(tokenizer.encode(pos_docs_text, add_special_tokens=True, max_length=args.max_doc_length, truncation=True))
                pos_docs_id.append(pos_docs_pids)
                #if args.is_hard_neg:
                #    neg_docs.extend(tokenizer.encode(neg_docs_text, add_special_tokens=True, max_length=args.max_doc_length, truncation=True))
                #    neg_docs_id.append(neg_docs_pids)
                pos_docs, pos_docs_mask = padding_seq_to_same_length(pos_docs, max_pad_length = args.max_concat_length)
                #neg_docs, neg_docs_mask = padding_seq_to_same_length(neg_docs, max_pad_length = args.max_concat_length)

            self.examples.append([sample_id, 
                            cur_utt,
                            cur_utt_mask,
                            flat_q_concat,
                            flat_q_concat_mask,
                            flat_qa_concat,
                            flat_qa_concat_mask,
                            flat_qp_concat,
                            flat_qp_concat_mask,
                            pos_docs,
                            pos_docs_mask,
                            neg_docs,
                            neg_docs_mask])

        if args.is_real_train:
            with open(args.extend_train_file_path, encoding="utf-8") as f_extend:
                data = f_extend.readlines()
                data = random.sample(data, int(len(data) * args.sample_ratio))
            n = len(data)

            for i in tqdm(trange(n)):
                record = json.loads(data[i])
                # <CUR> oracle_query <CTX> cq_{k-1} <SEP> ... cq_{1} (<BOS>)
                sample_id = record["id"]
                flat_q_concat = []
                flat_qa_concat = []
                flat_qp_concat = []
                cur_query = record["query"]
                query_context = record["history_query"]
                answer_context = record["history_answer"]
                if record["turn_id"] == 1:
                    last_response = ""
                else:
                    last_response = json.loads(data[i - 1])['pos_docs'][0]
                pos_docs_text = record['pos_docs'][0]
                #pos_docs_pids = record['pos_docs_pids'][0]

                cur_utt = tokenizer.encode(cur_query, add_special_tokens = True, max_length = args.max_query_length)

                flat_q_concat.extend(cur_utt)
                flat_qa_concat.extend(cur_utt)
                flat_qp_concat.extend(cur_utt)
                
                if len(last_response) > 0:
                    lp = []
                    lp.append(tokenizer.cls_token_id)
                    lp.extend(tokenizer.convert_tokens_to_ids(["<response>"]))
                    lp.extend(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(last_response)))
                    lp = lp[:args.max_doc_length]
                    lp.append(tokenizer.sep_token_id)
                    flat_qp_concat.extend(lp)
                
                
                for j in range(len(query_context) - 1, -1, -1):
                    utt_a = tokenizer.encode(answer_context[j], add_special_tokens=True, max_length=args.max_response_length, truncation=True) # not remove [CLS]
                    utt_q = tokenizer.encode(query_context[j], add_special_tokens=True, max_length=args.max_query_length, truncation=True) # not remove [CLS]
                    if len(flat_qa_concat) + len(utt_a) > args.max_concat_length:
                        flat_qa_concat += utt_a[:args.max_concat_length - len(flat_qa_concat) - 1] + [utt_a[-1]]    # must ended with [SEP]
                    else:
                        flat_qa_concat.extend(utt_a)
                    flat_qa_concat.extend(utt_q)  

                cur_utt, cur_utt_mask = padding_seq_to_same_length(cur_utt, max_pad_length = args.max_query_length)
                flat_q_concat, flat_q_concat_mask = padding_seq_to_same_length(flat_q_concat, max_pad_length = args.max_concat_length)
                flat_qa_concat, flat_qa_concat_mask = padding_seq_to_same_length(flat_qa_concat, max_pad_length = args.max_concat_length)
                flat_qp_concat, flat_qp_concat_mask = padding_seq_to_same_length(flat_qp_concat, max_pad_length = args.max_concat_length)

                # doc 
                pos_docs = []
                neg_docs = []
                pos_docs_mask = []
                neg_docs_mask = []
                pos_docs_id = []
                neg_docs_id = []
                if args.is_train:
                    pos_docs.extend(tokenizer.encode(pos_docs_text, add_special_tokens=True, max_length=args.max_doc_length, truncation=True))
                    #pos_docs_id.append(pos_docs_pids)
                    #if args.is_hard_neg:
                    #    neg_docs.extend(tokenizer.encode(neg_docs_text, add_special_tokens=True, max_length=args.max_doc_length, truncation=True))
                    #    neg_docs_id.append(neg_docs_pids)
                    pos_docs, pos_docs_mask = padding_seq_to_same_length(pos_docs, max_pad_length = args.max_concat_length)
                    #neg_docs, neg_docs_mask = padding_seq_to_same_length(neg_docs, max_pad_length = args.max_concat_length)

                self.examples.append([sample_id, 
                                cur_utt,
                                cur_utt_mask,
                                flat_q_concat,
                                flat_q_concat_mask,
                                flat_qa_concat,
                                flat_qa_concat_mask,
                                flat_qp_concat,
                                flat_qp_concat_mask,
                                pos_docs,
                                pos_docs_mask,
                                neg_docs,
                                neg_docs_mask])


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def get_collate_fn(args):
        
        def collate_fn(batch: list):
            collated_dict = {"bt_sample_ids": [],
                             "bt_raw_query": [],
                             "bt_raw_query_mask": [],
                             "bt_conv_q": [],
                             "bt_conv_q_mask": [],
                             "bt_conv_qa": [],
                             "bt_conv_qa_mask": [],
                             "bt_conv_qp": [],
                             "bt_conv_qp_mask": [],
                             "bt_pos_docs":[],
                             "bt_pos_docs_mask":[],
                             "bt_neg_docs":[],
                             "bt_neg_docs_mask":[],
                            }
            for example in batch:
                collated_dict["bt_sample_ids"].append(example[0])
                collated_dict["bt_raw_query"].append(example[1])
                collated_dict["bt_raw_query_mask"].append(example[2])
                collated_dict["bt_conv_q"].append(example[3])
                collated_dict["bt_conv_q_mask"].append(example[4])
                collated_dict["bt_conv_qa"].append(example[5])
                collated_dict["bt_conv_qa_mask"].append(example[6])
                collated_dict["bt_conv_qp"].append(example[7])
                collated_dict["bt_conv_qp_mask"].append(example[8])
                collated_dict["bt_pos_docs"].append(example[9])
                collated_dict["bt_pos_docs_mask"].append(example[10])
                collated_dict["bt_neg_docs"].append(example[11])
                collated_dict["bt_neg_docs_mask"].append(example[12])

            not_need_to_tensor_keys = set(["bt_sample_ids", "bt_cur_utt_text", "bt_oracle_utt_text"])

            for key in collated_dict:
                if key not in not_need_to_tensor_keys:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)
                    
            return collated_dict

        return collate_fn

class Retrieval_cast_topic(Dataset):
    def __init__(self, args, tokenizer, filename, collection=None):
        self.examples = []
        
        with open(filename, encoding="utf-8") as f:
            data = f.readlines()

        if args.is_extend_train:
            with open(args.extend_train_file_path) as f:
                extend_data = f.readlines()
            data.extend(extend_data)
        n = len(data)

        for i in tqdm(trange(n)):
            record = json.loads(data[i])
            # <CUR> oracle_query <CTX> cq_{k-1} <SEP> ... cq_{1} (<BOS>)
            sample_id = str(record["conversation_id"]) + '-' + str(record["turn"])
            flat_q_concat = []
            cur_query = record["query"]
            cur_answer = record["answer"]
            query_context = record["context"]
            answer_context = record["answer_context"]
            cur_topic = record["topic"]
            if args.pos_type == "bm25":       
                pos_docs_text = record['bm25_pos_docs']
                pos_docs_pids = record['bm25_pos_docs_pids']
            elif args.pos_type == "ance":
                pos_docs_text = record['ance_pos_docs']
                pos_docs_pids = record['ance_pos_docs_pids']
            cur_utt = tokenizer.encode(cur_query, add_special_tokens = True, max_length = args.max_query_length)
            flat_q_concat.extend(cur_utt)   
            for j in range(len(query_context) - 1, -1, -1):
                utt_q = tokenizer.encode(query_context[j], add_special_tokens=True, max_length=args.max_query_length, truncation=True) # not remove [CLS]
                if len(flat_q_concat) + len(utt_q) > args.max_concat_length:
                    flat_q_concat += utt_q[:args.max_concat_length - len(flat_q_concat) - 1] + [utt_q[-1]]    # must ended with [SEP]
                else:
                    flat_q_concat.extend(utt_q)

            flat_q_concat, flat_q_concat_mask = padding_seq_to_same_length(flat_q_concat, max_pad_length = args.max_concat_length)
            # doc 
            pos_docs, neg_docs, pos_docs_mask, neg_docs_mask = [], [], [], []
            if args.is_train:
                for pos_doc in pos_docs_text:
                    pos_docs, neg_docs, pos_docs_mask, neg_docs_mask = [], [], [], []
                    pos_docs.extend(tokenizer.encode(pos_doc, add_special_tokens=True, max_length=args.max_doc_length, truncation=True))
                    pos_docs, pos_docs_mask = padding_seq_to_same_length(pos_docs, max_pad_length = args.max_concat_length)
                    self.examples.append([sample_id, flat_q_concat, flat_q_concat_mask, pos_docs, pos_docs_mask, neg_docs, neg_docs_mask])
                #neg_docs, neg_docs_mask = padding_seq_to_same_length(neg_docs, max_pad_length = args.max_concat_length)
            else:
                self.examples.append([sample_id, flat_q_concat, flat_q_concat_mask, pos_docs, pos_docs_mask, neg_docs, neg_docs_mask])
            breakpoint()
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def get_collate_fn(args):
        
        def collate_fn(batch: list):
            collated_dict = {"bt_sample_ids": [],
                             "bt_conv_q": [],
                             "bt_conv_q_mask": [],
                             "bt_pos_docs":[],
                             "bt_pos_docs_mask":[],
                             "bt_neg_docs":[],
                             "bt_neg_docs_mask":[],
                            }
            for example in batch:
                collated_dict["bt_sample_ids"].append(example[0])
                collated_dict["bt_conv_q"].append(example[1])
                collated_dict["bt_conv_q_mask"].append(example[2])
                collated_dict["bt_pos_docs"].append(example[3])
                collated_dict["bt_pos_docs_mask"].append(example[4])
                collated_dict["bt_neg_docs"].append(example[3])
                collated_dict["bt_neg_docs_mask"].append(example[4])

            not_need_to_tensor_keys = set(["bt_sample_ids", "bt_cur_utt_text", "bt_oracle_utt_text"])

            for key in collated_dict:
                if key not in not_need_to_tensor_keys:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)
                    
            return collated_dict

        return collate_fn

class Retrieval_cast_augment(Dataset):
    def __init__(self, args, tokenizer, filename, collection=None):
        self.examples = []
        
        with open(filename, encoding="utf-8") as f:
            data = f.readlines()

        if args.is_extend_train:
            with open(args.extend_train_file_path) as f:
                extend_data = f.readlines()
            data.extend(extend_data)
            
        n = len(data)

        for i in tqdm(trange(n)):
            record = json.loads(data[i])
            # <CUR> oracle_query <CTX> cq_{k-1} <SEP> ... cq_{1} (<BOS>)
            if "id" in record:
                sample_id = record["id"]
            else:
                sample_id = str(record["conversation_id"]) + '-' + str(record["turn"])
            flat_q_concat = []
            cur_query = record["query"]
            if "context_qs" in record:
                context = record["context_qs"]
            else:
                context = record["context"]
            if 'pos_docs' in record:
                pos_docs_text = record['pos_docs']
                pos_docs_pids = record['pos_docs_pids']
            elif 'ance_pos_docs' in record:
                pos_docs_text = record['ance_pos_docs']
                pos_docs_pids = record['ance_pos_docs_pids']

            cur_utt = tokenizer.encode(cur_query, add_special_tokens = True, max_length = args.max_query_length)
            flat_q_concat.extend(cur_utt) 
            for j in range(len(context) - 1, -1, -1):
                utt_q = tokenizer.encode(context[j], add_special_tokens=True, max_length=args.max_query_length, truncation=True) # not remove [CLS]
                if len(flat_q_concat) + len(utt_q) > args.max_concat_length:
                    flat_q_concat += utt_q[:args.max_concat_length - len(flat_q_concat) - 1] + [utt_q[-1]]    # must ended with [SEP]
                else:
                    flat_q_concat.extend(utt_q)

            flat_q_concat, flat_q_concat_mask = padding_seq_to_same_length(flat_q_concat, max_pad_length = args.max_concat_length)

            # doc 
            pos_docs, neg_docs, pos_docs_mask, neg_docs_mask = [], [], [], []
            if args.is_train:
                for pos_doc in pos_docs_text:
                    pos_docs, neg_docs, pos_docs_mask, neg_docs_mask = [], [], [], []
                    pos_docs.extend(tokenizer.encode(pos_doc, add_special_tokens=True, max_length=args.max_doc_length, truncation=True))
                    pos_docs, pos_docs_mask = padding_seq_to_same_length(pos_docs, max_pad_length = args.max_concat_length)
                    self.examples.append([sample_id, flat_q_concat, flat_q_concat_mask, pos_docs, pos_docs_mask, neg_docs, neg_docs_mask])
                #neg_docs, neg_docs_mask = padding_seq_to_same_length(neg_docs, max_pad_length = args.max_concat_length)
            else:
                self.examples.append([sample_id, flat_q_concat, flat_q_concat_mask, pos_docs, pos_docs_mask, neg_docs, neg_docs_mask])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def get_collate_fn(args):
        
        def collate_fn(batch: list):
            collated_dict = {"bt_sample_ids": [],
                             "bt_conv_q": [],
                             "bt_conv_q_mask": [],
                             "bt_pos_docs":[],
                             "bt_pos_docs_mask":[],
                             "bt_neg_docs":[],
                             "bt_neg_docs_mask":[],
                            }
            for example in batch:
                collated_dict["bt_sample_ids"].append(example[0])
                collated_dict["bt_conv_q"].append(example[1])
                collated_dict["bt_conv_q_mask"].append(example[2])
                collated_dict["bt_pos_docs"].append(example[3])
                collated_dict["bt_pos_docs_mask"].append(example[4])
                collated_dict["bt_neg_docs"].append(example[3])
                collated_dict["bt_neg_docs_mask"].append(example[4])

            not_need_to_tensor_keys = set(["bt_sample_ids", "bt_cur_utt_text", "bt_oracle_utt_text"])

            for key in collated_dict:
                if key not in not_need_to_tensor_keys:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)
                    
            return collated_dict

        return collate_fn

class Test_Retrieval_topiocqa(Dataset):
    def __init__(self, args, tokenizer, filename, collection=None):
        self.examples = []
        
        with open(filename, encoding="utf-8") as f:
            data = f.readlines()

        n = len(data)

        for i in tqdm(trange(n)):
            record = json.loads(data[i])
            # <CUR> oracle_query <CTX> cq_{k-1} <SEP> ... cq_{1} (<BOS>)
            sample_id = record['sample_id']
            flat_q_concat = []
            flat_qa_concat = []
            flat_qp_concat = []
            ctx_utts_text = record['cur_utt_text'].strip().split(" [SEP] ") # [q1, a1, q2, a2, ...]
            cur_utt_text = ctx_utts_text[-1] 
            ctx_utts_text = ctx_utts_text[:-1]
            last_response = record['last_response']

            cur_utt = tokenizer.encode(cur_utt_text, add_special_tokens = True, max_length = args.max_query_length)
            flat_q_concat.extend(cur_utt)
            flat_qa_concat.extend(cur_utt)
            flat_qp_concat.extend(cur_utt)

            if len(last_response) > 0:
                lp = []
                lp.append(tokenizer.cls_token_id)
                lp.extend(tokenizer.convert_tokens_to_ids(["<response>"]))
                lp.extend(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(last_response)))
                lp = lp[:args.max_doc_length]
                lp.append(tokenizer.sep_token_id)
                flat_qp_concat.extend(lp)
                
            for j in range(len(ctx_utts_text) - 1, -1, -1):
                if j % 2 == 1: # answer
                    max_length = args.max_response_length
                elif j % 2 == 0: # query
                    max_length = args.max_query_length
                utt = tokenizer.encode(ctx_utts_text[j], add_special_tokens=True, max_length=max_length, truncation=True) # not remove [CLS]
                if len(flat_qa_concat) + len(utt) > args.max_concat_length:
                    flat_qa_concat += utt[:args.max_concat_length - len(flat_qa_concat) - 1] + [utt[-1]]    # must ended with [SEP]
                    if j % 2 == 0:
                        flat_q_concat += utt[:args.max_concat_length - len(flat_q_concat) - 1] + [utt[-1]]    # must ended with [SEP]
                    break
                else:
                    flat_qa_concat.extend(utt) 
                #if j % 2 == 0:
                #    if len(flat_qp_concat) + len(utt) > args.max_concat_length:
                #        flat_qp_concat += utt[:args.max_concat_length - len(flat_qp_concat) - 1] + [utt[-1]]    # must ended with [SEP]
                #    else:
                #        flat_qp_concat.extend(utt)


            cur_utt, cur_utt_mask = padding_seq_to_same_length(cur_utt, max_pad_length = args.max_query_length)
            flat_q_concat, flat_q_concat_mask = padding_seq_to_same_length(flat_q_concat, max_pad_length = args.max_concat_length)
            flat_qa_concat, flat_qa_concat_mask = padding_seq_to_same_length(flat_qa_concat, max_pad_length = args.max_concat_length)
            flat_qp_concat, flat_qp_concat_mask = padding_seq_to_same_length(flat_qp_concat, max_pad_length = args.max_concat_length)

            self.examples.append([sample_id, 
                            cur_utt,
                            cur_utt_mask,
                            flat_q_concat,
                            flat_q_concat_mask,
                            flat_qa_concat,
                            flat_qa_concat_mask,
                            flat_qp_concat,
                            flat_qp_concat_mask])
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def get_collate_fn(args):
        
        def collate_fn(batch: list):
            collated_dict = {"bt_sample_ids": [],
                             "bt_raw_query": [],
                             "bt_raw_query_mask": [],
                             "bt_conv_q": [],
                             "bt_conv_q_mask": [],
                             "bt_conv_qa": [],
                             "bt_conv_qa_mask": [],
                             "bt_conv_qp": [],
                             "bt_conv_qp_mask": [],
                            }
            for example in batch:
                collated_dict["bt_sample_ids"].append(example[0])
                collated_dict["bt_raw_query"].append(example[1])
                collated_dict["bt_raw_query_mask"].append(example[2])
                collated_dict["bt_conv_q"].append(example[3])
                collated_dict["bt_conv_q_mask"].append(example[4])
                collated_dict["bt_conv_qa"].append(example[5])
                collated_dict["bt_conv_qa_mask"].append(example[6])
                collated_dict["bt_conv_qp"].append(example[7])
                collated_dict["bt_conv_qp_mask"].append(example[8])

            not_need_to_tensor_keys = set(["bt_sample_ids", "bt_cur_utt_text", "bt_oracle_utt_text"])

            for key in collated_dict:
                if key not in not_need_to_tensor_keys:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)
                    
            return collated_dict

        return collate_fn

class Test_Retrieval_qrecc(Dataset):
    def __init__(self, args, tokenizer, filename, collection=None):
        self.examples = []
        
        with open(filename, encoding="utf-8") as f:
            data = f.readlines()

        n = len(data)

        for i in tqdm(trange(n)):
            record = json.loads(data[i])
            # <CUR> oracle_query <CTX> cq_{k-1} <SEP> ... cq_{1} (<BOS>)
            sample_id = record['sample_id']
            flat_q_concat = []
            flat_qa_concat = []
            #flat_qp_concat = []
            ctx_utts_text = record['ctx_utts_text']
            cur_utt_text = record['cur_utt_text']

            cur_utt = tokenizer.encode(cur_utt_text, add_special_tokens = True, max_length = args.max_query_length)
            flat_q_concat.extend(cur_utt)
            flat_qa_concat.extend(cur_utt)
            #flat_qp_concat.extend(cur_utt)
                
            for j in range(len(ctx_utts_text) - 1, -1, -1):
                if j % 2 == 1: # answer
                    max_length = args.max_response_length
                elif j % 2 == 0: # query
                    max_length = args.max_query_length
                utt = tokenizer.encode(ctx_utts_text[j], add_special_tokens=True, max_length=max_length, truncation=True) # not remove [CLS]
                if len(flat_qa_concat) + len(utt) > args.max_concat_length:
                    flat_qa_concat += utt[:args.max_concat_length - len(flat_qa_concat) - 1] + [utt[-1]]    # must ended with [SEP]
                    if j % 2 == 0:
                        flat_q_concat += utt[:args.max_concat_length - len(flat_q_concat) - 1] + [utt[-1]]    # must ended with [SEP]
                    break
                else:
                    flat_qa_concat.extend(utt) 
                    if j % 2 == 0:
                        flat_q_concat.extend(utt)

            cur_utt, cur_utt_mask = padding_seq_to_same_length(cur_utt, max_pad_length = args.max_query_length)
            flat_q_concat, flat_q_concat_mask = padding_seq_to_same_length(flat_q_concat, max_pad_length = args.max_concat_length)
            flat_qa_concat, flat_qa_concat_mask = padding_seq_to_same_length(flat_qa_concat, max_pad_length = args.max_concat_length)

            self.examples.append([sample_id, 
                            cur_utt,
                            cur_utt_mask,
                            flat_q_concat,
                            flat_q_concat_mask,
                            flat_qa_concat,
                            flat_qa_concat_mask])
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def get_collate_fn(args):
        
        def collate_fn(batch: list):
            collated_dict = {"bt_sample_ids": [],
                             "bt_raw_query": [],
                             "bt_raw_query_mask": [],
                             "bt_conv_q": [],
                             "bt_conv_q_mask": [],
                             "bt_conv_qa": [],
                             "bt_conv_qa_mask": [],
                            }
            for example in batch:
                collated_dict["bt_sample_ids"].append(example[0])
                collated_dict["bt_raw_query"].append(example[1])
                collated_dict["bt_raw_query_mask"].append(example[2])
                collated_dict["bt_conv_q"].append(example[3])
                collated_dict["bt_conv_q_mask"].append(example[4])
                collated_dict["bt_conv_qa"].append(example[5])
                collated_dict["bt_conv_qa_mask"].append(example[6])

            not_need_to_tensor_keys = set(["bt_sample_ids", "bt_cur_utt_text", "bt_oracle_utt_text"])

            for key in collated_dict:
                if key not in not_need_to_tensor_keys:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)
                    
            return collated_dict

        return collate_fn


class Test_Retrieval_cast(Dataset):
    def __init__(self, args, tokenizer, filename, collection=None):
        self.examples = []
        
        with open(filename, encoding="utf-8") as f:
            data = f.readlines()

        n = len(data)

        for i in tqdm(trange(n)):
            record = json.loads(data[i])
            # <CUR> oracle_query <CTX> cq_{k-1} <SEP> ... cq_{1} (<BOS>)
            flat_q_concat = []
            flat_qp_concat = []
            #sample_id = record['id']
            #conv_id = record['topic_number']
            #turn_id = record['query_number']
            #cur_utt_text = record["input"][-1] 
            #ctx_utts_text = record["input"][:-1]

            sample_id = record['turn_id']
            cur_utt_text = record["query"] 
            ctx_utts_text = record["context_qs"]
            if len(record["rewrite"]) > 0:
                rewrite = record["rewrite"][0]
            else:
                rewrite = cur_utt_text

            #if int(turn_id) > 1 and int(conv_id) > 80:
            #    last_response = data[i - 1]["automatic_response"][-1]
            #else:
            last_response = ""

            cur_utt = tokenizer.encode(cur_utt_text, add_special_tokens = True, max_length = args.max_query_length)
            rewrite = tokenizer.encode(rewrite, add_special_tokens = True, max_length = args.max_query_length)
            flat_q_concat.extend(cur_utt)
            #flat_qa_concat.extend(cur_utt)
            flat_qp_concat.extend(cur_utt)
            
            if len(last_response) > 0:
                lp = []
                lp.append(tokenizer.cls_token_id)
                lp.extend(tokenizer.convert_tokens_to_ids(["<response>"]))
                lp.extend(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(last_response)))
                lp = lp[:args.max_doc_length]
                lp.append(tokenizer.sep_token_id)
                flat_qp_concat.extend(lp)
            
            for j in range(len(ctx_utts_text) - 1, -1, -1):
                utt = tokenizer.encode(ctx_utts_text[j], add_special_tokens=True, max_length=args.max_query_length, truncation=True) # not remove [CLS]
                if len(flat_q_concat) + len(utt) > args.max_concat_length:
                    flat_q_concat += utt[:args.max_concat_length - len(flat_q_concat) - 1] + [utt[-1]]    # must ended with [SEP]
                    break
                else:
                    flat_q_concat.extend(utt)

            #cur_utt, cur_utt_mask = padding_seq_to_same_length(cur_utt, max_pad_length = args.max_query_length)
            rewrite, rewrite_mask = padding_seq_to_same_length(rewrite, max_pad_length = args.max_query_length)
            flat_q_concat, flat_q_concat_mask = padding_seq_to_same_length(flat_q_concat, max_pad_length = args.max_concat_length)
            #flat_qa_concat, flat_qa_concat_mask = padding_seq_to_same_length(flat_qa_concat, max_pad_length = args.max_concat_length)
            flat_qp_concat, flat_qp_concat_mask = padding_seq_to_same_length(flat_qp_concat, max_pad_length = args.max_concat_length)

            self.examples.append([sample_id, 
                            #cur_utt,
                            #cur_utt_mask,
                            flat_q_concat,
                            flat_q_concat_mask,
                            flat_qp_concat,
                            flat_qp_concat_mask,
                            rewrite,
                            rewrite_mask
                            ])
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def get_collate_fn(args):
        
        def collate_fn(batch: list):
            collated_dict = {"bt_sample_ids": [],
                             #"bt_raw_query": [],
                             #"bt_raw_query_mask": [],
                             "bt_conv_q": [],
                             "bt_conv_q_mask": [],
                             "bt_conv_qp": [],
                             "bt_conv_qp_mask": [],
                             "bt_rewrite": [],
                             "bt_rewrite_mask": []
                            }
            for example in batch:
                collated_dict["bt_sample_ids"].append(example[0])
                #collated_dict["bt_raw_query"].append(example[1])
                #collated_dict["bt_raw_query_mask"].append(example[2])
                collated_dict["bt_conv_q"].append(example[1])
                collated_dict["bt_conv_q_mask"].append(example[2])
                collated_dict["bt_conv_qp"].append(example[3])
                collated_dict["bt_conv_qp_mask"].append(example[4])
                collated_dict["bt_rewrite"].append(example[5])
                collated_dict["bt_rewrite_mask"].append(example[6])

            not_need_to_tensor_keys = set(["bt_sample_ids", "bt_cur_utt_text", "bt_oracle_utt_text"])

            for key in collated_dict:
                if key not in not_need_to_tensor_keys:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)
                    
            return collated_dict

        return collate_fn


class Search_Rel_Passage(Dataset):
    def __init__(self, args, tokenizer, filename):
        self.examples = []
        
        with open(filename, encoding="utf-8") as f:
            data = f.readlines()

        n = len(data)

        for i in tqdm(trange(n)):
            record = json.loads(data[i])
            # <CUR> oracle_query <CTX> cq_{k-1} <SEP> ... cq_{1} (<BOS>)
            sample_id = str(record["conversation_id"]) + '-' + str(record["turn"])
            cur_query = record["query"]
            cur_answer = record["answer"]
            query_context = record["context"]
            cur_topic = record["topic"]
            answer_context = record["answer_context"]

            search_query = []
            cur_utt = tokenizer.encode(cur_query, add_special_tokens = True, max_length = args.max_query_length)
            answer = tokenizer.encode(cur_answer, add_special_tokens = True, max_length = args.max_response_length)
            topic = tokenizer.encode(cur_topic, add_special_tokens = True, max_length = args.max_query_length)

            if args.query_type == "q":
                search_query.extend(cur_utt)
            elif args.query_type == "q+a":
                search_query.extend(cur_utt)
                search_query.extend(answer)
            elif args.query_type == "q+topic":
                search_query.extend(cur_utt)
                search_query.extend(topic)
            elif args.query_type == "q+a+topic":
                search_query.extend(cur_utt)
                search_query.extend(answer)
                search_query.extend(topic)
            
            search_query, search_query_mask = padding_seq_to_same_length(search_query, max_pad_length = args.max_query_length)

            self.examples.append([sample_id, 
                            search_query,
                            search_query_mask])
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def get_collate_fn(args):
        
        def collate_fn(batch: list):
            collated_dict = {"bt_sample_ids": [],
                             "bt_search_query": [],
                             "bt_search_query_mask": []
                            }
            for example in batch:
                collated_dict["bt_sample_ids"].append(example[0])
                collated_dict["bt_search_query"].append(example[1])
                collated_dict["bt_search_query_mask"].append(example[2])

            not_need_to_tensor_keys = set(["bt_sample_ids"])

            for key in collated_dict:
                if key not in not_need_to_tensor_keys:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)
                    
            return collated_dict

        return collate_fn

