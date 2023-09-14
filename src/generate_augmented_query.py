import json
import openai
import time

# OpenAI API key
openai.api_key = ""
model_engine = "gpt-3.5-turbo"

cur_line = 0
with open('datasets/Cast19/train_cast.jsonl', 'r') as f:
    lines = f.readlines()[cur_line:]

for line in lines:
    print("cur line is ", cur_line)
    data = json.loads(line)
    # first round
    new_data = {}
    cur_query = data["query"]
    qid = data["id"]
    #turn = data.get('turn_id', '')
    #conv_id = data.get('topic_id','')
    #cur_turn1 = turn + "-1"
    #cur_query = ' '.join(query_list)
    response = openai.ChatCompletion.create(
            model = model_engine,
            messages = [{"role":"system","content":"you are a helpful assitant"},{"role":"user","content":f"Transform one question:" + cur_query + "into another question with same meaning. Just give me the transformed question."+".\n"}]
            )
    cnv = response["choices"][0]["message"]["content"].split("\n\n")
    print(cnv)
    new_data["id"] = qid + "-1"
    new_data["ori_query"] = cur_query
    new_data["query"] = "".join(cnv)
    with open('datasets/Cast19/test_rewrite1.jsonl', 'a+') as outfile:
        json.dump(new_data, outfile)
        outfile.write('\n')
    time.sleep(1)

    # second round
    new_data = {}
    firstquery = "".join(cnv)
    response2 = openai.ChatCompletion.create(
            model = model_engine,
            messages = [{"role":"system","content":"you are a helpful assitant"},{"role":"user","content": f"Transform two questions with similar meanings:" + cur_query + firstquery + "into another question with same meaning. Just give me the transformed question." +".\n"}]
            )
    cnv2 = response2["choices"][0]["message"]["content"].split("\n\n")
    print(cnv2)
    #cur_turn2 = turn + "-2"
    new_data["id"] = qid + "-2"
    new_data["ori_query"] = cur_query
    new_data["query"] = "".join(cnv2)
    with open('datasets/Cast19/test_rewrite2.jsonl', 'a+') as outfile:
        json.dump(new_data, outfile)
        outfile.write('\n')
    time.sleep(1)
    cur_line += 1
