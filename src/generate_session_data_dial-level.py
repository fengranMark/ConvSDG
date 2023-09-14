import json
import openai
import time

with open('./datasets/Cast19/train_topics_cast19.jsonl') as f:
    topic_data = f.readlines()

with open('./datasets/Cast19/eval_topics.jsonl') as f:
    topic_data_2 = f.readlines()

topics = []
for line in topic_data:
    topic = json.loads(line)["title"]
    if topic not in topics:
        topics.append(topic)

for line in topic_data_2:
    topic = json.loads(line)["title"]
    if topic not in topics:
        topics.append(topic)

print(len(topics))

# OpenAI API key
openai.api_key = ""
model_engine = "gpt-3.5-turbo"


# generate conversation using ChatGPT
def generate_conversation(topic, conversation_id):
    queries = []
    answers = []
    context = []
    ans_context = []
    retry_count = 0

    while retry_count < 3:
        try:
            prompt = "Generate a conversation between human and system. In this conversation, human is asking and system is answering, the conversation topic would be" + topic 
            response = openai.ChatCompletion.create(
                model = model_engine,
                messages = [{"role":"system","content":"you are a helpful assitant"},{"role":"user","content":f"Generate a long conversation between human and system. In this conversation, human is asking and system is answering, the conversation topic would be" + topic + ".\n"}]
                )
            cnv = response["choices"][0]["message"]["content"].split("\n\n")
            qa_dict = {}
            for i in range(len(cnv)):
                if i % 2 == 0:
                    query = cnv[i].replace("Human: ","")
                    query = query.strip()
                else:
                    answer = cnv[i].replace("System: ","")
                    answer = answer.strip()
                    qa_dict[query] = answer
            queries = list(qa_dict.keys())
            answers = list(qa_dict.values())
            conversation = {}
            for i in range(len(queries)):
                context = []
                ans_context = []
                if i > 0:
                    context = queries[:i]
                    ans_context = answers[:i]
                conversation["conversation_id"] = conversation_id
                conversation["turn"] = i+1
                conversation["query"] = queries[i]
                conversation["answer"] = answers[i]
                conversation["context"] = context
                conversation["topic"] = topic
                conversation["answer_context"] = ans_context
                    
                with open("./datasets/generated_data/cast19_topic.jsonl", "a+") as f:
                    json.dump(conversation, f)
                    f.write('\n')
                conversation.clear()
                time.sleep(1)

            return conversation
        
        except openai.error.APIConnectionError as e:
            print(f"Encountered an error: {e}")
            print(f"Retrying in 5 seconds...")
            retry_count +=1
            time.sleep(5)
    
    print("Failed to generate conversation.")
    return None


conversations = []
begin_id = 1
conversation_id = begin_id
for i in range(begin_id - 1, len(topics)):
    topic = topics[i]
    print(f"Generating conversation for topic: {topic}, conversation id: {conversation_id}")
    conversation = generate_conversation(topic, conversation_id)
    if conversation is not None:
        conversations.extend(conversation)
        conversation_id += 1

print("Done!")
