import data
import torch
from sentence_transformers import SentenceTransformer
from llmrankers.setwise import SetwiseLlmRanker

model_path = "/mnt/netstore1_home/behrooz.mansouri/HF/llama-3.1-instruct-8B"
docs_path = "Answers.csv"
topics_path = "topics_1.csv"
initial_result_path = "top100s.tsv"

topic_dict, q_id_map, q_batch = data.getTopics(docs_path)
doc_dict, d_id_map, d_batch = data.getDocs(topics_path)

# run initial model top get top 100s
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("sentence-transformers/multi-qa-MiniLM-L6-cos-v1", device=device)
q_embs = data.getEmbeddings(q_batch, model)
d_embs = data.getEmbeddings(d_batch, model)
data.writeTopN(q_embs, d_embs, q_id_map, d_id_map, "bi_encoder", "top100s.tsv")

results = data.readTSV(initial_result_path, doc_dict)
# run setwise reranker
ranker = SetwiseLlmRanker(model_name_or_path=model_path,
                          tokenizer_name_or_path=model_path,
                          device=device,
                          num_child=10,
                          scoring='generation',
                          method='heapsort',
                          k=10)

for q_id, result in results.items():
    print(ranker.rerank(topic_dict[q_id], results[q_id])[0])