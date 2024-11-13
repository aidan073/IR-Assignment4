import data
import torch
from sentence_transformers import SentenceTransformer

docs_path = "Answers.json"
topics_path = "topics_2.json"

topic_dict, q_id_map, q_batch = data.getTopics(topics_path)
doc_dict, d_id_map, d_batch = data.getDocs(docs_path)

# run initial model top get top 100s
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("sentence-transformers/multi-qa-MiniLM-L6-cos-v1", device=device)
q_embs = data.getEmbeddings(q_batch, model)
d_embs = data.getEmbeddings(d_batch, model)
data.writeTopN(q_embs, d_embs, q_id_map, d_id_map, "bi_encoder", "top100s2.tsv")