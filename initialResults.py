import data
import torch
import argparse
from sentence_transformers import SentenceTransformer

parser = argparse.ArgumentParser(description="Initial ranks")

parser.add_argument("topics_path", type=str, help="Path to the topics file")
parser.add_argument("docs_path", type=str, help="Path to the documents file")
parser.add_argument("outfile_name", type=str, help="Desired output file name")

args = parser.parse_args()

docs_path = args.docs_path
topics_path = args.topics_path
outfile_name = args.outfile_name

topic_dict, q_id_map, q_batch = data.getTopics(topics_path)
doc_dict, d_id_map, d_batch = data.getDocs(docs_path)

# run initial model top get top 100s
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("sentence-transformers/multi-qa-MiniLM-L6-cos-v1", device=device)
q_embs = data.getEmbeddings(q_batch, model)
d_embs = data.getEmbeddings(d_batch, model)
data.writeTopN(q_embs, d_embs, q_id_map, d_id_map, "bi_encoder", outfile_name)
