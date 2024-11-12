import data
import torch
import csv
from collections import defaultdict
from llmrankers.rankers import SearchResult
from llmrankers.setwise import SetwiseLlmRanker
from llmrankers.listwise import ListwiseLlmRanker

def readTSV(result_path:str, docs:dict) -> dict:
		result = defaultdict(list)
		with open(result_path, "r") as file:
			reader = csv.reader(file, delimiter="\t")
			for row in reader:
				qid = row[0]
				doc_id = row[2]
				score = row[4]
				result[qid].append(SearchResult(docid=doc_id, score=score, text=docs[doc_id]))
		return result # defaultdict(<class 'list'>, {'q_id': [sr1, sr2, ...], ...})

model_path = "google/flan-t5-base"
#model_path = "/mnt/netstore1_home/aidan.bell@maine.edu/HF/Meta-Llama-3.1-8B-Instruct"
docs_path = "Answers.json"
topics_path = "topics_1.json"
initial_result_path = "top100s.tsv"

topic_dict, q_id_map, q_batch = data.getTopics(topics_path)
doc_dict, d_id_map, d_batch = data.getDocs(docs_path)
device = "cuda" if torch.cuda.is_available() else "cpu"

results = readTSV(initial_result_path, doc_dict)
# run setwise reranker
ranker = SetwiseLlmRanker(model_name_or_path=model_path,
                          tokenizer_name_or_path=model_path,
                          device=device,
                          num_child=10,
                          scoring='generation',
                          method='heapsort',
                          k=10)

with open("prompt1_1.tsv", "w", newline='') as f:
	writer = csv.writer(f, delimiter='\t')
	for q_id, result in results.items():
		reranks = ranker.rerank(topic_dict[q_id], results[q_id])
		rank = 1
		for rerank in reranks:
			writer.writerow([q_id, "Q0", rerank.docid, rank, rerank.score, "prompt1_1"])
			print([q_id, "Q0", rerank.docid, rank, rerank.score, "prompt1_1"])
			rank+=1