import data
import torch
import csv
import argparse
from collections import defaultdict
from llmrankers.rankers import SearchResult
from llmrankers.setwise import SetwiseLlmRanker
from llmrankers.listwise import ListwiseLlmRanker


parser = argparse.ArgumentParser(description="Run reranker")

parser.add_argument("topics_path", type=str, help="Path to the topics file")
parser.add_argument("docs_path", type=str, help="Path to the documents file")
parser.add_argument("rerank_type", type=str, help="Reranker type to use (setwise | listwise)")
parser.add_argument("initial_result_path", type=str, help="Path to the initial result file to rerank")
parser.add_argument("outfile_name", type=str, help="Desired output file name")

args = parser.parse_args()

topics_path = args.topics_path
docs_path = args.docs_path
rerank_type = args.rerank_type
initial_result_path = args.initial_result_path
outfile_name = args.outfile_name

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "google/flan-t5-base"
#model_path = "/mnt/netstore1_home/aidan.bell@maine.edu/HF/Meta-Llama-3.1-8B-Instruct"

# read in a TREC format TSV file to be reranked
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

# rerank and write results to a file
def writeRerank(output_path:str, run_name:str, ranker:SetwiseLlmRanker):
	with open(output_path, "w", newline='') as f:
		writer = csv.writer(f, delimiter='\t')
		for q_id, result in results.items():
			reranks = ranker.rerank(topic_dict[q_id], results[q_id])
			rank = 1
			for rerank in reranks:
				writer.writerow([q_id, "Q0", rerank.docid, rank, rerank.score, run_name])
				print([q_id, "Q0", rerank.docid, rank, rerank.score, run_name])
				rank+=1

topic_dict, q_id_map, q_batch = data.getTopics(topics_path)
doc_dict, d_id_map, d_batch = data.getDocs(docs_path)

results = readTSV(initial_result_path, doc_dict)
# run setwise reranker
if rerank_type == "setwise":
	ranker = SetwiseLlmRanker(model_name_or_path=model_path,
							tokenizer_name_or_path=model_path,
							device=device,
							num_child=10,
							scoring='generation',
							method='heapsort',
							k=10)
elif rerank_type == "listwise":
	ranker = ListwiseLlmRanker(model_name_or_path=model_path,
                          tokenizer_name_or_path=model_path,
                          device=device,
                          window_size=4,
						  step_size=2,
                          scoring='generation',
                          num_repeat=1,
                          cache_dir=None)
else:
	raise ValueError("No reranker type selected")
writeRerank(outfile_name, "prompt1_1", ranker)