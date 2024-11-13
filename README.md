# For getting initial results to rerank:
Tested on Python version 3.12.0

install requirements1.txt
python initialResults.py <topics_path.json> <docs_path.json> <outfile_name>

example: python initialResults.py topics_1.json Answers.json top100s.tsv

# For reranking intial results:
Tested on Python version 3.9.0

install requirements2.txt
python main.py <topics_path.json> <docs_path.json> <rerank_type> <initial_result_path> <outfile_name>

rerank_type can be "setwise" or "listwise"

example: python main.py topics_1.json Answers.json setwise top100s.tsv prompt1_1.tsv
