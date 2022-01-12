import multiprocessing
import subprocess
import random

def do_it(file_index):
  subprocess.call(["python", "k/jsonl_to_x", f"data/{file_index}.jsonl.zst", f"pytorch/{file_index}", "--normalize-with-ftfy", "--normalize-with-wikitext-detokenize", "--seed", random.randint(0, 60000), "--min-unique-tokens", "128"])

for i in range(8):
  # 8 cores
  # the pile has 29 jsonl files
  starting_index = 0 # next will be 7, then 15, then 23, then we'll be done
  p = multiprocessing.Process(target=do_it, args=(starting_index + i.zfill(2)))
  p.start()