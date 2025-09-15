from BPE import BPE
from tokenizer import Tokenizer
from pretokenization_example import find_chunk_boundaries
import numpy as np
import concurrent.futures
import os
from tqdm import tqdm

VOCAB_PATH = "data/tinystory/vocab"
MEREGS_PATH = "data/tinystory/merges"
TRAIN_PATH = "data/TinyStoriesV2-GPT4-train.txt"
VALID_PATH = "data/TinyStoriesV2-GPT4-valid.txt"
TRAIN_TOKENS_PATH = "data/tinystory/train_tokens"
VALID_TOKENS_PATH = "data/tinystory/valid_tokens"

# bpe = BPE(
#         "data/TinyStoriesV2-GPT4-train.txt",
#         10000,
#         ["<|endoftext|>"]
#     )

# bpe.train()
# bpe.save(VOCAB_PATH, MEREGS_PATH)

    
with open(TRAIN_PATH, "rb") as f:
    boundaries = find_chunk_boundaries(f, 4096, b"<|endoftext|>")

def _init_worker(vocab_path, merges_path, eos):
    global _TOKENIZER
    _TOKENIZER = Tokenizer.from_files(vocab_path, merges_path, eos)

def _init_writer(vocab_path, merges_path, eos, mmap_path, total_tokens):
    global _TOKENIZER, _MMAP, _TOTAL_TOKENS
    _TOKENIZER = Tokenizer.from_files(vocab_path, merges_path, eos)
    _TOTAL_TOKENS = total_tokens
    _MMAP = np.memmap(mmap_path, dtype=np.int32, mode='r+', shape=(total_tokens,))

#统计tokens数量
def cal_tokens(args):
    start, end, file_path = args
    with open(file_path, "rb") as f:
        f.seek(start)
        s = f.read(end - start).decode('utf-8', 'ignore')
        tokens = _TOKENIZER.encode(s)
        return len(tokens)

def write_tokens(args):
    start, end, file_path, offset = args
    with open(file_path, "rb") as f:
        f.seek(start)
        s = f.read(end - start).decode('utf-8', 'ignore')
        tokens = _TOKENIZER.encode(s)
        _MMAP[offset:offset + len(tokens)] = tokens

tasks = [(start, end, TRAIN_PATH) for start, end in zip(boundaries[:-1], boundaries[1:])]

print("计算总token数")
tokens_count = []
cpus = os.cpu_count() or 8
with concurrent.futures.ProcessPoolExecutor(
    max_workers=cpus, 
    initializer=_init_worker, 
    initargs=(VOCAB_PATH, MEREGS_PATH, ["<|endoftext|>"])
) as executor:
    tokens_count = list(tqdm(executor.map(cal_tokens, tasks, chunksize=16), total=len(tasks)))

total_tokens = 0
for tokens in tokens_count:
    total_tokens += tokens

print(f"总共{total_tokens}个令牌")
#total_tokens = 1558395438


os.makedirs(os.path.dirname(TRAIN_TOKENS_PATH), exist_ok=True)
mmap_array = np.memmap(TRAIN_TOKENS_PATH, dtype=np.int32, mode='w+', shape=(total_tokens,))
del mmap_array

offsets = np.cumsum([0] + tokens_count[:-1], dtype=np.int64)
write_tasks = [(start, end, TRAIN_PATH, offset) for start, end, offset in zip(boundaries[:-1], boundaries[1:], offsets)]

with concurrent.futures.ProcessPoolExecutor(
    max_workers=cpus,
    initializer=_init_writer,
    initargs=(VOCAB_PATH, MEREGS_PATH,["<|endoftext|>"], TRAIN_TOKENS_PATH, total_tokens)
) as executor:
    for _ in tqdm(executor.map(write_tokens, write_tasks, chunksize=8), total=len(write_tasks)):
        pass
    
mm = np.memmap(TRAIN_TOKENS_PATH, dtype=np.int32, mode='r+', shape=(total_tokens,))
mm.flush()
del mm
print("写入完成")
