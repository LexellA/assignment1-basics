from cs336_basics.BPE import BPE
from cs336_basics.tokenizer import Tokenizer
import time 

b = BPE(
    "data/example.txt", 
    266, 
    ["<|endoftext|>"],
)

b.train()
print(b.get_vocab())

t = Tokenizer(b.get_vocab(), b.get_merge())

res = t.encode("low widest")
print(res)