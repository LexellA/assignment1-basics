import pickle
import regex as re
from collections import defaultdict
from collections.abc import Iterable
from collections.abc import Iterator

class Tokenizer:
    def __init__(
        self, 
        vocab: dict[int, bytes], 
        merges: list[tuple[bytes, bytes]], 
        special_tokens: list[str] | None = None,
    ):
        """tokenize text

        Args:
            vocab (dict[int, bytes]): trained vocab
            merges (list[tuple[bytes, bytes]]): trained merges
            special_tokens (list[str] | None, optional): preserved token. Defaults to None.
        """
        self.vocab: dict[int, bytes] = vocab
        self.merges: list[tuple[bytes, bytes]] = merges
        self.special_tokens: list[str] | None = special_tokens
        
        if special_tokens is not None:
            for st in special_tokens:
                if st.encode('utf-8') not in self.vocab.values():
                    self.vocab[len(self.vocab)] = st.encode("utf-8")
                
        self.vocab_reverse = {v: i for i, v in self.vocab.items()}
        self.merges_rank = {pair: rank for rank, pair in enumerate(merges)}
    @classmethod
    def from_files(
        cls, 
        vocab_filepath, 
        merges_filepath, 
        special_tokens=None
    ):
        vocab = {}
        merges = []
        with open(vocab_filepath, "wb") as f:
            vocab = pickle.load(f)
            
        with open(merges_filepath, "wb") as f:
            merges = pickle.load(f)
        
        return cls(vocab, merges, special_tokens)
    
    def encode(self, text: str) -> list[int]:
        split_text = [text]
        if self.special_tokens is not None:
            sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
            split_pattern = "(" + "|".join(re.escape(st) for st in sorted_tokens) + ")"
            split_text = re.split(split_pattern, text)
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        result = []
        
        for split in split_text:
            if self.special_tokens is not None and split in self.special_tokens:
                result.append(self.vocab_reverse[split.encode('utf-8')])
                continue
            for word in re.finditer(PAT, split):
                result.extend(self.encode_word(word.group()))
        
        return result
    
    def encode_word(self, word: str):
        word_encode = word.encode('utf-8')
        result = []
        for c in word_encode:
            result.append(self.vocab_reverse[bytes([c])])
            
        while True:
            first_merge_rank = None
            first_merge_pos = -1
            for i in range(len(result) - 1):
                token1 = self.vocab[result[i]]
                token2 = self.vocab[result[i+1]]
                pair = (token1, token2)
                rank = self.merges_rank.get(pair)
                if rank is not None and (first_merge_rank is None or rank < first_merge_rank):
                    first_merge_rank = rank
                    first_merge_pos = i
            if first_merge_rank == None:
                break
            new_token = self.vocab[result[first_merge_pos]] \
                        + self.vocab[result[first_merge_pos+1]]
            new_id = self.vocab_reverse[new_token]
            result[first_merge_pos:first_merge_pos+2] = [new_id]
            if len(result) == 1:
                break
        return result
    
    def decode(self, ids: list[int])->str:
        result:bytes = b""
        for id in ids:
            try:
                result = result + self.vocab[id]
            except Exception:
                result = result + b'\xef\xbf\xbd'
        return result.decode('utf-8', errors='replace')
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)