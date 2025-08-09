import os
from typing import BinaryIO
from concurrent.futures import ProcessPoolExecutor
import regex as re
from collections import defaultdict
import pickle

class BPE:
    def __init__(
        self, input_path: str, 
        vocab_size: int, 
        special_tokens: list[str],
    ):
        self.input_path: str = input_path
        self.vocab_size: int = vocab_size
        self.special_tokens: list[str] = special_tokens
        self.split_special_token: bytes = b'<|endoftext|>' if '<|endoftext|>' in special_tokens else special_tokens[0]
        
        self.vocab: dict[int, bytes] = {}
        self.merges: list[tuple[bytes, bytes]] = []
        
        self.vocab_list: list[bytes] = []
        for st in self.special_tokens:
            self.vocab_list.append(st.encode("utf-8"))
        for i in range(256):
            self.vocab_list.append(bytes([i]))
        
    
    def train(self):
        file = open(self.input_path, "rb")
        num_cpus = os.cpu_count()
        boundaries = self.find_chunk_boundaries(file, num_cpus, self.split_special_token)
        file.close()
        
        pre_tokens: dict[tuple, int] = defaultdict(int)
        with ProcessPoolExecutor(max_workers=num_cpus) as executor:
            futures = [
                executor.submit(
                    BPE.pre_tokenize, 
                    self.input_path,
                    self.special_tokens,
                    start, 
                    end
                )
                for start, end in zip(boundaries[:-1], boundaries[1:])
            ]
            for future in futures:
                res = future.result()
                for k, v in res.items():
                    pre_tokens[k] += v
                    
        byte_pairs: dict[tuple, int] = defaultdict(int)
        pair_blongs: dict[tuple, set[tuple]] = defaultdict(set)
        for word, value in pre_tokens.items():
            for token1, token2 in zip(word[:-1], word[1:]):
                byte_pairs[(token1, token2)] += value
                pair_blongs[(token1, token2)].add(word)
                
        while len(self.vocab_list) < self.vocab_size:
            max_pair = max(byte_pairs.items(), key=lambda x:(x[1], x[0]))
            self.merges.append(max_pair[0])

            words = []
            new_words = []
            for word in pair_blongs[max_pair[0]]:
                new_word = self.merge_pair_in_pretoken_(word, max_pair[0])
                pre_tokens[new_word] += pre_tokens[word]
                words.append(word)
                new_words.append(new_word)

            for word in words:
                for token1, token2 in zip(word[:-1], word[1:]):
                    byte_pairs[(token1, token2)] -= pre_tokens[word]
                    pair_blongs[(token1, token2)].discard(word)
                del pre_tokens[word]
                
            for new_word in new_words:
                for token1, token2 in zip(new_word[:-1], new_word[1:]):
                    byte_pairs[(token1, token2)] += pre_tokens[new_word]
                    pair_blongs[(token1, token2)].add(new_word)
                                
            self.vocab_list.append(max_pair[0][0]+max_pair[0][1])
        self.vocab = {i: b for i, b in enumerate(self.vocab_list)}

            
    def get_merge(self):
        return self.merges
    
    def get_vocab(self):
        return self.vocab
    
    def save(self, vocab_filepath, merges_filepath):
        with open(vocab_filepath, "wb") as f:
            pickle.dump({"vocab": self.vocab}, f)
        
        with open(merges_filepath, "wb") as f:
            pickle.dump({"merges": self.merges}, f)
            
        
    
    def merge_pair_in_pretoken_(self, word: tuple[bytes], pair: tuple[bytes]) -> tuple[bytes]:
        merged = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i+1]) == pair:
                merged.append(word[i] + word[i+1])
                i += 2
            else:
                merged.append(word[i])
                i += 1
        return tuple(merged)
        
        
    @staticmethod
    def pre_tokenize(input_path, special_tokens: list[str], start, end) -> dict[tuple[bytes], int]: 
        result = defaultdict(int)
        with open(input_path, "r") as f:
            f.seek(start)
            data = f.read(end - start)
            split_pattern = "|".join(re.escape(st) for st in special_tokens)
            split_data = re.split(split_pattern, data)
            PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            
            for part in split_data:
                if part in special_tokens:
                    continue
                for matched in re.finditer(PAT, part):
                    matched_data = matched.group().encode("utf-8")
                    result[tuple(bytes([b]) for b in matched_data)] += 1
                    
        return result
        
    
    def find_chunk_boundaries(
        self,
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes,
    ) -> list[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))
