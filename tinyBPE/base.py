"""Contains helper functions and base Tokenizer class with save and load functionalities
The base class Tokenizer can be extended for further implementations."""
from typing import Union
import unicodedata



def get_pair_counts(ids, counts : Union[dict, None] = None):
    """Returns the pair wise counts for a sequence of ids.

    Example:
    ```python 
    >>> ids = [1,2,3,1,2]
    >>> get_pair_counts(ids)
    >>> {(1,2) : 2, (2,3) : 1, (3,1) : 1}
    ```"""
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge_pairs(ids : list[int], pair : Union[list[int], tuple[int]], idx : int):
    """Merged the byte pairs in the ids sequence and replace with the new idx"""
    updated_ids = []
    i = 0
    lenids = len(ids)
    while i < lenids:
        # if its not the last id and pairs are matching then replace
        if (ids[i] == pair[0]) and (i < lenids - 1) and (ids[i+1] == pair[1]):
            updated_ids.append(idx)
            i+=2
        # if its not matching or i is at the last index then just use the same i
        else:
            updated_ids.append(ids[i])
            i+=1
    return updated_ids

def replace_control_chars(text : str):
    """This function removed the control characters 
    which can cause distortion while printing.

    Ref:  https://www.unicode.org/reports/tr44/#GC_Values_Table
    """
    filtered_chars = []
    for c in text:
        if unicodedata.category(c)[0] != "C": # check ref link. why "C" categories are being removed
            filtered_chars.append(c)
        else:
            filtered_chars.append(f"\\u{ord(c):04x}")
    return "".join(filtered_chars)

def render_tokens(t : bytes):
    """Rendering the tokens from bytes"""
    decoded = t.decode('utf-8', errors='replace')
    text = replace_control_chars(decoded)
    return text


class BaseBPETokenizer:
    def __init__(self) -> None:
        """Base class for all Byte-Pair-Encoding tokenizers
        
        * Intial vocab of 256 characters.
        * No regex patterns for splitting."""
        self.merges = {} # (int, int) -> int
        self.special_tokens = {} # str -> int e.g., <|eos|> -> 100257
        self.pattern = ""
        self.vocab = self._build_vocab() # int -> bytes
    
    def _build_vocab(self):
        """Prepare the vocabulary from basic bytes, merges and special tokens"""
        vocab = {i : bytes([i]) for i in range(256)}
        for pair, idx in self.merges.items():
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
        for sp_token, idx in self.special_tokens.items():
            vocab[idx] = sp_token.encode("utf-8")
        return vocab

    def train(self, text : str, vocab_size : int, verbose : bool = False):
        """Training BPE algorithm on custom data"""
        raise NotImplementedError("This method is not implemented in base class")
    
    def encode_chunk(self, text_bytes: bytes) -> list[int]:
        tokens = list(text_bytes) # returns list of int

        # perform merging if there tokens more than 2
        while len(tokens) >=2:
            counts = get_pair_counts(tokens)

            # What below line does?
            # This will create a list from the pairs from the counts dict
            # then check if that pair is present in the merges or not. 
            # If not return Inf else the return the new index.
            # Noe pick the pair with minimum index.
            # merge until all the merges are checked this way.
            min_merge_idx_pair = min(counts, key= lambda pair : self.merges.get(pair, float('inf')))

            # if the pair doesn't exists in the merges that means we have merged all the possible merges
            # then all the values will be Inf in the min function above.
            if min_merge_idx_pair not in self.merges.keys():
                break
            else:
                tokens = merge_pairs(tokens, min_merge_idx_pair, self.merges[min_merge_idx_pair])
        return tokens

    def encode(self, text : str) -> list[int]:
        """Method to encode piece of text to tokens with trained BPE"""
        raise NotImplementedError("This method is not implemented in base class")

    def decode(self, ids : list[int]) -> str:
        """Method to decode encoded text"""
        raise NotImplementedError("This method is not implemented in base class")

    def to_local(self, file_initial : str):
        """Save vocab and merges documents to local file for future use"""
        model_file = file_initial + ".tbpe"
        with open(model_file, "w") as f:
            f.write("tinyBPE/v1.0\n")
            f.write(f"{self.pattern}\n")
            f.write(f"{len(self.special_tokens)}\n")
            for token, idx in self.special_tokens.items():
                f.write(f"{token} {idx}\n")
            for pair, idx in self.merges.items():
                f.write(f"{pair[0]} {pair[1]} {idx}\n")
        
        # vocab file is just for human interpritation. This is not used for building tokenizer
        # render_token is replacing the chcracters (partial bytes which are non-utf-8) by �
        # therefore .vocab file is a lossy way of conversion and should not be used for loading
        vocab_file = file_initial + ".vocab"
        inverted_merge = {idx:pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding='utf-8') as f:
            for idx, token in self.vocab.items():
                token = render_tokens(token)
                if idx in inverted_merge:
                    # there is a merge then will store the merging pair and new pair
                    pair0, pair1 = inverted_merge[idx]
                    pair0 = render_tokens(self.vocab[pair0])
                    pair1 = render_tokens(self.vocab[pair1])
                    f.write(f"[{pair0}][{pair1}]->[{token}] {idx}\n")
                else:
                    # there is no merge for that idx and it is one of the 256 raw bytes
                    f.write(f"[{token}] {idx}\n")
    
    def from_local(self, model_file : str):
        """Method to load a pratrained model"""
        assert model_file.endswith(".tbpe")

        merges = {}
        special_tokens = {}

        with open(model_file, "r", encoding='utf-8') as f:
            # version checking with .tbpe files
            version_ = f.readline().strip()
            assert version_ == "tinyBPE/v1.0"

            # retrieve the regex pattern for splitting
            self.pattern = f.readline().strip()

            # load special tokens for next n_special_tokens lines
            n_special_tokens = int(f.readline().strip())
            for i in range(n_special_tokens):
                token, idx = f.readline().strip().split()
                special_tokens[token] = int(idx)

            # load merges performed duting training
            for line in f:
                # this for loop will generate the next lines
                # line is "<int> <int> <int>"
                # strip and split will return ["int", "int", "int"]
                # mapping it will convert to ints [int, int, int]
                pair_0, pair_1, newidx = map(int, line.strip().split())
                merges[(pair_0, pair_1)] = newidx
            
            # assign the attributes
            self.merges = merges
            self.special_tokens = special_tokens

            # build the vocab from merges and special tokens
            self.vocab = self._build_vocab()