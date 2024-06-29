"""Implementation of base regex based BPE tokenizer

What it does?
- Instead of byte level splitting, it performs pattern based splitting
- Handles special tokens like <|startoftext|> or <|endoftext|>"""

from typing import Union
import regex as re
from .base import BaseBPETokenizer, get_pair_counts, merge_pairs



# the main GPT text split patterns, see
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py

# GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
# GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# Updated pattern for non english sentences. \p{L} => \p{L}\p{M} this will consider most of the language wo split in words
# Update ref. https://github.com/openai/tiktoken/issues/292
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?[\p{L}\p{M}]+| ?\p{N}+| ?[^\s\p{L}\p{M}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+[\p{L}\p{M}]+|\p{N}{1,3}| ?[^\s\p{L}\p{M}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RegexBPETokenizer(BaseBPETokenizer):
    def __init__(self, pattern : Union[str , None] = None) -> None:
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {} # str -> int
        self.inverse_special_tokens = {} # int -> str
    
    def train(self, text: str, vocab_size: int, verbose: bool = False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        text_chunks = re.findall(self.pattern, text, re.IGNORECASE)
        print(text_chunks)
        # This is peformed to keep the words separately and then merge the common parts in the words/chunks
        token_chunks = [list(ch.encode(encoding='utf-8')) for ch in text_chunks] # list[list[int]]

        merges = {}
        # start vocab with 256 raw bytes
        vocab = {idx : bytes([idx]) for idx in range(256)}

        for i in range(num_merges):
            counts = {}

            for token_chunk in token_chunks:
                # using modification of counts all over the token chunks
                counts = get_pair_counts(token_chunk, counts)
            if len(counts) < 1:
                if verbose:
                    print("Nothing to merge")
                break
            # get the pair with max count
            max_pair = max(counts, key=counts.get)
            new_id = 256+i
            
            # merge pairs
            token_chunks = [merge_pairs(token_chunk, max_pair, new_id) for token_chunk in token_chunks]
            
            # update the merges and vocab
            merges[max_pair] = new_id
            vocab[new_id] = vocab[max_pair[0]] + vocab[max_pair[1]]

            if verbose:
                print(f"Merge: {i+1}: {max_pair} -> {new_id}. ({vocab[new_id]}) occured {counts[max_pair]} times.")

        # update merges and vocab in attributes
        self.merges = merges
        self.vocab = vocab
    
    def encode_ordinary(self, text: str) -> list[int]:
        """This methods ignores special tokens present in the text. 
        This is useful when special tokens are not used"""

        # get the split
        text_chunks : list[str] = re.findall(self.compiled_pattern, text)

        token_ids = []
        # iterate over the splitted tokens and then extend
        for text_chunk in text_chunks:
            text_bytes = text_chunk.encode(encoding='utf-8')
            token_ids.extend(self.encode_chunk(text_bytes))
        
        return token_ids
    
    def encode(self, text: str, consider_special_tokens : Union[set, str] = "ALL") -> list[int]:
        """This method is a modified version of the base encode version for 
        handeling special tokens given some specific values.
        consider_special_tokens can be "ALL", "NONE", {set of special tokens as string}, "NONE_RAISE"
        """

        special_tokens = None
        if consider_special_tokens == "ALL":
            special_tokens = self.special_tokens
        elif consider_special_tokens == "NONE" : 
            special_tokens = {}
        elif consider_special_tokens == "NONE_RAISE":
            special_tokens = {}
            assert all([sp_token not in text for sp_token in self.special_tokens.keys()])
        elif isinstance(consider_special_tokens, set):
            # now it will become a dictionary from set
            special_tokens = {token:self.special_tokens[token] for token in special_tokens}
        else:
            raise ValueError(f"consider_special_tokens = '{consider_special_tokens}' is not allowed")
        
        if not special_tokens:
            # normal regex splitting tokenizer
            return self.encode_ordinary(text)
        
        # else special tokens to be handled separately without any more regex split in those
        splitting_pattern = f"({'|'.join(list(special_tokens.keys()))})" # (<|eot|>|<|startoftext|>|<|startofprompt|>)

        # split the text from the special tokens
        chunks = re.split(splitting_pattern, text)

        encoded_ids = [] # tokens/ ids  list[int]
        for chunk in chunks:
            if chunk in special_tokens:
                encoded_ids.append(special_tokens[chunk])
            else:
                encoded_ids.extend(self.encode_ordinary(chunk))
        return encoded_ids
    
    def add_special_tokens(self, special_tokens : dict):
        """special_token is a dict[str, int] which contains special tokens mapped with ids
        
        Remember: ids should be outside of vocab indexes else there will be an overlap."""

        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v:k for k,v in self.special_tokens.items()}
    
    def decode(self, ids: list[int], byte_mode : bool = False) -> str:
        byte_chunks = []
        for _id in ids:
            if _id in self.vocab:
                byte_chunks.append(self.vocab[_id])
            elif _id in self.special_tokens:
                byte_chunks.append(self.inverse_special_tokens[_id])
            else:
                raise ValueError(f"Invalid token id {_id}")
        byte_text = b"".join(byte_chunks)
        if byte_mode:
            return byte_text
        text = byte_text.decode(encoding='utf-8', errors='replace')
        return text