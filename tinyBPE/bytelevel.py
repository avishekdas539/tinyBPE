"""Character level BPE tokenizing - the most basic level of tokenizer.
This algorithmically follows the core BPE logic.

What is does?
- Performs character level tokenizing
- Merges starts from character level

When to use?
- When contex length is less
- Very basic level of language models
- Very less size of vocab required

What is does not?
- Doesn't use GPT regex splitting
"""


from .base import BaseBPETokenizer, get_pair_counts, merge_pairs


class ByteLevelBPETokenizer(BaseBPETokenizer):
    """This tokenizer doesnot consider special tokens as all are splitted into bytes"""
    def __init__(self) -> None:
        super().__init__()

    def train(self, text: str, vocab_size: int, verbose: bool = False):
        """Perform byte level splitting and training the tokenizer for 
        BPE until the vocab reaches the required size"""

        # vocab_size should be more that or equal to 256 as there are 256 raw bytes available
        # vocab_size =256 means no merges to be done and it will act as bacis character level tokenizer
        assert vocab_size >= 256

        num_merges = vocab_size - 256
        text_bytes = text.encode("utf-8")
        tokens = list(text_bytes) # this is doing the byte level splitting
        merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} # raw 256 bytes initially

        for i in range(num_merges):
            counts = get_pair_counts(tokens) # will return the pairwise counts

            if len(counts) < 1:
                if verbose:
                    print("There is nothing to merge")
                break
            max_pair = max(counts, key=counts.get) # this will return the key where count is max
            new_id = 256 + i
            # merge those occurances with new id
            tokens = merge_pairs(tokens, max_pair, new_id)

            # store the merge details
            merges[max_pair] = new_id

            # update vocab with the new bytes.
            vocab[new_id] = vocab[max_pair[0]] + vocab[max_pair[1]]

            if verbose:
                print(f"Merge: {i+1}: {max_pair} -> {new_id}. ({vocab[new_id]}) occured {counts[max_pair]} times.")

        # update object attributes
        self.vocab = vocab
        self.merges = merges

    def decode(self, ids: list[int], byte_mode : bool = False) -> str:
        """Method to decode a token ids to a byte string or string"""
        token_bytes = [self.vocab[i] for i in ids]
        bytes_string = b"".join(token_bytes)
        # below is a lossy conversion. if only raw bytes needed remove the decoded part
        if byte_mode:
            return bytes_string
        return bytes_string.decode(encoding='utf-8', errors='replace')
    
    def encode(self, text: str) -> list[int]:
        # tokens = list(text.encode(encoding='utf-8')) # returns list of int
        # # perform merging if there tokens more than 2
        # while len(tokens) >=2:
        #     counts = get_pair_counts(tokens)

        #     # What below line does?
        #     # This will create a list from the pairs from the counts dict
        #     # then check if that pair is present in the merges or not. 
        #     # If not return Inf else the return the new index.
        #     # Noe pick the pair with minimum index.
        #     # merge until all the merges are checked this way.
        #     min_merge_idx_pair = min(counts, key= lambda pair : self.merges.get(pair, float('inf')))

        #     # if the pair doesn't exists in the merges that means we have merged all the possible merges
        #     # then all the values will be Inf in the min function above.
        #     if min_merge_idx_pair not in self.merges.keys():
        #         break
        #     else:
        #         tokens = merge_pairs(tokens, min_merge_idx_pair, self.merges[min_merge_idx_pair])

        # method from base
        text_bytes = text.encode(encoding='utf-8')
        tokens = self.encode_chunk(text_bytes)
        return tokens