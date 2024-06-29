# tinyBPE - Trainable tokenizer based on Byte-Pair Encoding similar to GPT-2 and GPT-4

For LLM to understand texts it needs a translator between text and number, that is called ```Tokenizer```. For LLMs Byte Pair Encoding is the most used algorithm to avoid very little character level tokenizers and very huge word level/ n-gram level tokenizers.

The algorithm is inspired from the folloing references.

* OpenAI-GPT2 Paper: [GPT-2 Paper Link](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

* Wikipedia: [Byte Pair Encoding](https://en.wikipedia.org/wiki/Byte_pair_encoding)

* Video Suggestions by Andrej Karpathy: [Let's build the GPT Tokenizer](https://youtu.be/zduSFxRajkE?si=cNKo7AsE4iSppijW)


## File and Code Descriptions

* [tinyBPE/base.py](tinyBPE/base.py) conatins the helper functions like ```get_pair_counts```, ```merge_pairs```, ```replace_control_chars```, ```render_tokens``` and the base class ```BaseBPETokenizer``` for all tokenizers with ```to_local``` and ```from_loal``` functions.

* [tinyBPE/bytelevel.py](!tinyBPE/bytelevel.py) file contains a very basic level of tokenizer ```ByteLevelBPETokenizer``` where the base splitting is byte level. Then the merges are performed.

* [tinyBPE/regexBPE.py](!tinyBPE/regexBPE.py) this implements ```RegexBPETokenizer``` class which incorporates ```Regular Expressions``` for initial splittingg to optimize token splitting.


## Documentation
1. ### Training & Inference
```python
>>> from tinyBPE import ByteLevelBPETokenizer, RegexBPETokenizer
>>> tokenizer = ByteLevelBPETokenizer()
>>> text = """VERY_LONG_TEXT"""
>>> tokenizer.train(text, vocab_size= 4096, verbose=True)
>>> tokenizer.to_local("tokenizer") 
>>> # above function will generate tokenizer.tbpe which will be used for loading. tokenizer.vocab is a lossy version and will just for human interpretation
>>> tokens = tokenizer.encode("VERY_LONG_TEXT")
>>> tokenizer.decode(tokens)
"VERY_LONG_TEXT"
```
2. ### Infer with Special Tokens
```python
>>> from tinyBPE import RegexBPETokenizer
>>> tokenizer = RegexBPETokenizer()
>>> sp_tokens = {
    "<|startoftext|>" : 256,
    "<|endoftext|>" : 257,
    "<|midprompt|>" : 258
}
>>> text = "<|startoftext|> this is a new hello random text <|endoftext|>"
>>> tokenizer.add_special_tokens(sp_tokens)
>>> tokens = tokenizer.encode(text, consider_special_tokens = "ALL")
>>> tokenizer.decode(tokens)
"<|startoftext|> this is a new hello random text <|endoftext|>"
```


## My Contributions:
1. Removed repeatative function calls to one single call
2. Updated ```GPT2_SPLIT_PATTERN``` and ```GPT4_SPLIT_PATTERN``` to take care of multiple languages in ```regexBPE.py```
3. Updated ```to_local``` and ```from_local``` function by removing the dependency of order of merges in .tbpe file.  