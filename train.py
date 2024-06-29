from tinyBPE import ByteLevelBPETokenizer, RegexBPETokenizer, BaseBPETokenizer
import os



with open("test/code-data.txt", "r", encoding='utf-8') as f:
    text = f.read()
os.makedirs("models", exist_ok=True)

for Tokenizer in [ByteLevelBPETokenizer, RegexBPETokenizer]:
    print(f"Training: {Tokenizer.__name__}")

    tokenizer : BaseBPETokenizer = Tokenizer()
    tokenizer.train(text, 256 + 1, verbose=True)

    model_name = Tokenizer.__name__
    # tokenizer.to_local(os.path.join("models",model_name))