import pytest
from tinyBPE import ByteLevelBPETokenizer, RegexBPETokenizer


# test strings for multiple test cases below
# test 1 -> encoded and decoded strings are exactly same.
# test 2 -> special tokens are getting correctly encoded and decoded.
# test 3 -> Example from wikipedia page is working


test1 = [
    "Hi this is sample test case 1",

    """বিক্রমপুরের জমিদার বাড়িটা ছিল রহস্যে ঘেরা। 
    প্রবাল সেন, বর্তমান জমিদার, এই বাড়ির প্রতিটি ইটের সঙ্গে বড়ো হয়েছেন। 
    কিন্তু রাতের অन्धকারে, যখন ঝড়ো হাওয়া কাঁদে আর পুরোনো গাছের 
    পাতা কথা কહે, তখন এই বাড়িটা এক অন্য রূপ ধরে নেয়।"""
]

@pytest.mark.parametrize("tokenizer", [ByteLevelBPETokenizer(), RegexBPETokenizer()])
@pytest.mark.parametrize("text", test1)
def test1(tokenizer, text):
    tokens = tokenizer.encode(text=text)
    decoded = tokenizer.decode(tokens)
    assert text == decoded


test2 = [
    "<|startoftext|> this is a new hello random text <|endoftext|>",
    "this is another <|midprompt|> sample test"
]
sp_tokens = {
    "<|startoftext|>" : 256,
    "<|endoftext|>" : 257,
    "<|midprompt|>" : 258
}
@pytest.mark.parametrize("text", test2)
def test2(text):
    tokenizer = RegexBPETokenizer()
    tokenizer.add_special_tokens(sp_tokens)
    decoded = tokenizer.decode(tokenizer.encode(text))
    assert text == decoded

@pytest.mark.parametrize("tokenizer", [ByteLevelBPETokenizer(), RegexBPETokenizer()])
def test3(tokenizer):
    """As per the Wikipedia example the test case is
    text is "aaabdaaabac"
    
    Merge 1 will be Z = aa -> 256
    Merge 2 will be Y = ab -> 257
    Merge 3 will be X = ZY -> 258
    """
    text = "aaabdaaabac"
    tokenizer.train(text, vocab_size= 256 + 3, verbose=True)
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)

    assert tokens == [258, 100, 258, 97, 99]
    assert decoded == text


if __name__=="__main__":
    pytest.main()