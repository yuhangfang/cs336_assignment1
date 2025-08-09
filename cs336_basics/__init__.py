import importlib.metadata

__version__ = importlib.metadata.version("cs336_basics")

def test_print_2_1():
    print(chr(0))

    print(repr(chr(0)))

    print("this is a test" + chr(0) + "string")


def test_print_2_2a(encoder: str):
    print(f"Testing {encoder} encoder")

    test_string = "hello! こんにちは!"
    
    utf8_encoded = test_string.encode(encoder)

    print(utf8_encoded)

    print(type(utf8_encoded))

    list(utf8_encoded)

    print(len(test_string))

    print(len(utf8_encoded))

    print(utf8_encoded.decode(encoder))

def test_print_2_2b():
    def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
        return "".join([bytes([b]).decode("utf-8") for b in bytestring])

    test_string = "hello! こんにちは!"
    try:  
        print(decode_utf8_bytes_to_str_wrong(test_string.encode("utf-8")))
    except Exception as e:
        print('error: ', e)


# test_print_2_1()

# test_print_2_2a("utf-8")
# test_print_2_2a("utf-16")
# test_print_2_2a("utf-32")

# test_print_2_2b()


from cs336_basics.BPEtrainer import train_bpe

from cs336_basics.BPEencoder import Tokenizer
## Usage
if __name__ == "__main__":
    ## Usage
    # input_path = "/Users/yuhangfang/Documents/learning/LLMfromScratch/assignment1-basics-main/cs336_basics/data/TinyStoriesV2-GPT4-valid.txt"
    # vocab_size = 1000
    special_tokens = ["<|endoftext|>"]

    # train_bpe(input_path, vocab_size, special_tokens)

    vocab_filepath = "/Users/yuhangfang/Documents/learning/LLMfromScratch/assignment1-basics-main/cs336_basics/data/vocab.json"
    merges_filepath = "/Users/yuhangfang/Documents/learning/LLMfromScratch/assignment1-basics-main/cs336_basics/data/merges.json"

    tokener = Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens)
    tokenized = tokener.encode("I am a boy.<|endoftext|> She is a girl.")
    print('tokenized', tokenized)

    detokenized = tokener.decode(tokenized)
    print('detokenized', detokenized)

