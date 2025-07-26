from collections.abc import Iterator
import json

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str]=None):
        self.vocab = vocab.copy()  # dict[int, bytes]
        self.merges = merges.copy()  # list[tuple[bytes, bytes]]
        
        # Handle special tokens
        if special_tokens:
            self._add_special_tokens(special_tokens)
        
        # Create reverse mapping for encoding
        self.token_to_id = {token: id for id, token in self.vocab.items()}
    
    def _add_special_tokens(self, special_tokens):
        """Add special tokens to vocab if not already present"""
        next_id = max(self.vocab.keys()) + 1 if self.vocab else 0
        
        for token in special_tokens:
            token_bytes = token.encode('utf-8')
            if token_bytes not in self.token_to_id.values():
                self.vocab[next_id] = token_bytes
                next_id += 1

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str]=None):
        """Load tokenizer from saved vocab and merges files"""
        # Load vocab
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            vocab_json = json.load(f)
            vocab = {int(k): v.encode('utf-8') for k, v in vocab_json.items()}
        
        # Load merges  
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            merges_json = json.load(f)
            merges = [(m[0].encode('utf-8'), m[1].encode('utf-8')) for m in merges_json]
        
        # Create and return tokenizer instance using __init__
        return cls(vocab, merges, special_tokens)
    
    
    def encode(self, text: str) -> list[int]:

        tokens = []
        i = 0
        n = len(text)
        current_token = ""
        current_token_id = None

        while i<n:
            current_token += text[i]
            if current_token in self.token_to_id:            
                current_token_id = self.token_to_id[current_token]
            else:
                tokens.append(current_token_id)
                current_token = text[i]
                current_token_id = self.token_to_id[current_token]

            i += 1

        if current_token_id is not None:
            tokens.append(current_token_id)

        return tokens


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]
    # Given an iterable of
    # strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is
    # required for memory-eﬀicient tokenization of large files that we cannot directly load into
    # memory.

    def decode(self, ids: list[int]) -> str:
        return bpe_decoder(self.vocab, self.merges, tokens)
        
        