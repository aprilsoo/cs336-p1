
import heapq
import queue
import time
import json
import threading
import regex as re
from collections.abc import Iterable, Iterator

class BPETokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = {value: key for key, value in vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens is not None else []
        self.special_tokens = sorted(self.special_tokens, key=lambda x : len(x), reverse=True)
        
        for spe_token in self.special_tokens:
            if self.vocab.get(spe_token.encode('utf-8')) is None:
                self.vocab[spe_token.encode('utf-8')] = len(self.vocab)
        
        self.vocab_r = {value: key for key, value in self.vocab.items()}
        
    
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        pass
    
    def encode(self, text: str) -> list[int]:
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        
        origin_text = text
        encode_list = []
        if len(self.special_tokens) > 0:
            texts = re.split('|'.join(re.escape(t) for t in self.special_tokens), text)
        else:
            texts = [text]
        for text in texts:
            words = re.findall(PAT, text)
            for word in words:
                word = list(word.encode('utf-8'))
                for idx in range(0, len(word)):
                    word[idx] = self.vocab[bytes([word[idx]])]
                for subs in self.merges:
                    sub1, sub2 = self.vocab[subs[0]], self.vocab[subs[1]]
                    if sub1 not in word or sub2 not in word:
                        continue
                    new_word = []
                    for item in word:
                        if len(new_word) > 0 and new_word[-1] == sub1 and item == sub2:
                            new_word[-1] = self.vocab[subs[0] + subs[1]]
                        else:
                            new_word.append(item)
                    word = new_word
                
                encode_list.extend(word)
            
            origin_text = origin_text[len(text):]
            for spe_token in self.special_tokens:
                if origin_text.startswith(spe_token):
                    origin_text = origin_text[len(spe_token):]
                    encode_list.append(self.vocab[spe_token.encode('utf-8')])
                    break
        
        return encode_list
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for i in iterable:
            for j in self.encode(i):
                yield j
    
    def decode(self, ids: list[int]) -> str:
        decode_list = []
        for i in ids:
            decode_list.extend(list(self.vocab_r[i]))
        decode_str = bytes(decode_list).decode('utf-8', errors='replace')
        return decode_str


if __name__ == '__main__':
    vocab = {
        "a": 0,
        "b": 1,
        "ab": 2,
    }
    
    merges = [("a".encode('utf-8')), ("b".encode('utf-8'))]
    tokenizer = BPETokenizer(vocab, merges)