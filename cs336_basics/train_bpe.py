
import heapq
import queue
import time
import json
import threading
import regex as re


class PairItem:
    def __init__(self, cnt, subword1, subword2):
        self.cnt = cnt
        self.subword1 = subword1
        self.subword2 = subword2
    
    def __lt__(self, other):
        if self.cnt != other.cnt:
            return self.cnt > other.cnt
        if self.subword1 != other.subword1:
            return self.subword1 > other.subword1

class TrainBPE:
    def __init__(self):
        pass
    
    
    def pre_tokenize(
        self,
        data_path,
        num_worker: int = 20,
        max_size: int = 1000,
        output_path: str = None,
        split_token: str = "<|endoftext|>",
        special_tokens: list[str] = [],
        ) -> dict:
        task_queue = queue.Queue(maxsize=num_worker*2)
        result_queue = queue.Queue()
        # special_tokens.append(split_token)
        
        self.done_flag = False
        
        def producer(data_path, task_queue, max_size):
            try:
                input_f = open(data_path, 'r')
                
                batch = ""
                for line in input_f:
                    # if len(line) + len(batch) > max_size:
                    if split_token in line or len(line) + len(batch) > max_size:
                        # for spe_token in special_tokens:
                        #     batch = batch.replace(spe_token, " ")
                        task_queue.put(batch)
                        batch = line
                    else:
                        batch = batch + line
                
                if len(batch) > 0:
                    for spe_token in special_tokens:
                        batch = batch.replace(spe_token, " ")
                    task_queue.put(batch)
                
                for _ in range(num_worker):
                    task_queue.put(None)
            except Exception as e:
                print(e)
                for _ in range(num_worker):
                    task_queue.put(None)
        
        def worker(task_queue, result_queue):
            while True:
                try:
                    text = task_queue.get(timeout=1)
                    if text is None:
                        self.done_flag = True
                        break
                    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
                    texts = re.split('|'.join(re.escape(t) for t in special_tokens), text)
                    for text in texts:
                        words = re.findall(PAT, text)
                        res = {}
                        for word in words:
                            # word = word.replace(' ', '')
                            if res.get(word) is None:
                                res[word] = 1
                            else:
                                res[word] += 1
                    result_queue.put(res)
                except Exception as e:
                    print(e)
                    pass

        threads = []
        t = threading.Thread(target=producer, args=(data_path, task_queue, max_size))
        threads.append(t)
        t.start()
        
        for _ in range(num_worker):
            t = threading.Thread(target=worker, args=(task_queue, result_queue))
            threads.append(t)
            t.start()
        
        start_time = time.time()
        while not self.done_flag:
            cnt = result_queue.qsize()
            # print(f"cost time {time.time() - start_time}, finished {cnt}")
            time.sleep(1)
        
        res = {}
        while not result_queue.empty():
            res_tmp = result_queue.get()
            for k, v in res_tmp.items():
                if res.get(k) is None:
                    res[k] = v
                else:
                    res[k] += v
        
        if output_path is not None:
            with open(output_path, 'w') as f:
                f.write(json.dumps(res, ensure_ascii=False, indent=4))
        return res
        
    def train_bpe(
        self,
        input_path: str,
        vocab_size: int,
        special_tokens: list[str],
        words_cnt: dict = None,
        ) -> (dict, list):
        
        if words_cnt is None:
            words_cnt = self.pre_tokenize(input_path,
                                        num_worker=20,
                                        max_size=5000,
                                        output_path=None,
                                        split_token="<|endoftext|>",
                                        special_tokens=special_tokens)
        # print(f"words_cnt : {words_cnt}")
        vocab = {}
        vocab_r = {}
        for i in range(0, 256):
            val = bytes([i])
            ID = len(vocab)
            vocab[val] = ID
            vocab_r[ID] = val
            
        for token in special_tokens:
            val = token.encode('utf-8')
            ID = len(vocab)
            vocab[val] = ID
            vocab_r[ID] = val
        
        
        
        pair_map = {}
        split_map = {}
        for word, cnt in words_cnt.items():
            # word = word.replace(' ', '')
            word_list = list(word.encode('utf-8'))
            split_map[word] = [bytes([ii]) for ii in word_list]
            # print(f"word_list : {word_list}")
            for sub1, sub2 in zip(word_list[:-1], word_list[1:]):
                ID1 = vocab[bytes([sub1])]
                ID2 = vocab[bytes([sub2])]
                pair_map[(ID1, ID2)] = pair_map.get((ID1, ID2), 0) + cnt

        merges = []
        # print(f"pair_map : {pair_map}")
        while len(vocab) < vocab_size and len(pair_map) > 0:
            # print(f"pair_map : {pair_map}")
            sub1, sub2, cnt = None, None, -1
            for key, value in pair_map.items():
                s1, s2 = vocab_r[key[0]], vocab_r[key[1]]
                if value > cnt:
                    sub1, sub2, cnt = s1, s2, value
                elif value == cnt and s1 > sub1:
                    sub1, sub2, cnt = s1, s2, value
                elif value == cnt and s1 == sub1 and s2 > sub2:
                    sub1, sub2, cnt = s1, s2, value
            
            assert sub1 is not None
            # print(f"best pair {sub1} {sub2}")
            merges.append((sub1, sub2))
            
            ID = len(vocab)
            new_sub = sub1 + sub2
            vocab[new_sub] = ID
            vocab_r[ID] = new_sub
            
            
            for word, num in words_cnt.items():
                if sub1 in split_map[word] and sub2 in split_map[word]:
                    split_list = []
                    old_split_list = split_map[word]
                    for item in old_split_list:
                        if len(split_list) == 0:
                            split_list.append(item)
                        else:
                            if split_list[-1] == sub1 and item == sub2:
                                split_list[-1] = split_list[-1] + item
                                pair_map[(vocab[sub1], vocab[sub2])] -= num
                            else:
                                split_list.append(item)
                    
                    split_map[word] = split_list
                    for t1, t2 in zip(split_map[word][:-1], split_map[word][1:]):
                        if t1 == new_sub and t2 == new_sub:
                            pair_map[(vocab[sub2], vocab[sub1])] -= num
                            pair_map[(ID, ID)] = pair_map.get((ID, ID), 0) + num
                        elif t2 == new_sub:
                            pair_map[(vocab[t1], vocab[sub1])] -= num
                            pair_map[(vocab[t1], ID)] = pair_map.get((vocab[t1], ID), 0) + num
                        elif t1 == new_sub:
                            pair_map[(vocab[sub2], vocab[t2])] -= num
                            pair_map[(ID, vocab[t2])] = pair_map.get((ID, vocab[t2]), 0) + num
                            
        # print(f"pair_map : {pair_map}")
        
        vocab_ret = {value: key for key, value in vocab.items()}
        return vocab_ret, merges

def test():
    train_bpe = TrainBPE()
    # cnt = train_bpe.pre_tokenize("data/test.txt")
    # print(cnt)
    cnt = {
        "aaaaa" : 1,
    }
    vocab, merges = train_bpe.train_bpe("", 260, ["<|endoftext|>"], cnt)
    print(f"vocab : {vocab}")
    print(f"merges : {merges}")
    
    
    
if __name__ == '__main__':
    test()
    
    