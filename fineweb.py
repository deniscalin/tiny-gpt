import os
import numpy as np
import multiprocessing as mp
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"
shard_size = int(1e8) # 100M tokens per shard, 100 shards total

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split='train')

# Initialize the tokenizer and get the end of text token
enc = tiktoken.get_encoding('gpt2')
eot = enc._special_tokens('<|endoftext|>')

def tokenize(doc):
    """Tokenizes a single document, and returns a Numpy array of tokens in uint16"""
    tokens = [eot] # Start with the end of text token
    tokens.extend(enc.encode_ordinary(doc['text']))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens.astype(np.uint16)
    return tokens_np_uint16