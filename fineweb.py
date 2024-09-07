import os
import numpy as np
import multiprocessing as mp
import tiktoken
from datasets import load_dataset, load_from_disk
from tqdm import tqdm

# NOTE: think about permuting the data, namely the order of documents in the dataset. Perhaps each epoch can be permuted differently and shuffled randomly.

local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"
shard_size = int(1e8) # 100M tokens per shard, 100 shards total

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
DATASET_FILE_PATH = 'fineweb-edu-dataset'
os.makedirs(DATA_CACHE_DIR, exist_ok=True)


fw = load_from_disk(DATASET_FILE_PATH)

# Initialize the tokenizer and get the end of text token
enc = tiktoken.get_encoding('gpt2')
eot = enc._special_tokens['<|endoftext|>']

def tokenize(doc):
    """Tokenizes a single document, and returns a Numpy array of tokens in uint16"""
    tokens = [eot] # Start with the end of text token
    tokens.extend(enc.encode_ordinary(doc['text']))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def write_file(filename, tokens_np):
    """Writes the Numpy array of uint16 tokens to a local file"""
    np.save(filename, tokens_np)

if __name__ == '__main__':
    # Tokenize all docs and save them into shards, each one the size of shard_size
    nprocs = max(1, os.cpu_count() // 2)
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        # Preallocating buffer to hold current shard
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None
        for tokens in pool.imap(tokenize, fw, chunksize=16):
            # Check if there is enough space in the current shard for the new tokens
            if token_count + len(tokens) < shard_size:
                # Append tokens to the current shard
                all_tokens_np[token_count:token_count+len(tokens)] = tokens
                token_count += len(tokens)
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit='tokens', desc=f'Shard {shard_index}')
                progress_bar.update(len(tokens))
            else:
                split = 'val' if shard_index == 0 else 'train'
                filename = os.path.join(DATA_CACHE_DIR, f'fineweb-edu_{split}_{shard_index:06d}')
                remainder = shard_size - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
                write_file(filename, all_tokens_np)
                shard_index += 1
                progress_bar = None
                # Start populating the next shard with the remainder of the current doc
                all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
                token_count = len(tokens) - remainder

        # Writing any remaining tokens into the last shard
        if token_count != 0:
            split = 'val' if shard_index == 0 else 'train'
            filename = os.path.join(DATA_CACHE_DIR, f'fineweb-edu_{split}_{shard_index:06d}')
            write_file(filename, all_tokens_np[:token_count])