import json
import os
import numpy as np
import multiprocessing as mp
import tiktoken
from datasets import load_dataset, load_from_disk
from tqdm import tqdm


###############################
# Exploring the dataset
with open('alpaca_gpt4_data.json', 'r') as f:
    alpaca = json.loads(f.read())


# Dataset processing functions to wrap data into prompts
def prompt_no_input(row):
    return ("Below is an instruction that describes a task. "
           "Write a response that appropriately completes the request.\n\n"
           "### Instruction:\n{instruction}\n\n### Response:\n").format_map(row)


def prompt_input(row):
    return ("Below is an instruction that describes a task, paired with an input that provides further context. "
           "Write a response that appropriately completes the request.\n\n"
           "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n").format_map(row)


def create_prompt(row):
    return prompt_no_input(row) if row['input'] == '' else prompt_input(row)


def wrap_in_numpy(tokens):
    """Wraps the tokens in a Numpy array in uint16"""
    # tokens = [] # Start with an empty list
    # tokens.extend(enc.encode(doc['text'], allowed_special={'<|endoftext|>'}))
    tokens_np = np.array(tokens)
    # assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16


def write_file(filename, tokens_np):
    """Writes the Numpy array of uint16 tokens to a local file"""
    np.save(filename, tokens_np)



# Initialize the tokenizer and get the end of text token id
enc = tiktoken.get_encoding('gpt2')
eot_id = enc._special_tokens['<|endoftext|>']

# import sys; sys.exit(0)
# Wrap each example in prompts, get a list of outputs + eot, create a dataset with dictionaries containing p, o, p+0 for each example
prompts = [create_prompt(row) for row in alpaca]
eot = '<|endoftext|>'
outputs = [row['output'] + eot for row in alpaca]

# Dataset without instructions masking in the y
# dataset = [{'prompt': p, 'output': o, 'example': p+o} for p, o in zip(prompts, outputs)]

# Tokenize the dataset with instructions masking in the labels
dataset = [{'x': enc.encode(p + o, allowed_special={'<|endoftext|>'}), 'y': ([-100] * len(enc.encode(p, allowed_special={'<|endoftext|>'}))) + enc.encode(o, allowed_special={'<|endoftext|>'})} for p, o in zip(prompts, outputs)]
# dataset_examples = [s['example'] for s in dataset]

print("First example", dataset[0])
print('First example decoded: ', enc.decode(dataset[0]['x']))
print("Last example", dataset[-1])
print('Last example decoded: ', enc.decode(dataset[-1]['x']))

# print("First example", dataset[0]['example'])
# print('------------')
# print("First example: instructions + output", dataset[0]['prompt'] + dataset[0]['output'])
# print('------------')
# print("First example: instructions replaced with [-100] + output", str([-100]) * len(dataset[0]['prompt']) + dataset[0]['output'])
# print('------------')
# print("First example, encoded with instruction masking: ", [-100] * len(dataset[0]['prompt']) + enc.encode(dataset[0]['output'], allowed_special={'<|endoftext|>'}))

# print("First example", dataset_examples[0])
# print("Encoded: ", enc.encode(dataset_examples[0], allowed_special={'<|endoftext|>'}))
# print('------------')
# print(dataset_examples[1])
# print('------------')
# print(dataset_examples[2])
# print('------------')
# print(f"Lenght of dataset: {len(dataset_examples)}")

numpy_dataset = [{'x': wrap_in_numpy(s['x']), 'y': wrap_in_numpy(s['y'])} for s in dataset]

print("First example, numpy_dataset", numpy_dataset[0])
print("First example, len of x", len(numpy_dataset[0]['x']))
print("First example, len of y", len(numpy_dataset[0]['y']))
# print('First example decoded: ', enc.decode(dataset[0]['x']))
print("Last example, numpy_dataset", numpy_dataset[-1])
print("Last example, len of x", len(numpy_dataset[-1]['x']))
print("Last example, len of y", len(numpy_dataset[-1]['y']))
# print('Last example decoded: ', enc.decode(dataset[-1]['x']))

write_file('alpaca_set', numpy_dataset)

import sys; sys.exit(0)

# if __name__ == '__main__':
#     # Tokenize all docs and save them into shards, each one the size of shard_size
#     nprocs = max(1, os.cpu_count() // 2)
#     with mp.Pool(nprocs) as pool:
#         # Preallocating buffer to hold current shard
#         all_tokens_np = np.empty((len(dataset),), dtype=np.uint16)
#         token_count = 0
#         progress_bar = None
#         for tokens in pool.imap(tokenize, fw, chunksize=16):
#             # Check if there is enough space in the current shard for the new tokens
#             token_count + len(tokens) < shard_size
#             # Append tokens to the current shard
#             all_tokens_np[token_count:token_count+len(tokens)] = tokens
#             token_count += len(tokens)
#             if progress_bar is None:
#                 progress_bar = tqdm(total=shard_size, unit='tokens', desc=f'Shard {shard_index}')
#             progress_bar.update(len(tokens))
            # else:
            #     split = 'val' if shard_index == 0 else 'train'
            #     filename = os.path.join(DATA_CACHE_DIR, f'fineweb-edu_{split}_{shard_index:06d}')
            #     remainder = shard_size - token_count
            #     progress_bar.update(remainder)
            #     all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
            #     write_file(filename, all_tokens_np)
            #     shard_index += 1
            #     progress_bar = None
            #     # Start populating the next shard with the remainder of the current doc
            #     all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
            #     token_count = len(tokens) - remainder


# print(outputs[:10])

#print(prompt_input(alpaca[232]))
# print(len(alpaca))
#print(alpaca[0])
# print(alpaca[-1])

# print("This is an {adj} string".format_map(dict(adj='awesome')))
