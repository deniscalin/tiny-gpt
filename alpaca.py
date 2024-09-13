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
# Wrap each example in prompts, get a list of outputs + eot, create a dataset with dictionaries containing p, o, p+0 for each example
prompts = [create_prompt(row) for row in alpaca]
eot = '<|endoftext|>'
outputs = [row['output'] + eot for row in alpaca]

# Tokenize the dataset with instructions masking in the labels
dataset = [{'x': enc.encode(p + o, allowed_special={'<|endoftext|>'}), 'y': ([-100] * len(enc.encode(p, allowed_special={'<|endoftext|>'}))) + enc.encode(o, allowed_special={'<|endoftext|>'})} for p, o in zip(prompts, outputs)]


print("First example", dataset[0])
print('First example decoded: ', enc.decode(dataset[0]['x']))
print("Last example", dataset[-1])
print('Last example decoded: ', enc.decode(dataset[-1]['x']))


numpy_dataset = [{'x': wrap_in_numpy(s['x']), 'y': wrap_in_numpy(s['y'])} for s in dataset]

print("First example, numpy_dataset", numpy_dataset[0])
print("First example, len of x", len(numpy_dataset[0]['x']))
print("First example, len of y", len(numpy_dataset[0]['y']))
print("Last example, numpy_dataset", numpy_dataset[-1])
print("Last example, len of x", len(numpy_dataset[-1]['x']))
print("Last example, len of y", len(numpy_dataset[-1]['y']))

write_file('alpaca_set', numpy_dataset)