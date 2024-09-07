import json


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


prompts = [create_prompt(row) for row in alpaca]

eot = '<|endoftext|>'

outputs = [row['output'] + eot for row in alpaca]

dataset = [{'prompt': p, 'output': o, 'example': p+o} for p, o in zip(prompts, outputs)]

print(dataset[0])
# print(outputs[:10])

#print(prompt_input(alpaca[232]))
# print(len(alpaca))
#print(alpaca[0])
# print(alpaca[-1])

# print("This is an {adj} string".format_map(dict(adj='awesome')))
