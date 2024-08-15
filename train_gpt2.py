from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from torch.nn import functional as F


#--------------------------------------------------------
import tiktoken


class DataLoader:

    def __init__(self, B, T):
        self.B = B
        self.T = T

        # Load tokens from disc at object init
        # Encode 
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"Loaded {len(tokens)} tokens")
        print(f"1 epoch = {len(tokens) // (B * T)} batches")

        # Keeping state
        self.current_position = 0


    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = (buf[:-1].view(B, T))
        y = (buf[1:].view(B, T))
        self.current_position += B * T
        # If current_position would be out of bounds on next call, reset to 0
        if (self.current_position + B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y
#--------------------------------------------------------


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        # multiheaded self-attention in one tensor: queries, keys and values
        # This is the same as passing the input through 3 Linear tensors to get q, k, v
        # Here we are doing it in one batch
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.n_embed = config.n_embed
        self.n_head = config.n_head
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
        

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embed, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        att = (q @ k.transpose(-1, -2)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)


    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)


    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed),
            wpe = nn.Embedding(config.block_size, config.n_embed),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embed)
        ))

        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

        # Tying weights of LM Head and WTE, weight sharing
        self.transformer.wte.weight = self.lm_head.weight

        # Init the weights
        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)

    
    def forward(self, idx, targets=None):
        # idx shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # shape (T, n_embed)
        tok_emb = self.transformer.wte(idx) # shape (B, T, n_embed)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # shape (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    
    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("Loading weights from pretrained gpt: %s" % model_type)

        # Create the config_args dict
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embed=768), # 124M params
            'gpt-medium': dict(n_layer=24, n_head=16, n_embed=1024), # 350M params
            'gpt-large': dict(n_layer=36, n_head=20, n_embed=1280), # 774M params
            'gpt-xl': dict(n_layer=48, n_head=25, n_embed=1600) # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024

        # Create our local GPT model
        config = GPTConfig(**config_args)
        model = GPT(config=config)

        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        # Instantiate a huggingface model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['.attn.c_attn.weight', '.attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        
        assert len(sd_keys_hf) == len(sd_keys), f"Mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

#--------------------------------------------------------------
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
print(f"Using device: {device}")

# Get a data batch
import tiktoken
enc = tiktoken.get_encoding('gpt2')
with open('input.txt', 'r') as f:
    text = f.read()
data = text[:1000]
tokens = enc.encode(data)
B, T = 4, 32
buf = torch.tensor(tokens[:B * T + 1], device=device)
x = buf[:-1].view(B, T)
y = buf[1:].view(B, T)

# Different model inits
# from transformers import GPT2LMHeadModel
# model = GPT2LMHeadModel.from_pretrained('gpt2')
# model = GPT.from_pretrained('gpt2')

# Init a fresh model
model = GPT(GPTConfig())
model.eval()
model.to(device)

# Optimizer and data loader
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
data_loader = DataLoader(B=4, T=32)

for i in range(50):
    x, y = data_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    print(f"For step {i}, loss: {loss.item()}")


# Inspecting the accuracy
#
# logits = logits[:, -1, :]
# probs = F.softmax(logits, dim=-1)
# print("x", x)
# print("y", y)
# print("Probs", probs)
# # Gettting the top 50 probs and top 50 indices, both shape (5, 50)
# topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
# print("Topk probs", topk_probs)
# print("Topk indices", topk_indices)
# # # Selecting one index from the top 50, shape (5, 1)
# ix = torch.multinomial(topk_probs, 1)
# print("Selected idx for each pos", ix)
# # # Get the corresponding indices that were selected
# xcol = torch.gather(topk_indices, -1, ix)
# print("Idx col to append", xcol)
# # # Append the new indices to the input along dim=1
# # x = torch.cat((x, xcol), 1)

# num_return_sequences = 5
# max_length = 30

# for i in range(num_return_sequences):
#     tokens = x[i, :max_length].tolist()
#     decoded = enc.decode(tokens)
#     print(">", decoded)

import sys; sys.exit(0)

#--------------------------------------------------------------
num_return_sequences = 5
max_length = 30

import tiktoken
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("Latest breakthrough in quantum physics:")
tokens = torch.tensor(tokens, dtype=torch.long) # shape (8)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
x = tokens.to(device)

# Generating. Shape of x is (B, T) = (5, 8)
torch.manual_seed(42)
torch.mps.manual_seed(42)
while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x)
        logits = logits[:, -1, :]
        # Converting logits to probs
        probs = F.softmax(logits, dim=-1)
        # Gettting the top 50 probs and top 50 indices, both shape (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # Selecting one index from the top 50, shape (5, 1)
        ix = torch.multinomial(topk_probs, 1)
        # Get the corresponding indices that were selected
        xcol = torch.gather(topk_indices, -1, ix)
        # Append the new indices to the input along dim=1
        x = torch.cat((x, xcol), 1)


for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)