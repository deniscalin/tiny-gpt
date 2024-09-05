from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import inspect
# For use on Linux, check if this might work TODO: check how to modify the model to use this
# from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss 
from hellaswag import render_example, iterate_examples
#--------------------------------------------------------
import tiktoken
import numpy as np

import os
import time
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoader:

    def __init__(self, B, T, process_rank, total_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.total_processes = total_processes
        assert split in {'train', 'val'}

        # Get the list of shards from the dir, turn into full filenames
        data_root = 'edu_fineweb10B'
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f'Found no shards for split: {split}'
        if master_process:
            print(f"Found {len(shards)} shards for split: {split}")
        self.reset()

    
    def reset(self):
        # Keeping state
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank


    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = (buf[:-1].view(B, T))
        y = (buf[1:].view(B, T))
        self.current_position += B * T * self.total_processes
        # If current_position would be out of bounds on next call, reset to 0
        if self.current_position + (B * T * self.total_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank
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
        self.c_proj.NANOGPT_SCALE_INIT = 1 # A flag to scale the residual layer init to 1/sqrt(n)
        self.n_embed = config.n_embed
        self.n_head = config.n_head
        # No longer need the regiester_buffer for the attention mask since we switched to FlashAttention
        # self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size))
        #                     .view(1, 1, config.block_size, config.block_size))
        

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embed, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # Replace attention with FlashAttention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # att = (q @ k.transpose(-1, -2)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)
        self.c_proj.NANOGPT_SCALE_INIT = 1 # A flag to scale the residual layer init to 1/sqrt(n)


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
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)

    
    def configure_optimizers(self, weight_decay, learning_rate, device):
        # Create a param dict with all parameters that have gradients
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # Create optim_groups from 2 lists: one with 2D and above matmul tensors 
        # (e.g. Linear and Embeddings tensors) to be weight decayed
        # Another one with 1D tensors like LayerNorm, biases that won't be weight decayed. 
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with total {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with total {num_nodecay_params:,} parameters")
        # If fused if available, and we are running on cuda, use it with AdamW
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"Use fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, 
                                      betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
    

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
            # Try using the Liger Kernel: FusedCrossEntropy
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
    

# Set up DDP
# The torchrun command will get env variables like RANK, LOCAL_RANK and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 #  checking if RANK is set, and therefore this is a ddp run

if ddp:
    assert torch.cuda.is_available()
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    autocast_device = 'cuda'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # The master process does logging, checkpointing, etc
else:
    # Non-ddp run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
        autocast_device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        autocast_device = 'cpu' # Per https://github.com/karpathy/build-nanogpt errata
    print(f"Using device: {device}")

# Set global and generation seeds for RNGs
global_seed = 1337
gen_seed = 42

torch.manual_seed(global_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(global_seed)
elif torch.backends.mps.is_available():
    torch.mps.manual_seed(global_seed)

# Instantiate the model
model = GPT(GPTConfig(vocab_size=50304))
sd = torch.load('log/model_00500.pt', map_location=torch.device('mps'))
print("Loaded model sd")
print("Config: ", sd['config'])
print("Step: ", sd['step'])
print("Val loss: ", sd['val_loss'])
print("Loaded optimizer sd")
print("Global seed: ", sd['global_seed'])
print("Gen seed: ", sd['gen_seed'])

model.load_state_dict(sd['model'], strict=True)
model.eval()

import sys; sys.exit(0)

