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
    

# -----------------------------------------------------------------------------
# COPIED
# helper function for HellaSwag eval
# takes tokens, mask, and logits, returns the index of the completion with the lowest loss

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm

# COPIED
#---------------------------------------------------------------------------
# TO LAUNCH:
# Simple run: python train_gpt2_cuda.py
# DDP Run: torchrun --standalone --nproc_per_node=8 train_gpt2_cuda.py

import os
import time
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

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

### SETTING BATCH SIZE
total_batch_size = 524288 # total batch size to process for each update step (~0.5M per GPT3 paper)
B = 64
T = 1024
assert total_batch_size % (B * T * ddp_world_size) == 0, 'make sure total_batch_size is divisible by B * T * ddp_world_size'
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print("Total desired batch size: ", total_batch_size)
    print("Calculated gradient accum steps: ", grad_accum_steps)

# Data batch practice
# # Get a data batch
# import tiktoken
# enc = tiktoken.get_encoding('gpt2')
# with open('input.txt', 'r') as f:
#     text = f.read()
# data = text[:1000]
# tokens = enc.encode(data)
# B, T = 4, 32
# buf = torch.tensor(tokens[:B * T + 1], device=device)
# x = buf[:-1].view(B, T)
# y = buf[1:].view(B, T)

# Different model inits
# from transformers import GPT2LMHeadModel
# model = GPT2LMHeadModel.from_pretrained('gpt2')
# model = GPT.from_pretrained('gpt2')

# Data loader
train_loader = DataLoader(B=B, T=T, process_rank=ddp_rank, total_processes=ddp_world_size, split='train')
val_loader = DataLoader(B=B, T=T, process_rank=ddp_rank, total_processes=ddp_world_size, split='val')

torch.set_float32_matmul_precision('high')

# Init a fresh model
model = GPT(GPTConfig(vocab_size=50304))
# Or evaluate a pre-trained model
# model = GPT(GPTConfig(vocab_size=50304)).from_pretrained('gpt2')
model.to(device)
use_compile = False
if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

max_lr = 6e-4 # try 3x -> 1.8e-3
min_lr = max_lr * 0.1
warmup_steps = 715 # Try 27 (375e6 / 26 / 2**19)
max_steps = 19073 # try 4x -> 76,292

def get_lr(it):
    # Linear lr warmup to quickly ramp up to max_lr
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    # After max steps, use min_lr
    if it > max_steps:
        return min_lr
    # In between, use cosine decay to gradually decay from max_lr to min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


# Optimizer
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)


# Create a dir for logging and saving checkpoints
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'log.txt')
with open(log_file, 'w') as f: # open the file in w mode to clear it
    pass

# Get tiktoken encoder for generation
enc = tiktoken.get_encoding('gpt2')

if master_process:
    print(f"Using device: {device}")

for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # Evaluate the validation loss every 100 steps
    if step % 250 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_eval_steps = 20
            for _ in range(val_eval_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=autocast_device, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_eval_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"Validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, 'a') as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            if (step > 0) and (step % 5000 == 0 or last_step):
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "config": raw_model.config,
                    "step": step,
                    "val_loss": val_loss_accum.item(),
                    "optimizer": optimizer.state_dict(),
                    "global_seed": global_seed,
                    "gen_seed": gen_seed
                }
                torch.save(checkpoint, checkpoint_path)

    # once in a while evaluate hellaswag - COPIED
    if (step % 250 == 0 or last_step) and (not use_compile):
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # only process examples where i % ddp_world_size == ddp_rank
            if i % ddp_world_size != ddp_rank:
                continue
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # get the logits
            with torch.no_grad():
                with torch.autocast(device_type=autocast_device, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # reduce the stats across all processes
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")

    # Every 250th step, except for step 0, also generate from the model (if torch.compile is not being used -- bug)
    if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        x_gen = tokens.to(device=device)
        seed_gen = torch.Generator(device=device)
        seed_gen.manual_seed(gen_seed + ddp_rank)
        while x_gen.size(1) < max_length:
            with torch.no_grad():
                logits, loss = model(x_gen)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, -1) # (B, 1, vocab_size)
                topk_probs, topk_indices = torch.topk(probs, 50, -1) # Returns the 50 highest probabilities and their indices from probs for each B -> (4, 50)
                ix = torch.multinomial(topk_probs, 1, generator=seed_gen) # Samples 1 prob from the topk_probs distribution and returns its index -> (B, 1)
                x_gen_col = torch.gather(topk_indices, -1, ix) # (B, 1)
                x_gen = torch.cat((x_gen, x_gen_col), 1) # (B, T+1)
        for i in range(num_return_sequences):
            tokens = x_gen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"Rank {ddp_rank} sample {i}: {decoded}")

    # The training loop
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=autocast_device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        # import code; code.interact(local=locals())
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) # Time delta in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_throughput = tokens_processed / dt
    if master_process:
        print(f"For step {step:4d}: loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tokens/sec: {tokens_throughput}")
        with open(log_file, 'a') as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

if ddp:
    destroy_process_group()
import sys; sys.exit(0)

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
