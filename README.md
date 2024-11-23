### This is an implementation of the GPT2 124M model, optimized for concurrent training on multiple CUDA GPUs.
The model is written in low-level PyTorch API, with custom classes extending nn.Module. Uses FlashAttention, mixed precision and other optimizations for efficient training. 

Implements ~0.5M (!) batch size thanks to gradient accumulation for efficient large-scale training.

Based on Andrej Karpathy's style of [GPT2 implementation](https://www.youtube.com/watch?v=l8pRSuU81PU), with additional fine-tuning and generation modules built on top.