---
title: Inference
weight: 4
bookToc: true
---

Prefill and decode.


Prefill (similar to training): tokens are given, can process all at once (compute-bound)

Decode: need to generate one token at a time (memory-bound)

## Fast inference

### Systems optimization

KV caching

Batching

## CHeaper models

model pruning

quantization

distillation

Speculative decoding: use a cheaper "draft" model to generate multiple tokens, then use the full model to score in parallel (exact decoding).


