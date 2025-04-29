---
title: Tokenizers
weight: 2
bookToc: true
---

Tokenization pipeline
- Normalization
- Pre-tokenization
- Model
- Post-processor

## SentencePiece

https://github.com/google/sentencepiece

SentencePiece is a tokenization algorithm for the preprocessing of text that you can use with either BPE, WordPiece, or Unigram model.
- It considers the text as a sequence of Unicode characters, and replaces spaces with a special character, `▁`.
- Used in conjunction with the Unigram algorithm, it doesn’t require a pre-tokenization step, which is very useful for languages where the space character is not used (like Chinese or Japanese).
- SentencePiece is **reversible tokenization**: since there is no special treatment of spaces, decoding the tokens is done simply by concatenating them and replacing the `_`s with spaces — this results in the normalized text.

## BPE

> GPT-2

## WordPiece

> BERT

## Unigram

> T5
