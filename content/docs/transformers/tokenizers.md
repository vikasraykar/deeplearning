---
title: Tokenizers
weight: 1
bookToc: true
---


```python
class Tokenizer:
    """Tokenizer"""

    def encode(self, string: str) -> list[int]:
        """Convert a string to a sequence of integers (tokens)."""
        raise NotImplementedError

    def decode(self, indices: list[int]) -> str:
        """Convert a sequence of integers (tokens) to a string."""
        raise NotImplementedError
```

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

Byte-Pair Encoding (BPE)

{{% hint info %}}
[Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909), Rico Sennrich, Barry Haddow, Alexandra Birch, ACL 2016.
{{% /hint %}}




## WordPiece

> BERT

## Unigram

> T5

## Tokenizer-free approaches

Use bytes directly, promising, but have not yet been scaled up to the frontier.

https://arxiv.org/abs/2105.13626

https://arxiv.org/pdf/2305.07185

https://arxiv.org/abs/2412.09871

https://arxiv.org/abs/2406.19223
