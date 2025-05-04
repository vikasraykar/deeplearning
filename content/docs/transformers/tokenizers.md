---
title: Tokenizers
weight: 1
bookToc: true
---

## Tokenizers

A tokenizer converts text (string) to a sequence of tokens (represented as list of integer indices).

A Tokenizer is a class that implements the `encode` and `decode `methods.

The **vocabulary size** is number of possible tokens (integers).

> Explore various tokenizers here. https://tiktokenizer.vercel.app/

```python
from abc import ABC

class Tokenizer(ABC):
    """Abstract interface for a tokenizer."""

    def encode(self, string: str) -> list[int]:
        """Convert a string to a sequence of integer indices (token ids)."""
        raise NotImplementedError

    def decode(self, indices: list[int]) -> str:
        """Convert a sequence of integer indices (token ids) to a string."""
        raise NotImplementedError
```

## Character tokenizer

A Unicode string is a sequence of Unicode characters.

Version 16.0 of the [Unicode](https://en.wikipedia.org/wiki/List_of_Unicode_characters) standard defines **154998 characters** and 168 scripts.

Each character can be converted into a **code point** (integer) via `ord`. It can be converted back via `chr`.

```python
ord("a")
97
chr(97)
"a"
```

```python
ord("🌍")
127757
char(127757)
"🌍"
```

```python
class CharacterTokenizer(Tokenizer):
    """Represent a string as a sequence of Unicode code points."""

    def encode(self, string: str) -> list[int]:
        indices = list(map(ord, string))
        return indices

    def decode(self, indices: list[int]) -> str:
        string = "".join(list(map(chr, indices)))
        return string
```

```
Hello, 🌍! 你好!
[72, 101, 108, 108, 111, 44, 32, 127757, 33, 32, 20320, 22909, 33]
1.5384615384615385
```

- Very large vocabulary size (~150k).
- Many characters are quite rare.

## Byte tokenizer

Unicode strings can be represented as a sequence of bytes, which can be represented by integers between 0 and 255.

The most common Unicode encoding is [UTF-8](https://en.wikipedia.org/wiki/UTF-8).

```python
In [20]: bytes("a", encoding="utf-8")
Out[20]: b'a'

In [21]: bytes("🌍", encoding="utf-8")
Out[21]: b'\xf0\x9f\x8c\x8d'
```

```python
class ByteTokenizer(Tokenizer):
    """Represent a string as a sequence of bytes."""

    def encode(self, string: str) -> list[int]:
        string_bytes = string.encode("utf-8")
        indices = list(map(int, string_bytes))
        return indices

    def decode(self, indices: list[int]) -> str:
        string_bytes = bytes(indices)
        string = string_bytes.decode("utf-8")
        return string
```

```
Hello, 🌍! 你好!
[72, 101, 108, 108, 111, 44, 32, 240, 159, 140, 141, 33, 32, 228, 189, 160, 229, 165, 189, 33]
1.0
```

- Vocabulary size is small (255).
- Compression ratio is 1.

## Word tokenizer

This was used in classical NLP where we split string into words.

```python
class WordTokenizer(Tokenizer):
    """Split a string into words (usually on spaces)."""

    def __init__(self):
        # This regular expression keeps all alphanumeric characters together (words).
        self.regex_pattern = r"\w+|."

    def encode(self, string: str) -> list[str]:
        segments = regex.findall(self.regex_pattern, string)
        return segments

    def decode(self, segments: list[str]) -> str:
        string = "".join(segments)
        return string
```

```
Hello, 🌍! 你好!
['Hello', ',', ' ', '🌍', '!', ' ', '你好', '!']
2.5
```

- The number of words is huge.
- Many words are rare and the model won't learn much about them.
- New words we haven't seen during training get a special `UNK` token.


```python
# GPT2_TOKENIZER_REGEX
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py#L23
r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

## Pipeline

Tokenization pipeline
- Normalization
- Pre-tokenization
- Model
- Post-processor



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


## SentencePiece

https://github.com/google/sentencepiece

SentencePiece is a tokenization algorithm for the preprocessing of text that you can use with either BPE, WordPiece, or Unigram model.
- It considers the text as a sequence of Unicode characters, and replaces spaces with a special character, `▁`.
- Used in conjunction with the Unigram algorithm, it doesn’t require a pre-tokenization step, which is very useful for languages where the space character is not used (like Chinese or Japanese).
- SentencePiece is **reversible tokenization**: since there is no special treatment of spaces, decoding the tokens is done simply by concatenating them and replacing the `_`s with spaces — this results in the normalized text.

## Collateral

- [Let's build the GPT Tokenizer, Karpathy](https://www.youtube.com/watch?v=zduSFxRajkE)
