---
title: BPE tokenizer
weight: 2
bookToc: true
---

## Byte-Pair Encoding tokenizer

Byte-Pair Encoding (BPE) is a compression algorithm that iteratively replaces (**merges**) the most frequent pair if adjacent bytes/tokens
with a single new tokens. Intuitively if a word occurs in put text enough times it will be represented a s single sub-word token.

The BPE algorithm was introduced by Philip Gage in 1994 for [data compression](http://www.pennelynn.com/Documents/CUJ/HTML/94HTML/19940045.HTM) and later it was adapted to NLP for neural machine translation by  Sennrich et al., 2016.

{{% hint info %}}
[Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909), Rico Sennrich, Barry Haddow, Alexandra Birch, ACL 2016.
{{% /hint %}}

BPE was used in GPT-2.

{{% hint info %}}
[Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, GPT-2 2019.
{{% /hint %}}



### BPE Tokenizer training

### Vocabulary initialization

The tokenizer vocabulary is a one-to-one mapping from integer id (`int`) to bytestring (`bytes`) token. Our initial vocabulary is the set of all 256 possible byte values.

```python
class BPETokenizerTrainer:
    """Train the BPE tokenizer."""

    def __init__(self):
        # Our initial vocabulary is the set of all 256 possible byte values.
        self.vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}
        self.merges: dict[tuple[int, int], int] = {}
```

### Pre-tokenization

In practice before we do the merging we **pre-tokenize** the corpus into a sequence of **pre-tokens**. This is a coarse-grained tokenization over the corpus.

The original BPE implementation of Sennrich et al., 2016 pre-tokenizes by simply splitting on whitespace .
```python
>>> string = 'Hello, 🌍! 你好!'
>>> string.split(" ")
>>> ['Hello,', '🌍!', '你好!']
```
GPT-2 (Radford et al., 2019) uses a [regex-based pre-tokenizer](https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py#L23).

```python
>>> import regex as re
>>> PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s"""
>>> re.findall(PAT,string)
>>> ['Hello', ',', ' 🌍!', ' 你好', '!']
>>> [match.group() for match in re.finditer(PAT,string)]
>>> ['Hello', ',', ' 🌍!', ' 你好', '!']
```

Each pre-token is represented as a sequence of UTF-8 bytes.

```python
>>> [match.group().encode("utf-8") for match in re.finditer(pat,string)]
>>> [b'Hello', b',', b' \xf0\x9f\x8c\x8d!', b' \xe4\xbd\xa0\xe5\xa5\xbd', b'!']
```

```python
from collections import defaultdict
import regex as re

def pre_tokenize(string: str) -> dict[tuple[bytes], int]:
    """Regex based pre-tokenization."""

    # We will use a regex-based pre-tokenizer.
    # This was used by GPT-2 (Radford et al. 2019).
    # https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py#L23
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s"""

    pre_tokens_count = defaultdict(int)
    # Use finditer instead of findall.
    for match in re.finditer(PAT, string):
        # Key is tuple[bytes], for example (b'l', b'o', b'w').
        pre_token = tuple([bytes([c]) for c in match.group().encode("utf-8")])
        pre_tokens_count[pre_token] += 1

    return pre_tokens_count
```

```python
{ (b' ', b'l', b'o', b'w'): 4,
  (b' ', b'l', b'o', b'w', b'e', b'r'): 2,
  (b' ', b'n', b'e', b'w', b'e', b's', b't'): 6,
  (b' ', b'w', b'i', b'd', b'e', b's', b't'): 3,
  (b'l', b'o', b'w'): 1}
```

### Compute BPE merges

### Special tokens

## WordPiece

> BERT

## Unigram

> T5

## SentencePiece

https://github.com/google/sentencepiece

SentencePiece is a tokenization algorithm for the preprocessing of text that you can use with either BPE, WordPiece, or Unigram model.
- It considers the text as a sequence of Unicode characters, and replaces spaces with a special character, `▁`.
- Used in conjunction with the Unigram algorithm, it doesn’t require a pre-tokenization step, which is very useful for languages where the space character is not used (like Chinese or Japanese).
- SentencePiece is **reversible tokenization**: since there is no special treatment of spaces, decoding the tokens is done simply by concatenating them and replacing the `_`s with spaces — this results in the normalized text.


## Tokenizer-free approaches

Use bytes directly, promising, but have not yet been scaled up to the frontier.

https://arxiv.org/abs/2105.13626

https://arxiv.org/pdf/2305.07185

https://arxiv.org/abs/2412.09871

https://arxiv.org/abs/2406.19223


## Collateral

- [Let's build the GPT Tokenizer, Karpathy](https://www.youtube.com/watch?v=zduSFxRajkE)
- https://tiktokenizer.vercel.app/
