---
title: BPE tokenizer
weight: 2
bookToc: true
---

# Byte-Pair Encoding tokenizer

Byte-Pair Encoding (BPE) is a simple compression algorithm that iteratively replaces (**merges**) the most frequent pair of adjacent bytes/tokens
in a sequence  with a single new unused byte/token. Intuitively if a word occurs in the text enough times it will be represented a s single sub-word token.

The BPE algorithm was introduced by Philip Gage in 1994 for [data compression](http://www.pennelynn.com/Documents/CUJ/HTML/94HTML/19940045.HTM) and later it was adapted to NLP for neural machine translation by  Sennrich et al., 2016.
{{% hint info %}}
[Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909), Rico Sennrich, Barry Haddow, Alexandra Birch, ACL 2016.
{{% /hint %}}

BPE was later used in GPT-2.
{{% hint info %}}
[Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, GPT-2 2019.
{{% /hint %}}


## Training

### Vocabulary initialization

The tokenizer vocabulary is a one-to-one mapping from integer id (`int`) to bytestring (`bytes`) token. Our initial vocabulary is the set of all 256 possible byte values.

```python
self.vocab = {x: bytes([x]) for x in range(256)}
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

    pretokens_freq = defaultdict(int)
    # Use finditer instead of findall.
    for match in re.finditer(PAT, string):
        # Key is tuple[bytes], for example (b'l', b'o', b'w').
        pretoken = tuple([bytes([c]) for c in match.group().encode("utf-8")])
        pretokens_freq[pretoken] += 1

    return pretokens_freq
```

For illustration we will use this  stylized example from Sennrich et al. 2016.

```python
string = (
    " low low low low low "
    "lower lower widest widest widest "
    "newest newest newest newest newest newest"
)
```

We get the following after pre-tokenization.

```python
{
 (b' ', b'l', b'o', b'w'): 5,
 (b' ', b'l', b'o', b'w', b'e', b'r'): 2,
 (b' ', b'n', b'e', b'w', b'e', b's', b't'): 6,
 (b' ', b'w', b'i', b'd', b'e', b's', b't'): 3
 }
```

### BPE merges

After converting the input text into pre-tokens we compute the BPE merges.

The merge algorithm iteratively counts every pair of bytes and identifies the pair with the highest frequency (for example, `(b'e', b'st')`).
Every occurrence of this most frequent pair (`(b'e', b'st')`) is than merged and replaced with a new merged token (`(b'est')`). This new
merged token is added to our vocabulary.

>For efficiency we do not consider pairs that cross pre-token boundaries.


### Compute byte pair frequencies

```python
def __byte_pair_freq(self, pretokens_freq) -> dict[tuple[bytes], int]:
    """Compute byte pair frequencies."""

    byte_pair_freq = defaultdict(int)
    for pretoken, freq in pretokens_freq.items():
        n = len(pretoken)
        if n > 1:
            for i in range(n - 1):
                byte_pair = (pretoken[i], pretoken[i + 1])
                byte_pair_freq[byte_pair] += freq

    return byte_pair_freq
```

```python
{
(b' ', b'l'): 7,
(b' ', b'n'): 6,
(b' ', b'w'): 3,
(b'd', b'e'): 3,
(b'e', b'r'): 2,
(b'e', b's'): 9,
(b'e', b'w'): 6,
(b'i', b'd'): 3,
(b'l', b'o'): 7,
(b'n', b'e'): 6,
(b'o', b'w'): 7,
(b's', b't'): 9,
(b'w', b'e'): 8,
(b'w', b'i'): 3
}
```

### Find the best pair to merge

```python
def __best_pair_to_merge(self, byte_pair_freq) -> tuple[bytes]:
    """Return the best pair to merge."""

    # Find the pair with the highest frequency.
    # Break ties by preferring the lexicographically greater pair.
    max_freq = max(byte_pair_freq.values())
    max_pairs = [key for key, val in byte_pair_freq.items() if val == max_freq]
    best_pair_to_merge = max(max_pairs)

    return best_pair_to_merge
```

```python
(b's', b't') -> b'st'
```

### Merge the pre-tokens

```python
def __merge_pretokens(self, pretokens_freq, best_pair_to_merge) -> dict[tuple[bytes], int]:
    """Merge."""
    pretokens_freq_merged = defaultdict(int)
    for pretoken, count in pretokens_freq.items():
        pretoken_merged = pretoken
        n = len(pretoken)
        if n > 1:
            i = 0
            while i < n - 1:
                if (
                    pretoken[i] == best_pair_to_merge[0]
                    and pretoken[i + 1] == best_pair_to_merge[1]
                ):
                    pretoken_merged = (
                        pretoken[:i]
                        + tuple(
                            [bytes(best_pair_to_merge[0] + best_pair_to_merge[1])]
                        )
                        + pretoken[i + 2 :]
                    )
                i += 1
        pretokens_freq_merged[pretoken_merged] = count

    return pretokens_freq_merged
```

```python
{
(b' ', b'l', b'o', b'w'): 5,
(b' ', b'l', b'o', b'w', b'e', b'r'): 2,
(b' ', b'n', b'e', b'w', b'e', b'st'): 6,
(b' ', b'w', b'i', b'd', b'e', b'st'): 3
}
```

### Iterate

```python
def train(self, string: str, num_merges: int) -> BPETokenizerParams:
    """Train the byte-level BPE tokenizer.

    Args:
        string (str): The training data.
        num_merges (int): The number of BPE merges.

    Returns:
        BPETokenizerParams: The trained BPE tokenizer.
    """

    # Pre-tokenization.
    pretokens_freq = self.__pre_tokenize(string)

    # Merges.
    id = len(self.vocab)
    for i in range(num_merges):
        byte_pair_freq = self.__byte_pair_freq(pretokens_freq)

        best_pair_to_merge = self.__best_pair_to_merge(byte_pair_freq)
        merged_pair = bytes(best_pair_to_merge[0] + best_pair_to_merge[1])

        print(f"Merge {i+1}/{num_merges} {best_pair_to_merge} -> {merged_pair}")
        self.merges.append(best_pair_to_merge)
        self.vocab[id] = merged_pair
        id += 1

        pretokens_freq = self.__merge_pretokens(pretokens_freq, best_pair_to_merge)

    return BPETokenizerParams(
        vocab=self.vocab, merges=self.merges, special_tokens=self.special_tokens
    )
```

```python
Merge 1/10 (b's', b't') -> b'st'
Merge 2/10 (b'e', b'st') -> b'est'
Merge 3/10 (b'o', b'w') -> b'ow'
Merge 4/10 (b'l', b'ow') -> b'low'
Merge 5/10 (b' ', b'low') -> b' low'
Merge 6/10 (b'w', b'est') -> b'west'
Merge 7/10 (b'n', b'e') -> b'ne'
Merge 8/10 (b'ne', b'west') -> b'newest'
Merge 9/10 (b' ', b'newest') -> b' newest'
Merge 10/10 (b'w', b'i') -> b'wi'
```

### Special tokens

Often, some strings (for example, `<|endoftext|>`) are used to encode metadata (e.g., boundaries between documents).When encoding text, these special tokens should never be split into multiple tokens and should be preserved as a single token. These special tokens must be added to the vocablry, so they have a corresponding fixed token ID.


## Parameters

```python
@dataclass(frozen=True)
class BPETokenizerParams:
    """BPE tokenizer parameters."""

    # The tokenizer vocabulary is a one-to-one mapping
    # from integer id to bytestring token.
    vocab: dict[int, bytes]  # {258: b'est'}

    # A list of BPE merges produced from training.
    # The merges are in the order of creation.
    # Each item is a tuple of bytes (<token1>,<token2>),
    # meaning <token1> was merged with <token2>.
    merges: list[tuple[bytes, bytes]]  # [(b'e', b'st')]

    # The special tokens.
    # A list of strings to add to the vocabulary.
    special_tokens: list[str] # ["<|endoftext|>"]
```

## Encoding

## Decoding

## Collateral

- [Let's build the GPT Tokenizer, Karpathy](https://www.youtube.com/watch?v=zduSFxRajkE)
- https://tiktokenizer.vercel.app/
