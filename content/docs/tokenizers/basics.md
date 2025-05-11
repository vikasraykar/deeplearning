---
title: Basics
weight: 1
bookToc: true
---

## Tokenizers

A tokenizer encodes text (represented as a unicode string) to a **sequence of tokens** (represented as list of integer indices).

A Tokenizer is a class that implements the `encode` and `decode `methods.

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

The **vocabulary size** is number of possible tokens (integers).

> Explore various tokenizers here. https://tiktokenizer.vercel.app/

## Character based approaches

### Unicode standard

A Unicode string is a sequence of Unicode characters. **Unicode** is a text encoding standard that maps characters to integer **code points**. The version 16.0 of the [Unicode](https://en.wikipedia.org/wiki/List_of_Unicode_characters) (September 2004) standard defines **154,998 characters** across 168 scripts.

For example, the character "a" has the code point 97 and  the character "🌍" has the code point 127757.

Each character can be converted into a Unicode **code point** (integer) via `ord()` function. It can be converted back via `chr()` function.

```python
>>> ord("a")
97
>>> chr(97)
"a"
>>> ord("🌍")
127757
>>> chr(127757)
"🌍"
>>> ord('ಕ')
3221
>>> chr(3221)
'ಕ'
```

### Character-level tokenizer

We can encode a string as a sequence of Unicode code points as below.

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

{{% hint danger %}}
While the Unicode standard defines a mapping from characters to integer code points,
it is impractical to train tokenizers directly on Unicode code points, since the vocabulary would
be prohibitively **large** (~150k) and **sparse** (since many characters are quite rare).
{{% /hint %}}

### Unicode encodings

Unicode encoding converts a Unicode character into a **sequence of bytes**. The Unicode standard defines three encodings: UTF-8, UTF-16, and UTF-32, with [UTF-8](https://en.wikipedia.org/wiki/UTF-8) being the dominant encoding for
most of webpages.

To encode a Unicode string into UTF-8 we use the `encode()` function and and `decode` to convert it back.
```python
>>> "a".encode("utf-8")
>>> b'a'
>>> "🌍".encode("utf-8")
>>> b'\xf0\x9f\x8c\x8d'
>>> "ಕ".encode("utf-8")
>>> b'\xe0\xb2\x95'
>>> list(map(int,"ಕ".encode("utf-8")))
>>> [224, 178, 149]
```
{{% hint danger %}}
One byte does not necessarily correspond to one Unicode character.
{{% /hint %}}
```python
>>> len("ಕ")
>>> 1
>>> len("ಕ".encode("utf-8"))
>>> 3
```



### Byte-level tokenizer

Unicode strings can be represented as a sequence of bytes, which can be represented by integers between 0 and 255.

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

{{% hint danger %}}
By converting our Unicode code points into a sequence of bytes (via the UTF-8 encoding), we are essentially taking a sequence of code points (integers in the range 0 to 154,997) and transforming it into a sequence of of byte values (integers in the range 0 to 255). This 256-length vocabulary is easy to manage. When using byte-level tokenization, we do not worry about out-of-vocabulary tokens, since *any* input text can be encoded as a sequence of integers from 0 to 255. However this can result in extremely long input sequences (Compression ratio is 1).
{{% /hint %}}


## Word based approaches

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


## Sub-word based approaches

While byte-level tokenizers can alleviate out-of-vocabulary issues faced by word-level tokenizers,
it results in extremely long input sequences of bytes. This slows model training and inference.

Sub-word tokenization is a midpoint between word-level tokenizers and byte-level tokenizers. A
sub-word tokenizer trades-off a larger vocabulary size for better compression of input byte sequences.
For example, if the byte sequence `b'the'` often appears in our rat text training data, assigning it
an entry in the vocabulary would reduce this 3-token sequence to a single token.

