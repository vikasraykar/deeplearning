---
title: Transformers102
weight: 3
bookToc: true
---


## Transformer Language Models


|  category | task | example | sample use case
|:--|:--|:--|:--|
| **Decoder** | `vec2seq`  | GPT  | chat, image captioning
| **Encoder** | `seq2vec`  | BERT  | sentiment analysis
| **Encoder-Decoder** | `seq2seq`  | T5  | machine translation


## Auto-regressive models

Decompose the distribution over sequence of tokens into a product of conditional distributions.

{{< katex display=true >}}p(x_1,...,x_N)=\prod_{n=1}^{N} p(x_n|x_1,...,x_{n-1}){{< /katex >}}

### Markov models

Assume the conditional distribution is independent of all previous tokens except the {{< katex >}}L{{< /katex >}} most recent tokens (known as {{< katex >}}n{{< /katex >}}-gram models).

### Bi-gram model

{{< katex >}}L=1{{< /katex >}} bi-gram model

{{< katex display=true >}}p(x_1,...,x_N)=p(x_1)p(x_2|x_1)\prod_{n=3}^{N} p(x_n|x_{n-1}){{< /katex >}}

### Tri-gram model

{{< katex >}}L=2{{< /katex >}} tri-gram model

{{< katex display=true >}}p(x_1,...,x_N)=p(x_1)p(x_2|x_1)p(x_3|x_1,x_2)\prod_{n=4}^{N} p(x_n|x_{n-1},x_{n-2}){{< /katex >}}


## Decoder transformers

> `vec2seq` - Generative models that output sequence of tokens.
> GPT (Generative Pre-trained Transformer)

Use the transformer architecture to construct an auto-regressive model.

{{< katex display=true >}}p(x_1,...,x_N)=\prod_{n=1}^{N} p(x_n|x_1,...,x_{n-1}){{< /katex >}}

*The conditional distribution {{< katex >}}p(x_n|x_1,...,x_{n-1}){{< /katex >}} is modelled using a transformer.*

### Decoding at a high level

- The model takes as input a sequence of the first {{< katex >}}n-1{{< /katex >}} tokens.
- The output represents the conditional distribution for token {{< katex >}}n{{< /katex >}}.
- Draw a sample from this distribution to extend the sequence now to {{< katex >}}n{{< /katex >}} tokens.
- This new sequence can now be fed back to the model to generate a distribution over token {{< katex >}}n+1{{< /katex >}}.
- ...

### Add a linear-softmax layer

Input is a sequence of {{< katex >}}N{{< /katex >}} tokens {{< katex >}}x_1,...,x_N{{< /katex >}} each of dimensionality {{< katex >}}D{{< /katex >}}.

Transformer layer
{{< katex display=true >}}\mathbf{\widetilde{X}} = \text{TransformerLayer}[\mathbf{X}]{{< /katex >}}

Output  is a sequence of {{< katex >}}N{{< /katex >}} tokens {{< katex >}}\widetilde{x}_1,...,\widetilde{x}_N{{</ katex >}} each of dimensionality {{< katex >}}D{{< /katex >}}.

Each output needs to be a probability distribution over the vocabulary of {{< katex >}}K{{< /katex >}} tokens.

Add a linear-softmax layer.

{{< katex display=true >}}\mathbf{Y}_{N \times K} = \text{SoftMax}(\mathbf{\widetilde{X}}_{N \times D} {\mathbf{W}^p}_{D \times K}){{< /katex >}}

{{< katex >}}\mathbf{Y}{{< /katex >}} is matrix whose {{< katex >}}n{{< /katex >}}th row is {{< katex >}}y_n^{T}{{< /katex >}}.

Each softmax output unit has an associated cross-entropy loss.

<img src="/img/decoder.png"  width="600"/>

### Self-supervised training

The model can be trained using a large corpus of unlabelled natural language data in a self-supervised fashion.

(input) {{< katex >}}x_1,...,x_n{{< /katex >}} -> (output) {{< katex >}}x_{n+1}{{< /katex >}}

Each example can be processed independently.

We can actually process the entire sequence. **Shift the input sequence right by one step** so that each token acts both as a target value for the sequence of previous tokens and as an input value for subsequent tokens,

|   | | |  |  |  |  |
|:--|:--|:--|:--|:--|:--|:--|
| `INPUT` | `<start>` | {{< katex >}}x_1{{< /katex >}}  |  {{< katex >}}x_2{{< /katex >}}  | {{< katex >}}x_3{{< /katex >}}  | {{< katex >}}...{{< /katex >}}  | {{< katex >}}x_n{{< /katex >}}  |
| `OUPUT` |  {{< katex >}}y_1{{< /katex >}}  |  {{< katex >}}y_2{{< /katex >}}  | {{< katex >}}y_3{{< /katex >}}  |  {{< katex >}}y_4{{< /katex >}} | {{< katex >}}...{{< /katex >}}   | {{< katex >}}y_{n+1}{{< /katex >}}  |
| `TARGET` |  {{< katex >}}x_1{{< /katex >}}  |  {{< katex >}}x_2{{< /katex >}}  | {{< katex >}}x_3{{< /katex >}}  |  {{< katex >}}x_4{{< /katex >}} | {{< katex >}}...{{< /katex >}}   | {{< katex >}}x_{n+1}{{< /katex >}}  |

### Masked attention

The transformer can simply learn to copy the next input directly to the output, which is not available during decoding.

Set to zero all attention weights that correspond to a token attending to any later token in the sequence.

<img src="/img/mask.png"  width="300"/>

{{< katex display=true >}}\mathbf{Y} = \text{SoftMax}[\mathbf{Q}\mathbf{K}^{T}] \mathbf{V}{{< /katex >}}

> In practice we set the pre-activation values to {{< katex >}}-\infty{{< /katex >}} so that softmax evaluates to zero for the associated outputs. and also takes care of normalization across non-zero outputs.

### Sampling strategies

The output of the trained model is a probability distribution over the space of tokens, given by the softmax activation function, which represents the probability of the next token given the current token sequence.

### Greedy search

Select the token with the highest probability.

Deterministic.

> {{< katex >}}\mathcal{O}(KN){{< /katex >}}

### Beam search

- Instead of choosing the single most probable token value at each step, we maintain a set of {{< katex >}}B{{< /katex >}} (*beam width*) hypothesis, each consisting of a sequence of token values up to step {{< katex >}}n{{< /katex >}}.
- We then feed all these sequences through the network and for each sequence we find the {{< katex >}}B{{< /katex >}} most probable token values, thereby creating {{< katex >}}B^2{{< /katex >}} possible hypothesis.
- This list is then pruned by selecting the most probable {{< katex >}}B{{< /katex >}} hypotheses according to the total probability of the extended sequence.

> {{< katex >}}\mathcal{O}(BKN){{< /katex >}}

### Sampling from softmax

Sampling from the softmax distribution at each step.

Can lead to nonsensical sequences sometime due to large vocabulary

### Top-k sampling

Consider only the states having a top-{{< katex >}}k{{< /katex >}} probabilities and sample from these according to their renormalized probabilities.

### Top-p sampling/Nucleus sampling

Calculates the cumulative probability of the top outputs until a threshold is reached and then samples from this restricted set of tokens.

### Soft top-k sampling

Introduce a parameter {{< katex >}}T{{< /katex >}} called temperature  into the definition of softmax function.

{{< katex display=true >}}y_i=\frac{\exp(a_i/T)}{\sum_j \exp(a_j/T)}{{< /katex >}}

{{< katex >}}T=0{{< /katex >}} Greedy sampling

{{< katex >}}T=1{{< /katex >}} Softmax sampling


## Encoder transformers

> `seq2vec` - Generative models that output sequence of tokens.
> BERT (Bidirectional Encoder Representations from Transformers)

Take sequence of tokens as input and produce fixed-length vectors to be used for various downstream tasks.
- Pre-training
- Fine-tuning

The first token of every input sequence is given by a special token `<class>`.

<img src="/img/encoder.png"  width="600"/>

### Pre-training

A randomly chosen subset of tokens (say 15%) is replaced with a special token denoted by `<mask>`.

The transformer is trained to predict the missing tokens at the corresponding output nodes.

Of the masked tokens
- 80% are replaced with `<mask>`
- 10% are replaced with a random token.
- 10% the original token is retained.


### Fine-tuning

Once the encoder is trained it can then be fine-tuned for a task.

Typically a new layer is built on top of the embedding for the `<class>` token.


## Encoder-Decoder transformers

> `seq2seq`
> T5

An encoder is used to map the input token sequence to a suitable internal representation.

Cross-attention - Same as self-attention but the key and the value vectors come from the encoder representation.

<img src="/img/crossattention.png"  width="200"/>

<img src="/img/encoderdecoder.png"  width="600"/>

