---
title: Transformers101
weight: 1
bookToc: true
---

# Transformers 101

A transformer has 3 main components.
1. Multi-head scaled self-attention.
1. MLP with residual connections and layer normalization.
1. Positional encodings.

## TLDR

|Transformer| Description |
| :-- | :-- |
{{< katex >}}\mathbf{X}{{< /katex >}} | embedding matrix
{{< katex >}}\mathbf{X} = \mathbf{X} + \mathbf{R}{{< /katex >}} | position encoding matrix
{{< katex >}}\mathbf{Y} = \text{TransformerLayer}[\mathbf{X}]{{< /katex >}} | transformer
{{< katex >}}\mathbf{Y} = \text{SoftMax}[\mathbf{X}\mathbf{X}^{T}] \mathbf{X}{{< /katex >}} | dot-product self-attention
{{< katex >}}\mathbf{Y} = \text{SoftMax}[\mathbf{X} \mathbf{W}^{q}\mathbf{W}^{kT}\mathbf{X}^{T}] \mathbf{X} \mathbf{W}^{v}{{< /katex >}} | Query, Key, Value matrices
{{< katex >}}\mathbf{Y} = \text{SoftMax}[\mathbf{Q}\mathbf{K}^{T}] \mathbf{V} {{< /katex >}} | Query, Key, Value matrices
{{< katex >}}\mathbf{Y} = \text{SoftMax}[\frac{\mathbf{Q}\mathbf{K}^{T}}{\sqrt{D}}] \mathbf{V} {{< /katex >}} | Scaled dot-product self attention
{{< katex >}}\mathbf{Y} = \text{Concat}[\mathbf{H}_1,...,\mathbf{H}_H]\mathbf{W}^o {{< /katex >}} where {{< katex >}}\mathbf{H}_h = \text{SoftMax}\left[\frac{\mathbf{Q_h}\mathbf{K_h}^{T}}{\sqrt{D_k}}\right] \mathbf{V_h}{{< /katex >}} | Multi-head attention
{{< katex >}}\mathbf{Z} = \text{LayerNorm}\left[\mathbf{Y}(\mathbf{X})+\mathbf{X}\right]{{< /katex >}} | layer normalization and residual connection
{{< katex >}}\mathbf{X^*} = \text{LayerNorm}\left[\text{MLP}(\mathbf{Z})+\mathbf{Z}\right]{{< /katex >}} | MLP layer

| Parameter| Count | Description |
|:-- | :-- | :-- |
| {{< katex >}}\mathbf{E}{{< /katex >}} |  {{< katex >}}VD{{< /katex >}}   | The token embedding matrix. {{< katex >}}V{{< /katex >}} is the size of the vocabulary and {{< katex >}}D{{< /katex >}} is the dimensions of the embeddings.
| {{< katex >}}\mathbf{W}^q_h{{< /katex >}} {{< katex >}}\mathbf{W}^k_h{{< /katex >}} {{< katex >}}\mathbf{W}^v_h{{< /katex >}} | {{< katex >}}3HD^2{{< /katex >}} | The query, key and the value matrices each of dimension {{< katex >}}D \times D {{< /katex >}}for the {{< katex >}}H{{< /katex >}}heads.
| {{< katex >}}\mathbf{W}^o{{< /katex >}} | {{< katex >}}HD^2{{< /katex >}} | The output matrix of dimension {{< katex >}}HD \times D {{< /katex >}}.
| {{< katex >}}\mathbf{W}^{ff}_{1}{{< /katex >}} {{< katex >}}\mathbf{W}^{ff}_{2}{{< /katex >}} | {{< katex >}}2DD_{ff}{{< /katex >}} | The parameters of the two-layer MLP.
|  | {{< katex >}}8D^2{{< /katex >}} | Typically {{< katex  >}}D_{ff} = 4 D{{< /katex  >}}
|   | | **{{< katex >}}(4H+8)D^2{{< /katex >}} total parameters**

## Multi-head scaled self-attention.

### Tokens

We will start with the concept of **tokens**. As token can be
- word
- sub-word
- image patch
- amino acid
- etc.

### Token embeddings

Let {{< katex >}}\mathbf{x}_n \in \mathbb{R}^D{{< /katex >}} be a column vector of {{< katex >}}D{{< /katex >}} features corresponding to a token {{< katex >}}n{{< /katex >}}.

This corresponds to the {{< katex >}}D{{< /katex >}}-dimensional **embedding vector** of the token.

### Embedding matrix

We can stack all the embedding vectors {{< katex >}}\left\{\mathbf{x}_n\right\}_{n=1}^{N}{{< /katex >}} for a **sequence** of {{< katex >}}N{{< /katex >}} tokens as rows into an embedding matrix {{< katex >}}\mathbf{X}{{< /katex >}}.

{{< katex display=true >}}\mathbf{X}_{\text{N (tokens)} \times \text{D (features)}}{{< /katex >}}

### Transformer layer

A **transformer** transforms the embedding matrix {{< katex >}}\mathbf{X}{{< /katex >}} to another matrix {{< katex >}}\mathbf{Y}{{< /katex >}} of the same dimension.

{{< katex display=true >}}
\mathbf{Y}_{N \times D} = \text{TransformerLayer}[\mathbf{X}_{N \times D}]
{{< /katex >}}

The goal of transformation is that the new space {{< katex >}}\mathbf{Y}{{< /katex >}} will have a richer internal representation that is better suited to solve downstream tasks.

The embeddings are trained to capture elementary semantic properties, words with similar meaning should map to nearby locations in the embedding space.

> A transformer can be viewed as a richer form of embedding in which the embedding vector for a token is mapped to a location that depends on the embedding vectors of other tokens in the sequence.

1. I _swam_ across the _river_ to get to the other **bank**. (bank~water)
1. I _walked_ across the road to get _cash_ from the **bank**. (bank~money)

### Attention

We do this via the notion of attention.

To determine the appropriate interpretation of the token **bank** the transformer processing a sentence should **attend to** (or give more importance to) specific words from the rest of the sequence.

> Originally developed by  Bahdanau, Cho, and Bengio, 2015 [^nmt_iclr2015] as an enhancement to RNNs for machine translation. Vaswani _et al_, 2017 [^attention_iclr2015] later completely eliminated the recurrence structure and instead focussed only on the attention mechanism.

We will do this via the notion of **attention** where we generate the output transformed vector {{< katex >}}\mathbf{y}_n{{< /katex >}} via a linear combination of all the input vectors, that is, by attending to all the input vectors.

{{< katex display=true >}}
\mathbf{y}_n = \sum_{m=1}^{N} a_{nm} \mathbf{x}_m
{{< /katex >}}

{{< katex >}}a_{nm}{{< /katex >}} are called **attention weights/coefficients**.

The attention coefficients should satisfy the following two properties.
- {{< katex >}}a_{nm} \geq 0{{< /katex >}}
- {{< katex >}}\sum_{m=1}^N a_{nm} =1{{< /katex >}}

Partition of unity ({{< katex >}}0 \leq a_{nm} \leq 1{{< /katex >}}).

### Self-attention

We want to capture the notion of how similar a token is to other tokens.

This can be done via a dot product between the **query vector** ({{< katex >}}\mathbf{x}_{n}{{< /katex >}}) and the **key vector** ({{< katex >}}\mathbf{x}_{m}{{< /katex >}}).

{{< katex display=true >}}
a_{nm} \propto (\mathbf{x}_{n}^T\mathbf{x}_m)
{{< /katex >}}

The attention coefficients should satisfy the following two properties.

- {{< katex >}}a_{nm} \geq 0{{< /katex >}}
- {{< katex >}}\sum_{m=1}^N a_{nm} =1{{< /katex >}}

This can be achieved by a **soft-max** of the dot products.

{{< katex display=true >}}
a_{nm} = \text{SoftMax}(\mathbf{x}_{n}^T\mathbf{x}_m) = \frac{\exp(\mathbf{x}_{n}^T\mathbf{x}_m)}{\sum_{m'=1}^{N} \exp(\mathbf{x}_{n}^T\mathbf{x}_m')}
{{< /katex >}}

{{< katex display=true >}}
\mathbf{y}_n = \sum_{m=1}^{N} \text{SoftMax}(\mathbf{x}_{n}^T\mathbf{x}_m) \mathbf{x}_m
{{< /katex >}}

### Query, Key, Value

A bit of terminology taken from the IR literature.

- **Query** The search query that the user types on a search engine.
- **Key** The feature representation of each document.
- **Value** The actual document.

The **query** is attending to a particular **value** whose key closely matches the **query** (hard attention).
{{< katex display=true >}}
\mathbf{y}_n = \sum_{m=1}^{N} \text{SoftMax}(\mathbf{x}_{n}^T\mathbf{x}_m) \mathbf{x}_m
{{< /katex >}}

- {{< katex >}}\mathbf{x}_{n}{{< /katex >}} is the query.
- {{< katex >}}\mathbf{x}_m{{< /katex >}} ({{< katex >}}m=1,...N{{< /katex >}}) are the keys.
- {{< katex >}}\mathbf{x}_m{{< /katex >}} ({{< katex >}}m=1,...N{{< /katex >}}) are the values.


### Dot-product self-attention
So we now have the first definition of the transformer layer.

{{< katex display=true >}}
\mathbf{Y}_{N \times D} = \text{TransformerLayer}[\mathbf{X}_{N \times D}]
{{< /katex >}}

{{< katex display=true >}}\mathbf{Y} = \text{SoftMax}[\mathbf{X}\mathbf{X}^{T}] \mathbf{X}{{< /katex >}}

{{< katex display=true >}}\mathbf{Y}_{N \times D} = \text{SoftMax}[\mathbf{X}_{N \times D} \mathbf{X}^{T}_{D \times N} ] \mathbf{X}_{N \times D} {{< /katex >}}

> Other than the embedding matrix this has no no learnable parameters yet.

| Parameter| Count | Description |
| -------- | ------- | --- |
| {{< katex >}}\mathbf{E}{{< /katex >}} |  {{< katex >}}VD{{< /katex >}}   | The token embedding matrix. {{< katex >}}V{{< /katex >}} is the size of the vocabulary and {{< katex >}}D{{< /katex >}} is the dimensions of the mebeddings.

> {{< katex >}}\mathcal{O}(2N^2D){{< /katex >}} computations.

### Network parameters

The above has no learnable parameters.

Each feature value {{< katex >}}x_{ni}{{< /katex >}} is equally important in the dot product.

We will introduce a {{< katex >}}D \times D{{< /katex >}} matrix {{< katex >}}\mathbf{U}{{< /katex >}} of **learnable weights**.

{{< katex display=true >}}\mathbf{X^{*}}_{N \times D}=\mathbf{X}_{N \times D}\mathbf{U}_{D \times D}{{< /katex >}}

With this we have the second iteration of the transformer layer.

{{< katex display=true >}}\mathbf{Y} = \text{SoftMax}[\mathbf{X}\mathbf{U}\mathbf{U}^{T}\mathbf{X}^{T}] \mathbf{X}\mathbf{U}{{< /katex >}}

> We now have {{< katex >}}D^2{{< /katex >}} learnable parameters.
> {{< katex >}}\mathcal{O}(2N^2D+ND^2){{< /katex >}} computations

While this has more flexibility the matrix {{< katex >}}\mathbf{X}\mathbf{U}\mathbf{U}^{T}\mathbf{X}^{T}{{< /katex >}} is symmetric, whereas we would like the attention matrix to support significant asymmetry.

- The word *chisel* should be strongly associated with *tool* since every chisel is a tool.
- The word *tool* should be weakly associated with the word *chisel* since there are many other kinds of tools besides chisel.

### Query, Key, Value matrices

To overcome these limitations, we introduce separate Query, Key, Value matrices each having their own independent linear transformations.

{{< katex display=true >}}\mathbf{Q} = \mathbf{X} \mathbf{W}^{q}{{< /katex  >}}
{{< katex display=true >}}\mathbf{K} = \mathbf{X} \mathbf{W}^{k}{{< /katex  >}}
{{< katex display=true >}}\mathbf{V} = \mathbf{X} \mathbf{W}^{v}{{< /katex  >}}

Let's check the dimensions.

{{< katex display=true >}}\mathbf{Q}_{N \times D_q} = \mathbf{X}_{N \times D} \mathbf{W}^{q}_{D \times D_q}{{< /katex  >}}
{{< katex display=true >}}\mathbf{K}_{N \times D_k} = \mathbf{X}_{N \times D} \mathbf{W}^{k}_{D \times D_k}{{< /katex  >}}
{{< katex display=true >}}\mathbf{V}_{N \times D_v} = \mathbf{X}_{N \times D} \mathbf{W}^{v}_{D \times D_v}{{< /katex  >}}

Typically {{< katex >}}D_q=D_k{{< /katex >}}

{{< katex display=true >}}\mathbf{Q}_{N \times D_k} = \mathbf{X}_{N \times D} \mathbf{W}^{q}_{D \times D_k}{{< /katex  >}}
{{< katex display=true >}}\mathbf{K}_{N \times D_k} = \mathbf{X}_{N \times D} \mathbf{W}^{k}_{D \times D_k}{{< /katex  >}}
{{< katex display=true >}}\mathbf{V}_{N \times D_v} = \mathbf{X}_{N \times D} \mathbf{W}^{v}_{D \times D_v}{{< /katex  >}}

With this we have the third iteration of the transformer layer.

{{< katex display=true >}}\mathbf{Y} = \text{SoftMax}[\mathbf{Q}\mathbf{K}^{T}] \mathbf{V}{{< /katex >}}

Let's check the dimensions once.

{{< katex display=true >}}\mathbf{Y}_{N \times D_v} = \text{SoftMax}[\mathbf{Q}_{N \times D_k}\mathbf{K}^{T}_{D_k \times N}] \mathbf{V}_{N \times D_v}{{< /katex >}}

A common choice is {{< katex >}}D_k=D_v=D{{< /katex >}}. This also makes the output dimension same as the input and helps later to add residual connections.

{{< katex display=true >}}\mathbf{Y}_{N \times D} = \text{SoftMax}[\mathbf{Q}_{N \times D}\mathbf{K}^{T}_{D \times N}] \mathbf{V}_{N \times D}{{< /katex >}}

> We now have {{< katex >}}3D^2{{< /katex >}} learnable parameters.

| Parameter| Count | Description |
|:-- | :-- | :-- |
| {{< katex >}}\mathbf{E}{{< /katex >}} |  {{< katex >}}VD{{< /katex >}}   | The token embedding matrix. {{< katex >}}V{{< /katex >}} is the size of the vocabulary and {{< katex >}}D{{< /katex >}} is the dimensions of the embeddings.
| {{< katex >}}\mathbf{W}^q{{< /katex >}} {{< katex >}}\mathbf{W}^k{{< /katex >}} {{< katex >}}\mathbf{W}^v{{< /katex >}} | {{< katex >}}3D^2{{< /katex >}} | The query, key and the value matrices each of dimension {{< katex >}}D \times D{{< /katex >}}

> {{< katex >}}\mathcal{O}(2N^2D+3ND^2){{< /katex >}} computations

### Recap

|Transformer| Description |
| :-- | :-- |
{{< katex >}}\mathbf{Y} = \text{TransformerLayer}[\mathbf{X}]{{< /katex >}} | transformer
{{< katex >}}\mathbf{Y} = \text{SoftMax}[\mathbf{X}\mathbf{X}^{T}] \mathbf{X}{{< /katex >}} | dot-product self-attention
{{< katex >}}\mathbf{Y} = \text{SoftMax}[\mathbf{X}\mathbf{U}\mathbf{U}^{T}\mathbf{X}^{T}] \mathbf{X}\mathbf{U}{{< /katex >}} | learnable parameters
{{< katex >}}\mathbf{Y} = \text{SoftMax}[\mathbf{X} \mathbf{W}^{q}\mathbf{W}^{kT}\mathbf{X}^{T}] \mathbf{X} \mathbf{W}^{v}{{< /katex >}} | Query, Key, Value matrices
{{< katex >}}\mathbf{Y} = \text{SoftMax}[\mathbf{Q}\mathbf{K}^{T}] \mathbf{V} {{< /katex >}} | Query, Key, Value matrices


### Scaled dot-product self-attention

The gradients of the soft-max become exponentially small for inputs of high magnitude.

Hence we scale it as follows.

{{< katex display=true >}}\mathbf{Y} = \text{SoftMax}\left[\frac{\mathbf{Q}\mathbf{K}^{T}}{\sqrt{D_k}}\right] \mathbf{V}{{< /katex >}}

{{< katex >}}D_k{{< /katex >}} which is the variance of the dot-product.

> If the elements of the query and key vectors were all independent random variables with zero mean and unit variance, then the variance of the dot product would be {{< katex >}}D_k{{< /katex >}}

### Single attention head

This is known a s single attention head.
{{< katex display=true >}}\mathbf{Y} = \text{SoftMax}\left[\frac{\mathbf{Q}\mathbf{K}^{T}}{\sqrt{D_k}}\right] \mathbf{V}{{< /katex >}}


### Multi-head attention

Capture multiple patterns of attention (for example, tense, vocabulary etc.).

> Sort of similar to using multiple different filters in each layer of a convolutional neural network.

We have {{< katex >}}H{{< /katex >}} attention heads indexed by {{< katex >}}h=1,...,H{{< /katex >}}.

{{< katex display=true >}}\mathbf{H}_h = \text{Attention}(\mathbf{Q_h},\mathbf{K_h},\mathbf{V_h}) = \text{SoftMax}\left[\frac{\mathbf{Q_h}\mathbf{K_h}^{T}}{\sqrt{D_k}}\right] \mathbf{V_h}{{< /katex >}}

{{< katex display=true >}}\mathbf{Q}_h = \mathbf{X} \mathbf{W}^{q}_h{{< /katex >}}
{{< katex display=true >}}\mathbf{K}_h = \mathbf{X} \mathbf{W}^{k}_h{{< /katex >}}
{{< katex display=true >}}\mathbf{V}_h = \mathbf{X} \mathbf{W}^{v}_h{{< /katex >}}

The output from each heads are first concatenated into a single matrix and then linearly transformed using another matrix.

{{< katex display=true >}}\mathbf{Y}(\mathbf{X}) =\text{Concat}[\mathbf{H}_1,...,\mathbf{H}_H]\mathbf{W}^o{{< /katex >}}

Let's check the dimensions

{{< katex display=true >}}\mathbf{Y}_{N \times D} =\text{Concat}[\mathbf{H}_1,...,\mathbf{H}_H]_{N \times HD_v} \mathbf{W}^o_{HD_v \times D}{{< /katex  >}}

Typically {{< katex >}}D_v = D/H{{< /katex >}}.

> We now have {{< katex >}}3HD^2{{< /katex >}} learnable parameters.

| Parameter| Count | Description |
|:-- | :-- | :-- |
| {{< katex >}}\mathbf{E}{{< /katex >}} |  {{< katex >}}VD{{< /katex >}}   | The token embedding matrix. {{< katex >}}V{{< /katex >}} is the size of the vocabulary and {{< katex >}}D{{< /katex >}} is the dimensions of the embeddings.
| {{< katex >}}\mathbf{W}^q_h{{< /katex >}} {{< katex >}}\mathbf{W}^k_h{{< /katex >}} {{< katex >}}\mathbf{W}^v_h{{< /katex >}} | {{< katex >}}3HD^2{{< /katex >}} | The query, key and the value matrices each of dimension {{< katex >}}D \times D {{< /katex >}}for the {{< katex >}}H{{< /katex >}}heads.
| {{< katex >}}\mathbf{W}^o{{< /katex >}} | {{< katex >}}HD^2{{< /katex >}} | The output matrix of dimension {{< katex >}}HD \times D {{< /katex >}}.


> {{< katex >}}\mathcal{O}(2HN^2D+4HND^2){{< /katex >}} computations

### Recap

|Transformer| Description |
| :-- | :-- |
{{< katex >}}\mathbf{Y} = \text{TransformerLayer}[\mathbf{X}]{{< /katex >}} | transformer
{{< katex >}}\mathbf{Y} = \text{SoftMax}[\mathbf{X}\mathbf{X}^{T}] \mathbf{X}{{< /katex >}} | dot-product self-attention
{{< katex >}}\mathbf{Y} = \text{SoftMax}[\mathbf{X}\mathbf{U}\mathbf{U}^{T}\mathbf{X}^{T}] \mathbf{X}\mathbf{U}{{< /katex >}} | learnable parameters
{{< katex >}}\mathbf{Y} = \text{SoftMax}[\mathbf{X} \mathbf{W}^{q}\mathbf{W}^{kT}\mathbf{X}^{T}] \mathbf{X} \mathbf{W}^{v}{{< /katex >}} | Query, Key, Value matrices
{{< katex >}}\mathbf{Y} = \text{SoftMax}[\mathbf{Q}\mathbf{K}^{T}] \mathbf{V} {{< /katex >}} | Query, Key, Value matrices
{{< katex >}}\mathbf{Y} = \text{SoftMax}[\frac{\mathbf{Q}\mathbf{K}^{T}}{\sqrt{D}}] \mathbf{V} {{< /katex >}} | Scaled dot-product self attention
{{< katex >}}\mathbf{Y} = \text{Concat}[\mathbf{H}_1,...,\mathbf{H}_H]\mathbf{W}^o {{< /katex >}} where {{< katex >}}\mathbf{H}_h = \text{SoftMax}\left[\frac{\mathbf{Q_h}\mathbf{K_h}^{T}}{\sqrt{D_k}}\right] \mathbf{V_h}{{< /katex >}} | Multi-head attention

## MLP layers

### Residual connections

To improve training efficiency we introduce residual connections that bypass the multi-head structure.

{{< katex display=true >}}\mathbf{Y}(\mathbf{X}) = \text{TransformerLayer}[\mathbf{X}]{{< /katex  >}}

{{< katex display=true >}}\mathbf{Z} = \mathbf{Y}(\mathbf{X})+\mathbf{X}{{< /katex  >}}

### Layer normalization

Layer normalization is then added also to improve training efficiency (Bao, Kiros, and Hinton 2016)

### Post-norm

{{< katex display=true >}}\mathbf{Z} = \text{LayerNorm}\left[\mathbf{Y}(\mathbf{X})+\mathbf{X}\right]{{< /katex >}}

### Pre-norm

{{< katex display=true >}}\mathbf{Z} = \mathbf{Y}(\text{LayerNorm}\left[\mathbf{X}\right])+\mathbf{X}{{< /katex >}}

> Pre-norm is most widely used these days while the original paper used post-norm.

### MLP layer

We further add a MLP layer, for example, a two layer fully connected network with ReLU hidden units (typically bias is excluded).

{{< katex display=true >}}\mathbf{X^*} = \text{MLP}(\mathbf{Z})=\text{R/GeLU}(\mathbf{Z}\mathbf{W}^{ff}_{1})\mathbf{W}^{ff}_2{{< /katex  >}}

Let's check the dimensions.

{{< katex display=true >}}\mathbf{X^*} = \text{R/GeLU}(\mathbf{Z}_{N\times D}{\mathbf{W}^{ff1}_{1}}_{D \times D_{ff}}){\mathbf{W}^{ff}_{2}}_{D_{ff} \times D}{{< /katex  >}}

> Typically {{< katex  >}}D_{ff} = 4 D{{< /katex  >}}
### Residual connection again

{{< katex display=true >}}\mathbf{X^*} = \text{MLP}(\mathbf{Z})+\mathbf{Z}{{< /katex  >}}

### Layer normalization again

{{< katex display=true >}}\mathbf{X^*} = \text{LayerNorm}\left[\text{MLP}(\mathbf{Z})+\mathbf{Z}\right]{{< /katex  >}}

> We now have {{< katex >}}2DD_{ff}{{< /katex >}} learnable parameters.

| Parameter| Count | Description |
|:-- | :-- | :-- |
| {{< katex >}}\mathbf{E}{{< /katex >}} |  {{< katex >}}VD{{< /katex >}}   | The token embedding matrix. {{< katex >}}V{{< /katex >}} is the size of the vocabulary and {{< katex >}}D{{< /katex >}} is the dimensions of the embeddings.
| {{< katex >}}\mathbf{W}^q_h{{< /katex >}} {{< katex >}}\mathbf{W}^k_h{{< /katex >}} {{< katex >}}\mathbf{W}^v_h{{< /katex >}} | {{< katex >}}3HD^2{{< /katex >}} | The query, key and the value matrices each of dimension {{< katex >}}D \times D {{< /katex >}}for the {{< katex >}}H{{< /katex >}}heads.
| {{< katex >}}\mathbf{W}^o{{< /katex >}} | {{< katex >}}HD^2{{< /katex >}} | The output matrix of dimension {{< katex >}}HD \times D {{< /katex >}}.
| {{< katex >}}\mathbf{W}^{ff}_{1}{{< /katex >}} {{< katex >}}\mathbf{W}^{ff}_{2}{{< /katex >}} | {{< katex >}}2DD_{ff}{{< /katex >}} | The parameters of the two-layer MLP.
|  | {{< katex >}}8D^2{{< /katex >}} | Typically {{< katex  >}}D_{ff} = 4 D{{< /katex  >}}

> {{< katex >}}\mathcal{O}(2HN^2D+4HND^2+2NDD_{ff}){{< /katex >}} computations

> {{< katex >}}\mathcal{O}(2HN^2D+4HND^2+8ND^2){{< /katex >}} computations

### Recap

|Transformer| Description |
| :-- | :-- |
{{< katex >}}\mathbf{Y} = \text{TransformerLayer}[\mathbf{X}]{{< /katex >}} | transformer
{{< katex >}}\mathbf{Y} = \text{SoftMax}[\mathbf{X}\mathbf{X}^{T}] \mathbf{X}{{< /katex >}} | dot-product self-attention
{{< katex >}}\mathbf{Y} = \text{SoftMax}[\mathbf{X} \mathbf{W}^{q}\mathbf{W}^{kT}\mathbf{X}^{T}] \mathbf{X} \mathbf{W}^{v}{{< /katex >}} | Query, Key, Value matrices
{{< katex >}}\mathbf{Y} = \text{SoftMax}[\mathbf{Q}\mathbf{K}^{T}] \mathbf{V} {{< /katex >}} | Query, Key, Value matrices
{{< katex >}}\mathbf{Y} = \text{SoftMax}[\frac{\mathbf{Q}\mathbf{K}^{T}}{\sqrt{D}}] \mathbf{V} {{< /katex >}} | Scaled dot-product self attention
{{< katex >}}\mathbf{Y} = \text{Concat}[\mathbf{H}_1,...,\mathbf{H}_H]\mathbf{W}^o {{< /katex >}} where {{< katex >}}\mathbf{H}_h = \text{SoftMax}\left[\frac{\mathbf{Q_h}\mathbf{K_h}^{T}}{\sqrt{D_k}}\right] \mathbf{V_h}{{< /katex >}} | Multi-head attention
{{< katex >}}\mathbf{Z} = \text{LayerNorm}\left[\mathbf{Y}(\mathbf{X})+\mathbf{X}\right]{{< /katex >}} | layer normalization and residual connection
{{< katex >}}\mathbf{X^*} = \text{LayerNorm}\left[\text{MLP}(\mathbf{Z})+\mathbf{Z}\right]{{< /katex >}} | MLP layer

## Positional encodings

A transformer is equivariant with respect to input permutation, that is, it does not depend on the order of the tokens.

- The food was bad, not good at all.
- The food was good, not bad at all.

We need to find a way to inject the token order information.

We add a **position encoding vector** ({{< katex >}}\mathbf{r}_n{{< /katex >}}) to each token vector ({{< katex >}}\mathbf{x}_n{{< /katex >}}).

{{< katex display=true >}}\mathbf{x}_n^* = \mathbf{x}_n + \mathbf{r}_n{{< /katex  >}}

We could associate an integer 1, 2, 3... with each position.
- Magnitude of the value may increase without bound and corrupt the embedding vector.
- Will not generalize well the new input sequences longer than the training data.

Desiderata (Dufter, Schmitt, and Schutze, 2021)

- Unique representation for each token.
- Should be bounded.
- Generalize to longer sequences.
- Compute relative position of tokens.

### Sinusoidal functions

For a given position {{< katex >}}n{{< /katex >}} the position encoding vector has components {{< katex >}}\mathbf{r}_{ni}{{< /katex >}} given by sinusoids of steadily increasing wavelengths.

{{< katex display=true >}}\mathbf{r}_{ni} = \text{sin}\left(\frac{n}{L^{i/D}}\right) \text{if i is even}{{< /katex  >}}
{{< katex display=true >}}\mathbf{r}_{ni} = \text{cos}\left(\frac{n}{L^{(i-1)/D}}\right) \text{if i is odd}{{< /katex  >}}

>Sort of binary representation of numbers.

## Summary

|Transformer| Description |
| :-- | :-- |
{{< katex >}}\mathbf{X}{{< /katex >}} | embedding matrix
{{< katex >}}\mathbf{X} = \mathbf{X} + \mathbf{R}{{< /katex >}} | position encoding matrix
{{< katex >}}\mathbf{Y} = \text{TransformerLayer}[\mathbf{X}]{{< /katex >}} | transformer
{{< katex >}}\mathbf{Y} = \text{SoftMax}[\mathbf{X}\mathbf{X}^{T}] \mathbf{X}{{< /katex >}} | dot-product self-attention
{{< katex >}}\mathbf{Y} = \text{SoftMax}[\mathbf{X} \mathbf{W}^{q}\mathbf{W}^{kT}\mathbf{X}^{T}] \mathbf{X} \mathbf{W}^{v}{{< /katex >}} | Query, Key, Value matrices
{{< katex >}}\mathbf{Y} = \text{SoftMax}[\mathbf{Q}\mathbf{K}^{T}] \mathbf{V} {{< /katex >}} | Query, Key, Value matrices
{{< katex >}}\mathbf{Y} = \text{SoftMax}[\frac{\mathbf{Q}\mathbf{K}^{T}}{\sqrt{D}}] \mathbf{V} {{< /katex >}} | Scaled dot-product self attention
{{< katex >}}\mathbf{Y} = \text{Concat}[\mathbf{H}_1,...,\mathbf{H}_H]\mathbf{W}^o {{< /katex >}} where {{< katex >}}\mathbf{H}_h = \text{SoftMax}\left[\frac{\mathbf{Q_h}\mathbf{K_h}^{T}}{\sqrt{D_k}}\right] \mathbf{V_h}{{< /katex >}} | Multi-head attention
{{< katex >}}\mathbf{Z} = \text{LayerNorm}\left[\mathbf{Y}(\mathbf{X})+\mathbf{X}\right]{{< /katex >}} | layer normalization and residual connection
{{< katex >}}\mathbf{X^*} = \text{LayerNorm}\left[\text{MLP}(\mathbf{Z})+\mathbf{Z}\right]{{< /katex >}} | MLP layer

## References

[^nmt_iclr2015]: [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473), D. Bahdanau, K. Cho, Y. Bengio, ICLR 2015.

[^attention_iclr2015]: [Attention Is All You Need](https://arxiv.org/abs/1706.03762), A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, I. Polosukhin, NeurIPS 2017.
