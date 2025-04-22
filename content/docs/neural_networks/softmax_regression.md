---
title: Softmax Regression
weight: 3
bookToc: true
---

## Softmax Regression


Softmax Regression is a single layer neural network for multi-class classification.

{{<mermaid>}}
stateDiagram-v2
    direction LR
    x1: $$x_1$$
    x2: $$x_2$$
    x3: $$x_3$$
    xd: $$x_d$$
    a1: $$a_1=\sum_i w_{1i} x_i$$
    a2: $$a_2=\sum_i w_{2i} x_i$$
    ak: $$a_k=\sum_i w_{ki} x_i$$
    z1: $$z_1=\text{softmax}(\mathbf{a})_1$$
    z2: $$z_2=\text{softmax}(\mathbf{a})_2$$
    zk: $$z_k=\text{softmax}(\mathbf{a})_k$$
    x1 --> a1:$$w_{11}$$
    x2 --> a1:$$w_{12}$$
    x3 --> a1:$$w_{13}$$
    xd --> a1:$$w_{1d}$$
    x1 --> a2:$$w_{21}$$
    x2 --> a2:$$w_{22}$$
    x3 --> a2:$$w_{23}$$
    xd --> a2:$$w_{2d}$$
    x1 --> ak:$$w_{k1}$$
    x2 --> ak:$$w_{k2}$$
    x3 --> ak:$$w_{k3}$$
    xd --> ak:$$w_{kd}$$
    a1 --> z1
    a2 --> z2
    ak --> zk
    z1 --> END1:::hidden
    z2 --> END2:::hidden
    zk --> END:::hidden
    note left of xd : Inputs
    note right of a1 : Pre-activations
    note left of zk : Activations
    note left of END : Outputs
    classDef hidden display: none;
{{</mermaid>}}

## Statistical model

Given {{< katex >}}k{{< /katex >}} classes the probability of class {{< katex >}}i{{< /katex >}} for a given feature vector ({{< katex >}}\mathbf{x}\in \mathbb{R}^d{{< /katex >}}) is given by
{{< katex display=true >}}
\text{Pr}[y=i|\mathbf{x},(\mathbf{w}_1,\mathbf{w}_2,...,\mathbf{w}_k)] = \frac{\exp(\mathbf{w}_i^T\mathbf{x})}{\sum_{j=1}^{k} \exp(\mathbf{w}_j^T\mathbf{x})}
{{< /katex >}}
where {{< katex >}}\mathbf{w}_1,\mathbf{w}_2,...,\mathbf{w}_k\in \mathbb{R}^d{{< /katex >}} are the weight vector or **parameters** of the model for each class.

Stacking the weights vectors {{< katex >}}\mathbf{w}_1,\mathbf{w}_2,...,\mathbf{w}_k\in \mathbb{R}^d{{< /katex >}} into a **weight matrix** {{< katex >}}\mathbf{W} \in \mathbb{R}^{d \times k}{{< /katex >}} we can write the **pre-activation** vector {{< katex >}}\mathbf{a} \in \mathbb{R}^{k}{{< /katex >}} as follows.
{{< katex display=true >}}
\mathbf{a} = \mathbf{W}^T\mathbf{x}
{{< /katex >}}
The **activation** vector {{< katex >}}\mathbf{z} \in \mathbb{R}^{k}{{< /katex >}} is given by
{{< katex display=true >}}
\mathbf{z} = \text{softmax}(\mathbf{a})
{{< /katex >}}
and the **softmax** activation function is defined as
{{< katex display=true >}}
\text{softmax}(\mathbf{a})_i = \frac{\exp(\mathbf{a}_i)}{\sum_{j=1}^{k} \exp(\mathbf{a}_j)}
{{< /katex >}}
Hence
{{< katex display=true >}}
\text{Pr}[y=i|\mathbf{x},\mathbf{W}] = \text{softmax}(\mathbf{W}^T\mathbf{x})_i
{{< /katex >}}
We often stack all the {{< katex >}}n{{< /katex >}} examples into a *design matrix* {{< katex >}}\mathbf{X} \in \mathbb{R^{n \times d}}{{< /katex >}}, where each row is one instance. The predictions for all the {{< katex >}}n{{< /katex >}} instances {{< katex >}}\mathbf{y} \in \mathbb{R}^{n \times K }{{< /katex >}} can be written conveniently as a matrix-vector product.
{{< katex display=true >}}
\mathbf{y} = \text{softmax}(\mathbf{X}\mathbf{W})
{{< /katex >}}

## Likelihood

Given a dataset {{< katex >}}\mathcal{D}=\{\mathbf{x}_i \in \mathbb{R}^d,\mathbf{y}_i \in [0,1]\}_{i=1}^N{{< /katex >}} containing {{< katex >}}n{{< /katex >}} examples we need to estimate the parameter vector {{< katex >}}\mathbf{w}{{< /katex >}} by maximizing the likelihood of data.

> In practice we minimize the **negative log likelihood**.

Let {{< katex >}} \mu_i = \text{Pr}[y_i=1|\mathbf{x}_i,\mathbf{w}] = \sigma(\mathbf{w}^T\mathbf{x}_i){{< /katex >}} be the model prediction for each example in the training dataset. The the negative log likelihood (NLL) is given by
{{< katex display=true >}}
\begin{align}
L(\mathbf{w}) &= - \sum_{i=1}^{N} \log\left[\mu_i^{y_i}(1-\mu_i)^{1-y_i}\right] \nonumber \\
                       &= - \sum_{i=1}^{N} \left[ y_i\log(\mu_i) + (1-y_i)\log(1-\mu_i) \right] \nonumber \\

\end{align}
{{< /katex >}}
This is referred to as the **Binary Cross Entropy** (BCE) loss. We need to choose the model parameters that optimizes (minimizes) the loss function.
{{< katex display=true >}}
\hat{\mathbf{w}} = \argmin_{\mathbf{w}} L(\mathbf{w})
{{< /katex >}}

## Loss function

**Binary Cross Entropy** (BCE) loss
{{< katex display=true >}}
L(\mathbf{w}) - \sum_{i=1}^{N} \left[ y_i\log(\mu_i) + (1-y_i)\log(1-\mu_i) \right]
{{< /katex >}}

{{<button href="https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html#torch.nn.BCELoss">}}torch.nn.BCELoss{{</button>}} {{<button href="https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html#torch.nn.BCEWithLogitsLoss">}}torch.nn.BCEWithLogitsLoss{{</button>}}


## Gradient
The gradient of the loss function if given by
{{< katex display=true >}}
\nabla_{\mathbf{w}} L(\mathbf{w}) = \mathbf{X}^T\left(\sigma(\mathbf{X}\mathbf{w})-\mathbf{y}\right)
{{< /katex >}}

