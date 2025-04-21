---
title: Logistic Regression
weight: 2
bookToc: true
---

## Logistic Regression


Logistic Regression is a single layer neural network for binary classification.

{{<mermaid>}}
stateDiagram-v2
    direction LR
    z1: $$x_1$$
    z2: $$x_2$$
    zi: $$x_i$$
    zM: $$x_d$$
    aj: $$a=\sum_i w_{i} x_i$$
    zj: $$z=\sigma(a)$$
    z1 --> aj:$$w_{1}$$
    z2 --> aj:$$w_{2}$$
    zi --> aj:$$w_{i}$$
    zM --> aj:$$w_{d}$$
    aj --> zj
    zj --> END:::hidden
    note left of zM : Inputs
    note left of aj : Pre-activation
    note left of zj : Activation
    note left of END : Output
    classDef hidden display: none;
{{</mermaid>}}

## Statistical model

The probability of the positive class ({{< katex >}}y=1{{< /katex >}}) for a given feature vector ({{< katex >}}\mathbf{x}\in \mathbb{R}^d{{< /katex >}}) is given by
{{< katex display=true >}}
\text{Pr}[y=1|\mathbf{x},\mathbf{w}] = \sigma(\mathbf{w}^T\mathbf{x})
{{< /katex >}}
where {{< katex >}}\mathbf{w}\in \mathbb{R}^d{{< /katex >}} are the weights/**parameters** of the model and {{< katex >}}\sigma{{< /katex >}} is the **sigmoid** activation function defined as
{{< katex display=true >}}
\sigma(x) = \frac{1}{1-e^{-z}}
{{< /katex >}}
{{% hint warning %}}
Without loss of generalization we ignore the bias term as it can be incorporated into the feature vector.
{{% /hint %}}
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

{{<button href="https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html#torch.nn.BCELoss">}}torch.nn.BCELoss{{</button>}} {{<button href="https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html#torch.nn.BCEWithLogitsLoss">}}torch.nn.BCEWithLogitsLoss{{</button>}}






## Loss functions

https://pytorch.org/docs/stable/nn.html#loss-functions
