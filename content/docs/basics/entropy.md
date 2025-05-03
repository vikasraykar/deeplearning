---
title: Entropy
weight: 1
bookToc: true
---

> A brief primer on entropy, cross-entropy and perplexity.

##  Entropy
{{< katex >}}{{< /katex >}}The **entropy** of a discrete random variable {{< katex >}}X{{< /katex >}} with {{< katex >}}K{{< /katex >}} states/categories with distribution {{< katex >}}p_k = \text{Pr}(X=k){{< /katex >}} for {{< katex >}}k=1,...,K{{< /katex >}}  is a measure of uncertainty and is defined as follows.
{{< katex display=true >}}H(X) = \sum_{k=1}^{K} p_k \log_2 \frac{1}{p_k} = - \sum_{k=1}^{K} p_k \log_2 p_k {{< /katex >}}
{{< katex >}}{{< /katex >}}
The term {{< katex >}}\log_2\frac{1}{p}{{< /katex >}} quantifies the notion or surprise or uncertainty and hence entropy is the average uncertainty.

The unit is bits ({{< katex >}}\in [0,\log_2 K]{{< /katex >}}) (or nats incase of natural log).

The discrete distribution with maximum entropy ({{< katex >}}\log_2 K{{< /katex >}}) is uniform.

The discrete distribution with minimum entropy ({{< katex >}}0{{< /katex >}}) is any delta function which puts all mass on one state/category.

{{% hint info %}}
[Prediction and Entropy of Printed English](https://www.princeton.edu/~wbialek/rome/refs/shannon_51.pdf), C. E. Shannon, 1950.
{{% /hint %}}


## Binary entropy

{{< katex >}}{{< /katex >}}For a binary random variable {{< katex >}}X \in {0,1}{{< /katex >}} with {{< katex >}}\text{Pr}(X=1) = \theta{{< /katex >}} and {{< katex >}}\text{Pr}(X=0) = 1-\theta{{< /katex >}} the entropy is as follows.

{{< katex display=true >}}H(\theta) = - [ \theta \log_2 \theta + (1-\theta) \log_2 (1-\theta) ] {{< /katex >}}

The range is {{< katex >}}H(\theta) \in [0,1]{{< /katex >}} and is maximum when {{< katex >}}\theta=0.5{{< /katex >}}.

## Cross entropy

{{< katex >}}{{< /katex >}}Cross entropy is the average number of bits needed to encode the data from from a source {{< katex >}}p{{< /katex >}} when we model it using {{< katex >}}q{{< /katex >}}.

{{< katex display=true >}}H(p,q) = - \sum_{k=1}^{K} p_k \log_2 q_k {{< /katex >}}

## Perplexity

{{< katex display=true >}}\text{PPL}(p,q) = 2^{H(p,q)}{{< /katex >}}

{{< katex display=true >}}\text{PPL}(p,q) = e^{H(p,q)}{{< /katex >}}

## KL Divergence

The **Kullback-Leibler** (KL) divergence or **relative entropy** measures the dissimilarity between two probability distributions {{< katex >}}p{{< /katex >}} and {{< katex >}}q{{< /katex >}}.

{{< katex display=true >}}KL(p,q) = \sum_{k=1}^{K} p_k \log_2 \frac{p_k}{q_k}{{< /katex >}}

{{< katex display=true >}}KL(p,q) = H(p,q) - H(p,p) \geq 0{{< /katex >}}

## Mutual Information

{{< katex >}}I(X,Y) = KL(P(X,Y)\|P(X)P(Y)){{< /katex >}}
