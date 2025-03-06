---
title: Gradient Descent
weight: 3
bookToc: true
---

## Gradient Descent

> Steepest descent.

Let {{< katex >}}\mathbf{w}{{< /katex >}} be a vector of all the parameters for a model.

Let {{< katex >}}L(\mathbf{w}){{< /katex >}} be the loss function (or error function).

We need to choose the model parameters that optimizes (minimizes) the loss function.

{{< katex display=true >}}
\hat{\mathbf{w}} = \argmin_{\mathbf{w}} L(\mathbf{w})
{{< /katex >}}

Let {{< katex >}}\nabla L(\mathbf{w}){{< /katex >}} be the **gradient vector**, where each element is the partial derivative of the loss fucntion wrt each parameter.

The gradient vector points in the direction of the greatest rate of increase of the loss function.

So to mimimize the loss function we take small steps in the direction of {{< katex >}}-\nabla L(\mathbf{w}){{< /katex >}}.

At the mimimum {{< katex >}}\nabla L(\mathbf{w})=0{{< /katex >}}.

{{< katex >}}\nabla L(\mathbf{w})=0{{< /katex >}}.

{{% details "Stationary points" %}}
{{< katex >}}\nabla L(\mathbf{w})=0{{< /katex >}} are knows as stationary points, which can be either a minima, maxima or a saddle point. The necessary and sufficient condition for a local minima is

1. The gradient of the loss function should be zero.
1. The Hessian matrix should be positive definite.
{{% /details %}}

{{% hint info %}}
For now we will assume the gradient is given. For deep neural networks the gradient can be computed efficiently via [**backpropagation**](/docs/training/backpropagation/) (which we will revisit later).
{{% /hint %}}


### Batch Gradient Descent

We take a small step in the direction of the **negative gradient**.

{{< katex display=true >}}
\mathbf{w}^t \leftarrow \mathbf{w}^{t-1} - \eta \nabla L(\mathbf{w}^{t-1})
{{< /katex >}}

The parameter {{< katex >}}\eta > 0{{< /katex >}} is called the **learning rate** and determines the step size at each iteration.

This update is repeated multiple times (till covergence).
```python
for epoch in range(n_epochs):
  dw = gradient(loss, data, w)
  w = w - lr * dw
```

Each step requires that the **entire training data** be processed to compute the gradient {{< katex >}}\nabla L(\mathbf{w}^{t-1}){{< /katex >}}. For large datasets this is not comptationally efficient.


### Stochastic Gradient Descent

In general most loss functions can be written as sum over each training instance.
{{< katex display=true >}}
L(\mathbf{w}) = \sum_{i=1}^{N} L_i(\mathbf{w})
{{< /katex >}}

In Stochastic Gradient Descent (SGD) we update the parameters **one data point at a time**.
{{< katex display=true >}}
\mathbf{w}^t \leftarrow \mathbf{w}^{t-1} - \eta \nabla L_i(\mathbf{w}^{t-1})
{{< /katex >}}

> A complete passthrough of the whole dataset is called an **epoch**. Training is done for multiple epochs depending on the size of the dataset.

```python
for epoch in range(n_epochs):
  for i in range(n_data):
    dw = gradient(loss, data[i], w)
    w = w - lr * dw
```

- SGD is much faster and more computationally efficient, but it has noise in the estimation of the gradient.
- Since it updates the weight frequently, it can lead to big oscillations and that makes the training process highly unstable.


{{% hint warning %}}
Bottou, L. (2010). [Large-Scale Machine Learning with Stochastic Gradient Descent](https://leon.bottou.org/publications/pdf/compstat-2010.pdf). In: Lechevallier, Y., Saporta, G. (eds) Proceedings of COMPSTAT'2010. Physica-Verlag HD.
{{% /hint %}}


### Mini-batch Stochastic Gradient Descent

Using a single example results in a very noisy estimate of the gradient. So we use a small random subset of data called **mini-batch** of size B (**batch size**) to compute the gradient.

{{< katex display=true >}}
\mathbf{w}^t \leftarrow \mathbf{w}^{t-1} - \eta \nabla L_{batch}(\mathbf{w}^{t-1})
{{< /katex >}}


```python
for epoch in range(n_epochs):
  for mini_batch in get_batches(data, batch_size):
    dw = gradient(loss, mini_batch, w)
    w = w - lr * dw
```
{{<button href="https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD">}}PyTorch{{</button>}}
```python
optimizer = optim.SGD(model.parameters(), lr=1e-3)
```

{{% hint info %}}
Mini-batch SGD is the most commonly used method and is sometimes refered to as just SGD.
- Typical choices of the batch size are B=32,64,128,256,..
- In practice we do a random shuffle of the data per epoch.

In practice, mini-batch SGD is the most frequently used variation because it is both computationally cheap and results in more robust convergence.
{{% /hint %}}



## Adding momentum
One of the basic improvements over SGD comes from adding a **momentum** term.

At every time step, we update **velocity** by decaying the previous velocity by a factor of {{< katex >}}0 \leq \mu \leq 1{{< /katex >}} (called the **momentum** parameter) and adding the current gradient update.
{{< katex display=true >}}
\mathbf{v}^{t-1} \leftarrow \mu \mathbf{v}^{t-2} - \eta \nabla L(\mathbf{w}^{t-1})
{{< /katex >}}
Then, we update our weights in the direction of the velocity vector.
{{< katex display=true >}}
\mathbf{w}^t \leftarrow \mathbf{w}^{t-1} + \mathbf{v}^{t-1}
{{< /katex >}}

```python
for epoch in range(n_epochs):
  for mini_batch in get_batches(data, batch_size):
    dw = gradient(loss, mini_batch, w) # gradient
    v = momentum * v - lr * dw # velocity
    w = w + v
```

{{<button href="https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD">}}PyTorch{{</button>}}
```python
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

{{% hint info %}}
We now have two hyper-parameters learnign rate and momentum. Typically we set the momentum parameter to 0.9.
{{% /hint %}}

{{% details "Effective learning rate" %}}
{{< katex >}}{{< /katex >}} One interpretation of momentum to increase the effective learning rate from {{< katex >}}\eta{{< /katex >}} to {{< katex >}}\frac{\eta}{(1-\mu)}{{< /katex >}}. If we make the approximation that the gradient is unchanging then
{{< katex display=true >}}
 -\eta \nabla L \{1+\mu+\mu^2+...\} = - \frac{\eta}{1-\mu} \nabla L
{{< /katex >}} By contrast, in a region of high curvature in which gradient descent is oscillatory, successive contributions from the momentum term will tend to cancel and effective learning rate will be close to {{< katex >}}\eta{{< /katex >}}.
{{% /details %}}





- We can now escape local minima or saddle points because we keep moving downwards even though the gradient of the mini-batch might be zero.
- Momentum can also help us reduce the oscillation of the gradients because the velocity vectors can smooth out these highly changing landscapes.
- It reduces the noise of the gradients and follows a more direct walk down the landscape.


{{% hint warning %}}
Ilya Sutskever, James Martens, George Dahl, and Geoffrey Hinton. 2013. [On the importance of initialization and momentum in deep learning.](https://dl.acm.org/doi/10.5555/3042817.3043064) In Proceedings of the 30th International Conference on International Conference on Machine Learning - Volume 28 (ICML'13). JMLR.org, III–1139–III–1147.
{{% /hint %}}






## Adaptive Learning Rates

### Adagrad

### RMSProp

### Adam

## Learning rate schedule

## Learning curve

## Training loop

![image](https://cdn-images-1.medium.com/v2/resize:fit:1000/1*Nb39bHHUWGXqgisr2WcLGQ.gif)
