---
title: Automatic Differentiation
weight: 3
bookToc: true
---

# Automatic differentiation

> Algorithmic differentiation, autodiff, autograd

There are broadly 4 appoaches to compute derivatives.

| Approach  | Pros | Cons |
| ------------- | ------------- | ------------- |
| **Manual** derivation of backprop equations. | If done carefully can result in efficient code.  | Manual process, prone to errors and not easy to iterate on models |
| **Numerical** evaluation of gradients via finite differences. | Sometimes used to check for correctness of other methods.| Limited by computational accuracy. Scales poorly with the size of the network.
| **Symbolic** differentiation using packages like `sympy` | | Closed form needed. Resulting expression can be very long (*expression swell*).|
| **Automatic differentiation** | Most preferred. |

{{% hint warning %}}
Atılım Günes Baydin, Barak A. Pearlmutter, Alexey Andreyevich Radul, and Jeffrey Mark Siskind. 2017. [Automatic differentiation in machine learning: a survey.](https://dl.acm.org/doi/pdf/10.5555/3122009.3242010) J. Mach. Learn. Res. 18, 1 (January 2017), 5595–5637.
{{% /hint %}}

## Forward-mode automatic differentiation

> We augment each intermediate variable {{< katex >}}z_i{{< /katex >}} (known as **primal** variable) with an additional variable representing the value of some derivative of that variable, which we denote as {{< katex >}}\dot{z}_i{{< /katex >}}, known as **tangent** variable. The tangent variables are generated automatically.

Consider the following function.
{{< katex display=true >}}
f(x_1,x_2) = x_1x_2 + \exp(x_1x_2) - \sin(x_2)
{{< /katex >}}
When implemented in software the code consists of a sequence of operations than can be expressed as an **evaluation trace** of the underlying elementary operations. This trace can be visualized as a computation graph with respect to the following 7 primal variables.
{{<mermaid>}}
stateDiagram-v2
    direction LR
    x1: $$x_1$$
    x2: $$x_2$$
    v1: $$v_1 = x_1$$
    v2: $$v_2 = x_2$$
    v3: $$v_3 = v_1v_2$$
    v4: $$v_4 = \sin(v_2)$$
    v5: $$v_5 = \exp(v_3)$$
    v6: $$v_6 = v_3 - v_4$$
    v7: $$v_7 = v_5 + v_6$$
    f: $$f = v_5 + v_6$$
    x1 --> v1
    x2 --> v2
    v1 --> v3
    v2 --> v4
    v2 --> v3
    v3 --> v5
    v4 --> v6
    v3 --> v6
    v5 --> v7
    v6 --> v7
    v7 --> f
{{</mermaid>}}
We first write code to implement the evaluation of the primal variables.
{{< katex display=true >}}
v_1 = x_1
{{< /katex >}}
{{< katex display=true >}}
v_2 = x_2
{{< /katex >}}
{{< katex display=true >}}
v_3 = v_1v_2
{{< /katex >}}
{{< katex display=true >}}
v_4 = \sin(v_2)
{{< /katex >}}
{{< katex display=true >}}
v_5 = \exp(v_3)
{{< /katex >}}
{{< katex display=true >}}
v_6 = v_3 - v_4
{{< /katex >}}
{{< katex display=true >}}
v_7 = v_5 + v_6
{{< /katex >}}
Now say we wish to evaluate the derivative {{< katex >}}\partial f/\partial x_1{{< /katex >}}. First we define the tangent variables by
{{< katex display=true >}}\dot{v}_i = \frac{\partial v_i}{\partial x_1}{{< /katex >}}
Expressions for evaluating these can be constructed automatically using the chain rule of calculus.
{{< katex display=true >}}
\dot{v}_i = \frac{\partial v_i}{\partial x_1} = \sum_{j\in\text{parents}(i)} \frac{\partial v_i}{\partial v_j} \frac{\partial v_j}{\partial x_1} = \sum_{j\in\text{parents}(i)} \dot{v}_j \frac{\partial v_i}{\partial v_j}
{{< /katex >}}
where {{< katex >}}\text{parents}(i){{< /katex >}} denotes the set of **parents** of node {{< katex >}}i{{< /katex >}} in the evaluation trace diagram.

The associated equations and corresponding code for evaluating the tangent variables are generated automatically.
{{< katex display=true >}}
\dot{v}_1 = 1
{{< /katex >}}
{{< katex display=true >}}
\dot{v}_2 = 0
{{< /katex >}}
{{< katex display=true >}}
\dot{v}_3 = v_1\dot{v}_2+\dot{v}_1v_2
{{< /katex >}}
{{< katex display=true >}}
\dot{v}_4 = \dot{v}_2\cos(v_2)
{{< /katex >}}
{{< katex display=true >}}
\dot{v}_5 = \dot{v}_3\exp(v_3)
{{< /katex >}}
{{< katex display=true >}}
\dot{v}_6 = \dot{v}_3 - \dot{v}_4
{{< /katex >}}
{{< katex display=true >}}
\dot{v}_7 = \dot{v}_5 + \dot{v}_6
{{< /katex >}}

To evaluate the derivative {{< katex >}}\frac{\partial f}{\partial x_1}{{< /katex >}} we input specific values of {{< katex >}}x_1{{< /katex >}} and {{< katex >}}x_2{{< /katex >}} and the code then executes the primal and tangent equations, numerically evaluating the tuples {{< katex >}}(v_i,\dot{v}_i){{< /katex >}} in **forward** order until we obtain the required derivative.

{{% hint danger %}}
The forward mode with slight modifications can handle multiple outputs in the same pass but the process has to be repeated for every parameter that we need the derivative. Since we are often in the regime of one output with millions of parameters this is not scalable for modern deep neural networks. We therefore turn to an alternative version based on the backwards flow of derivative data through the evaluation trace graph.
{{% /hint %}}

## Reverse-mode automatic differentiation

Reverse-mode automatic differentiation is a generalization of the error backpropagation procedure we discussed earlier.

As with forward mode, we augment each primal variable {{< katex >}}v_i{{< /katex >}} with an additional variable called **adjoint** variable, denoted as {{< katex >}}\bar{v}_i{{< /katex >}}.
{{< katex display=true >}}\bar{v}_i = \frac{\partial f}{\partial v_i}{{< /katex >}}
Expressions for evaluating these can be constructed automatically using the chain rule of calculus.
{{< katex display=true >}}
\bar{v}_i = \frac{\partial f}{\partial v_i} = \sum_{j\in\text{children}(i)} \frac{\partial f}{\partial v_j} \frac{\partial v_j}{\partial v_i} = \sum_{j\in\text{children}(i)} \bar{v}_j \frac{\partial v_j}{\partial v_i}
{{< /katex >}}
where {{< katex >}}\text{children}(i){{< /katex >}} denotes the set of **children** of node i in the evaluation trace diagram.

> The successive evaluation of the adjoint variables represents a flow of information backwards through the graph. For multiple parameters a single backward pass is enough. Reverse mode is more memory intensive than forward mode.

{{< katex display=true >}}
\bar{v}_7 = 1
{{< /katex >}}
{{< katex display=true >}}
\bar{v}_6 = \bar{v}_7
{{< /katex >}}
{{< katex display=true >}}
\bar{v}_5 = \bar{v}_7
{{< /katex >}}
{{< katex display=true >}}
\bar{v}_4 = -\bar{v}_6
{{< /katex >}}
{{< katex display=true >}}
\bar{v}_3 = \bar{v}_5v_5+\bar{v}_6
{{< /katex >}}
{{< katex display=true >}}
\bar{v}_2 = \bar{v}_2v_1+\bar{v}_4\cos(v_2)
{{< /katex >}}
{{< katex display=true >}}
\bar{v}_1 = \bar{v}_3v_2
{{< /katex >}}

## Autograd in pytorch

- [A Gentle Introduction to `torch.autograd`](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)
- [The Fundamentals of Autograd](https://pytorch.org/tutorials/beginner/introyt/autogradyt_tutorial.html)
