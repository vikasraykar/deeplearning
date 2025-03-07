---
title: Backpropagation
weight: 4
bookToc: true
---

## Backpropagation

> Backprop, Error Backpropagation.

Backpropagation (or backprop) is an efficient technique to compute the gradient of the loss function.

It boils down to a local message passing scheme in which information is sent backwards through the network.

### Forward propagation

{{<mermaid>}}
stateDiagram-v2
    direction LR
    z1: $$z_1$$
    z2: $$z_2$$
    zi: $$z_i$$
    zM: $$...$$
    aj: $$a_j=\sum_i w_{ji} z_i$$
    zj: $$z_j=h(a_j)$$
    START1:::hidden --> z1
    START2:::hidden --> z2
    STARTi:::hidden --> zi
    STARTM:::hidden --> zM
    z1 --> aj
    z2 --> aj
    zi --> aj:$$w_{ji}$$
    zM --> aj
    aj --> zj
    zj --> END:::hidden
    note left of aj : Pre-activation
    note left of zj : Activation
    classDef hidden display: none;
{{</mermaid>}}


Let's consider a hidden unit in a general feed forward neural nework.
{{< katex display=true >}}
a_j=\sum_i w_{ji} z_i
{{< /katex >}}
where {{< katex >}}z_i{{< /katex >}} is the activation of anoter unit or an input that sends an connection of unit {{< katex >}}j{{< /katex >}} and {{< katex >}}w_{ji}{{< /katex >}} is the weight associated with that connection. {{< katex >}}a_j{{< /katex >}} is known as **pre-activation** and is transformed by a non-linear activation fucntion to give the **activation** {{< katex >}}z_j{{< /katex >}} of unit {{< katex >}}j{{< /katex >}}.
{{< katex display=true >}}
z_j=h(a_j)
{{< /katex >}}
For any given data point in the traning set, we can pass the input and compute the activations of all the hidden and output units. This process is called **forward propagation** since it is the forward flow of information through the network.

### Backward propagation

{{< katex display=true >}}
\frac{\partial L_n}{\partial w_{ji}} = \frac{\partial L_n}{\partial a_{j}} \frac{\partial a_j}{\partial w_{ji}} = \delta_j z_i
{{< /katex >}}
where {{< katex >}}\delta_j{{< /katex >}} are referred to as **errors**
{{< katex display=true >}}
\frac{\partial L_n}{\partial a_{j}} := \delta_j
{{< /katex >}}
and
{{< katex display=true >}}
\frac{\partial a_j}{\partial w_{ji}} = z_i
{{< /katex >}}
So we now have
{{< katex display=true >}}
\frac{\partial L_n}{\partial w_{ji}} = \delta_j z_i
{{< /katex >}}
The required derivative is simply obtained by multiplying the value of {{< katex >}}\delta{{< /katex >}} for the unit at the output end of the weight by the value of {{< katex >}}z{{< /katex >}} for the unit at the input end of the weight.

{{< katex >}}\delta{{< /katex >}} for the output units are based on the losss function.

To evaluate the {{< katex >}}\delta{{< /katex >}} for the hidden units we again make use of the the chain rule for partial derivatives.
{{< katex display=true >}}
\delta_j := \frac{\partial L_n}{\partial a_{j}} = \sum_{k} \frac{\partial L_n}{\partial a_{k}} \frac{\partial a_k}{\partial a_{j}}
{{< /katex >}}
where the sum runs over all the units k to which j sends connections.
{{< katex display=true >}}
\delta_j = h^{'}(a_j)\sum_{k} w_{kj} \delta_k
{{< /katex >}}
This tells us that the value of {{< katex >}}\delta{{< /katex >}} for a particular hidden unit can be obtained by propagating the {{< katex >}}\delta{{< /katex >}} backward from uits higher up in the network.

{{<mermaid>}}
stateDiagram-v2
    direction LR
    z1: $$z_1$$
    z2: $$z_2$$
    zi: $$z_i$$
    zM: $$...$$
    delta1: $$\delta_1$$
    delta2: $$\delta_2$$
    deltak: $$\delta_k$$
    deltaM: $$...$$
    aj: $$a_j$$
    zj: $$z_j$$
    START1:::hidden --> z1
    START2:::hidden --> z2
    STARTi:::hidden --> zi
    STARTM:::hidden --> zM
    z1 --> aj
    z2 --> aj
    zi --> aj:$$w_{ji}$$
    zM --> aj
    aj --> zj
    zj --> delta1
    zj --> delta2
    zj --> deltak:$$w_{kj}$$
    zj --> deltaM
    delta1 --> zj
    delta2 --> zj
    deltak --> zj
    deltaM --> zj
    delta1 --> START11:::hidden
    delta2 --> START22:::hidden
    deltak --> STARTii:::hidden
    deltaM --> STARTMM:::hidden
    note left of aj : Pre-activation
    note left of zj : Activation
    note right of deltak : Errors
    classDef hidden display: none;
{{</mermaid>}}

{{% columns %}}
### Forward propagation
For all hidden and ouput units compute in **forward order**

{{< katex display=true >}}
a_j \leftarrow \sum_i w_{ji} z_i
{{< /katex >}}
{{< katex display=true >}}
z_j \leftarrow h(a_j)
{{< /katex >}}

<--->

### Error evaluation
For all output units compute

{{< katex display=true >}}
\delta_k \leftarrow \frac{\partial L_n}{\partial a_k}
{{< /katex >}}

<--->

### Backward propagation
For all hidden units compute in **reverse order**

{{< katex display=true >}}
\delta_j \leftarrow h^{'}(a_j)\sum_{k} w_{kj} \delta_k
{{< /katex >}}
{{< katex display=true >}}
\frac{\partial L_n}{\partial w_{ji}} \leftarrow \delta_j z_i
{{< /katex >}}

{{% /columns %}}

## Automatic differenciation

> Algorithmic differentiation, autodiff, autograd

There are broadly 4 appoaches to compute derivatives.

| Approach  | Pros | Cons |
| ------------- | ------------- | ------------- |
| **Manual** derivation of backprop equations. | If done carefully can result in efficent code.  | Manual process, prone to erros and not easy to iterate on models |
| **Numerical** evaluation of gradients via finite differences. | Sometime sused to check for correctness of other methods.| Limited by computational accuracy. Scales poorly with the size of the network.
| **Symbolic** differenciation using packages like `sympy` | | Closed form needed. Resulting expression can be very long (*expression swell*).|
| **Automatic differentiation** | Most prefered. |

{{% hint warning %}}
Atılım Günes Baydin, Barak A. Pearlmutter, Alexey Andreyevich Radul, and Jeffrey Mark Siskind. 2017. [Automatic differentiation in machine learning: a survey.](https://dl.acm.org/doi/pdf/10.5555/3122009.3242010) J. Mach. Learn. Res. 18, 1 (January 2017), 5595–5637.
{{% /hint %}}

### Forward-mode automatic differentiation

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
v7 = v_5 + v_6
{{< /katex >}}
Not say we wish to evaluate the derivative {{< katex >}}\partial f/\partial x_1{{< /katex >}}. First we define the tangent variables by
{{< katex display=true >}}\dot{v}_i = \frac{\partial v_i}{\partial x_1}{{< /katex >}}
Expressions for evaluating these can be constructed automatically using the chain rule of calculus.
{{< katex display=true >}}
\dot{v}_i = \frac{\partial v_i}{\partial x_1} = \sum_{j\in\text{parents}(i)} \frac{\partial v_i}{\partial v_j} \frac{\partial v_j}{\partial x_1} = \sum_{j\in\text{parents}(i)} \dot{v}_j \frac{\partial v_i}{\partial v_j}
{{< /katex >}}
where {{< katex >}}\text{parents}(i){{< /katex >}} denotes the set of **parents** of node i in the evaluation trace diagram.

The associated euqations and correspoding code for evaluating the tangent variables are generated automatically.
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

To evaluate the derivative {{< katex >}}\frac{\partial f}{\partial x_1}{{< /katex >}} we input specific values of {{< katex >}}x_1{{< /katex >}} and {{< katex >}}x_2{{< /katex >}} and the code then executes the primal and tangent equations, numerically evalating the tuples {{< katex >}}(v_i,\dot{v}_i){{< /katex >}} in **forward** order untill we obtain the required derivative.

{{% hint danger %}}
The forward mode with slight modifications can handle multiple outputs in the same pass but the proces has to be repeated for every parameter that we need the derivative. Since we are often in the rgime of one output with millions of parameters this is not scalable for modern deep neural networks. We therefore turn to an alternative version based on the backwards flow of derivative data through the evaluation trace graph.
{{% /hint %}}

### Reverse-mode automatic differentiation

Reverse-mode automatic differentiation is a gernalization of the error backpropagation procedure we discussed earlier.

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

### Autograd in pytorch

- [A Gentle Introduction to `torch.autograd`](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)
- [The Fundamentals of Autograd](https://pytorch.org/tutorials/beginner/introyt/autogradyt_tutorial.html)
