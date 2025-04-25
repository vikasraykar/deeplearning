---
title: Backpropagation
weight: 2
bookToc: true
---

# Backpropagation

> Backpropagation, Error Backpropagation, Backprop.

Backpropagation (or backprop) is an efficient technique to compute the gradient of the loss function.

It boils down to a local message passing scheme in which information is sent backwards through the network.

## Forward propagation

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


Let's consider a hidden unit in a general feed forward neural network.
{{< katex display=true >}}
a_j=\sum_i w_{ji} z_i
{{< /katex >}}
where {{< katex >}}z_i{{< /katex >}} is the activation of another unit or an input that sends an connection of unit {{< katex >}}j{{< /katex >}} and {{< katex >}}w_{ji}{{< /katex >}} is the weight associated with that connection. The sum {{< katex >}}a_j{{< /katex >}} is known as **pre-activation** and is transformed by a non-linear activation function {{< katex >}}h(){{< /katex >}} to give the **activation** {{< katex >}}z_j{{< /katex >}} of unit {{< katex >}}j{{< /katex >}}.
{{< katex display=true >}}
z_j=h(a_j)
{{< /katex >}}
For any given data point in the training set, we can pass the input and compute the activations of all the hidden and output units. This process is called **forward propagation** since it is the forward flow of information through the network.

## Backward propagation
In general most loss functions can be written as sum over each training instance.
{{< katex display=true >}}
L(\mathbf{w}) = \sum_{n=1}^{N} L_n(\mathbf{w})
{{< /katex >}}
Hence we will consider evaluating the gradient of {{< katex >}}L_n(\mathbf{w}){{< /katex >}} with respect to the weight parameters {{< katex >}}w_{ji}{{< /katex >}}. We will now use chain rule to derive the gradient of the loss function.
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
> The required derivative for {{< katex >}}w_{ij}{{< /katex >}} is simply obtained by multiplying the value of {{< katex >}}\delta_j{{< /katex >}} for the unit at the output end of the weight by the value of {{< katex >}}z_i{{< /katex >}} for the unit at the input end of the weight. This can be seen as a **local computation** involving the **error signal** at the output end with the **activation signal** at the input end.

{{<mermaid>}}
stateDiagram-v2
    direction LR
    zi: $$z_i$$
    zj: $$z_j$$
    zi --> zj:$$w_{ji}$$
    note left of zi : $$\delta_i$$
    note right of zj : $$\delta_j$$
{{</mermaid>}}

So this now boils down to computing {{< katex >}}\delta_j{{< /katex >}}  for all the hidden and the output units.
{{< katex >}}\delta{{< /katex >}} for the output units are based on the loss function. For example for the MSE loss
{{< katex display=true >}}
\delta_k = y_{nk} - t_{nk}
{{< /katex >}}
To evaluate the {{< katex >}}\delta{{< /katex >}} for the hidden units we again make use of the the chain rule for partial derivatives.
{{< katex display=true >}}
\delta_j := \frac{\partial L_n}{\partial a_{j}} = \sum_{k} \frac{\partial L_n}{\partial a_{k}} \frac{\partial a_k}{\partial a_{j}}
{{< /katex >}}
where the sum runs over all the units k to which j sends connections. This gives rise to the final **backpropagation formula**
{{< katex display=true >}}
\delta_j = h^{'}(a_j)\sum_{k} w_{kj} \delta_k
{{< /katex >}}
{{<mermaid>}}
stateDiagram-v2
    direction LR
    zi: $$z_i$$
    zj: $$z_j$$
    z1: $$z_1$$
    zk: $$z_k$$
    zi --> zj:$$w_{ji}$$
    zj --> zk:$$w_{kj}$$
    zj --> z1:$$w_{1j}$$
    zk --> zj
    z1 --> zj
    note right of zj : $$\delta_j$$
    note right of zk : $$\delta_k$$
    note right of z1 : $$\delta_1$$
{{</mermaid>}}
This tells us that the value of {{< katex >}}\delta{{< /katex >}} for a particular hidden unit can be obtained by propagating the {{< katex >}}\delta{{< /katex >}} backward from units higher up in the network.

## Algorithm

{{% columns %}}
**Forward propagation**

For all hidden and output units compute in **forward order**

{{< katex display=true >}}
a_j \leftarrow \sum_i w_{ji} z_i
{{< /katex >}}
{{< katex display=true >}}
z_j \leftarrow h(a_j)
{{< /katex >}}

<--->

**Error evaluation**

For all output units compute

{{< katex display=true >}}
\delta_k \leftarrow \frac{\partial L_n}{\partial a_k}
{{< /katex >}}

<--->

**Backward propagation**

For all hidden units compute in **reverse order**

{{< katex display=true >}}
\delta_j \leftarrow h^{'}(a_j)\sum_{k} w_{kj} \delta_k
{{< /katex >}}
{{< katex display=true >}}
\frac{\partial L_n}{\partial w_{ji}} \leftarrow \delta_j z_i
{{< /katex >}}

{{% /columns %}}

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


