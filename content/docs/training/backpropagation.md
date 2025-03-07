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

## Algorithmic differenciation

### Forward mode

### Reverse mode
