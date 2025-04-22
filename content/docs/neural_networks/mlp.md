---
title: Multilayer perceptron
weight: 4
bookToc: true
---

## Multilayer perceptron

A 3-layer multilayer perceptron.
{{< katex display=true >}}
\begin{align}
\mathbf{X}  &= \mathbf{X}  \nonumber \\
\mathbf{H}^{(1)} &= g_1\left(\mathbf{X}\mathbf{W}^{(1)}+\mathbf{b}^{(1)}\right) \nonumber \\
\mathbf{H}^{(2)} &= g_2\left(\mathbf{H}^{(1)}\mathbf{W}^{(2)}+\mathbf{b}^{(2)}\right) \nonumber \\
\mathbf{O} &= \mathbf{H}^{(2)}\mathbf{W}^{(3)}+\mathbf{b}^{(3)} \nonumber \\
\end{align}
{{< /katex >}}

{{< katex >}}g{{< /katex>}} is a nonlinear **activation function**

{{<mermaid>}}
stateDiagram-v2
    direction LR
    x1: $$x_1$$
    x2: $$x_2$$
    x3: $$x_3$$
    h11: $$h^1_1$$
    h12: $$h^1_2$$
    h13: $$h^1_3$$
    h14: $$h^1_4$$
    h21: $$h^2_1$$
    h22: $$h^2_2$$
    h23: $$h^2_3$$
    h24: $$h^2_4$$
    h25: $$h^2_5$$
    o1: $$o_1$$
    o2: $$o_2$$
    o3: $$o_3$$
    x1 --> h11
    x1 --> h12
    x1 --> h13
    x1 --> h14
    x2 --> h11
    x2 --> h12
    x2 --> h13
    x2 --> h14
    x3 --> h11
    x3 --> h12
    x3 --> h13
    x3 --> h14
    h11 --> h21
    h11 --> h22
    h11 --> h23
    h11 --> h24
    h11 --> h25
    h12 --> h21
    h12 --> h22
    h12 --> h23
    h12 --> h24
    h12 --> h25
    h13 --> h21
    h13 --> h22
    h13 --> h23
    h13 --> h24
    h13 --> h25
    h14 --> h21
    h14 --> h22
    h14 --> h23
    h14 --> h24
    h14 --> h25
    h21 --> o1
    h22 --> o1
    h23 --> o1
    h24 --> o1
    h25 --> o1
    h21 --> o2
    h22 --> o2
    h23 --> o2
    h24 --> o2
    h25 --> o2
    h21 --> o3
    h22 --> o3
    h23 --> o3
    h24 --> o3
    h25 --> o3
    o1 --> END1:::hidden
    o2 --> END2:::hidden
    o3 --> END3:::hidden
    note left of x3 : Input layer
    note left of h14 : Hidden layer 1
    note left of h25 : Hidden layer 2
    note left of o1 : Output layer
    classDef hidden display: none;
{{</mermaid>}}

