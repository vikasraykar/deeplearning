---
weight: 2
bookFlatSection: true
title: "Neural Networks"
---

## Single Layer Networks

> For simplicity for this chapter we will mainly introduce single layer networks (for regression and classification).

{{<mermaid>}}
stateDiagram-v2
    direction LR
    z1: $$x_1$$
    z2: $$x_2$$
    zi: $$x_i$$
    zM: $$x_d$$
    aj: $$a=\sum_i w_{i} x_i$$
    zj: $$z=h(a)$$
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
