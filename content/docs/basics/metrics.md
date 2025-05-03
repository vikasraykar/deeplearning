---
title: Metrics
weight: 2
bookToc: true
---

> Some core evaluation metrics. To be revised.

## Binary Classification

### True Positive Rate

> Recall, Sensitivity.

{{< katex display=true >}}\text{TPR} = \text{Pr}[y_{pred}=1|y=1] = \frac{TP}{TP+FN} = \frac{TP}{N_{+}}{{< /katex >}}

### False Positive Rate

> 1-Specificity

{{< katex display=true >}}\text{FPR} = \text{Pr}[y_{pred}=1|y=0] = \frac{FP}{FP+TN} = \frac{FP}{N_{-}}{{< /katex >}}

### ROC Curve

x-axis is FPR and y-axis is TPR

### AUC

Area under the ROC Curve.

{{< katex display=true >}}\text{AUC}=\text{Pr}[y^{+} > y^{-}]{{< /katex >}}

### Accuracy

{{< katex display=true >}}\text{Accuracy}=\frac{TP+TN}{TP+FP+TN+FN}{{< /katex >}}

### Precision

Precision is the fraction of predicted positives that are actually positive.

{{< katex display=true >}}\text{Precision} = \text{Pr}[y=1|y_{pred}=1] = \frac{TP}{TP+FP}{{< /katex >}}

### Recall

Precision is the fraction of positives that are correctly predicted as positive.

{{< katex display=true >}}\text{Recall} = \text{Pr}[y_{pred}=1|y=1] = \frac{TP}{TP+FN}{{< /katex >}}

### F1 Score

F1 Score is the harmonic mean of precision and recall.

{{< katex display=true >}}F1=2 \times \frac{\text{Precision}\times\text{Recall}}{\text{Precision}+\text{Recall}}{{< /katex  >}}

### PR Curve

## Retrieval

### Precision@k

### Average Precision

### Mean Average Precision


### MRR

Mean Reciprocal Rank (MRR) is the average of the reciprocal ranks of the ground truth contexts in the retrieved contexts.

{{< katex display=true >}}MRR = \frac{1}{Q} \sum_{i=1}^{Q} \frac{1}{\text{rank}_i}{{< /katex >}}

### nDCG

Normalized Cumulative Discounted Gain is the ratio of the Discounted Cumulative Gain and the ideal DCG.

DCG is the sum of the discounted relevance scores of the ground truth contexts.

{{< katex display=true >}}\text{DCG}=\sum_{i}^{N} \frac{\text{rel}_i}{\log_2(i+1)}{{< /katex >}}

> For a query given the set of **relevant offers** and the **retrieved offers** we would like to compute the following offline metrics. The metrics are then average over the set of Q queries.

## Recall

For a given query q, **recall(q)** is the fraction of the offers that are relevant to the query that are successfully retrieved.

{{< katex display=true >}}\text{recall(q)} = \frac{|\{\text{relevant offers for query q}\} \cap \{\text{retrieved offers for query q}\}|}{|\{\text{relevant offers for query q}\}|}{{< /katex >}}

It can also be evaluated considering only the top-k results returned by the system using **recall@k**.

## Precision

For a given query q, **precision(q)** is the fraction of the offers retrieved that are relevant to the user's query.

{{< katex display=true >}}\text{precision(q)} = \frac{|\{\text{relevant offers for query q}\} \cap \{\text{retrieved offers for query q}\}|}{|\{\text{retrieved offers for query q}\}|}{{< /katex >}}

It can also be evaluated considering only the top-k results returned by the system using **precision@k**.

## PR Curve

By computing the precision and recall at every position in the ranked sequence of documents (by varying {{< katex >}}k=1,...,n{{< /katex >}} where {{< katex >}}n{{< /katex >}} is the total number of retrieved offers), one can plot the Precision-Recall (PR) Curve by plotting precision {{< katex >}}p(r){{< /katex >}} on the y-axis as a function of recall {{< katex >}}r{{< /katex >}} on the x-axis.

## AveP - Average Precision

Average precision computes the average value of {{< katex >}}p(r){{< /katex >}} over the interval from {{< katex >}}r=0{{< /katex >}} to {{< katex >}}r=1{{< /katex >}}. This essentially the Area under the PR Curve (PR AUC) computes as
{{< katex display=true >}}\text{AveP}=\sum_{k=1}^{n}p(k)\Delta r(k){{< /katex >}}
where {{< katex >}}k{{< /katex >}} is the rank in the sequence of retrieved offers, {{< katex >}}n{{< /katex >}} is the number of retrieved offers, {{< katex >}}p(k){{< /katex >}} is the precision at cut-off {{< katex >}}k{{< /katex >}} in the list, and {{< katex >}}\Delta r(k){{< /katex >}} is the change in recall from items {{< katex >}}k-1{{< /katex >}} to {{< katex >}}k{{< /katex >}}.

## MAP - Mean Average Precision

Mean average precision (MAP) for a set of queries is the mean of the average precision scores for each query.

{{< katex display=true >}}MAP=\frac{\sum_{q=1}^{Q} \text{AveP}(q)}{Q}{{< /katex >}}

