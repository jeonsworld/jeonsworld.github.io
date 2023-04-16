---
title: 'SPDF: Sparse Pre-training and Dense Fine-tuning for Large Language Models'
date: 2019-09-02
category: 'NLP'
draft: false
---

> **SPDF: Sparse Pre-training and Dense Fine-tuning for Large Language Models**
Vithursan Thangarasa, Abhay Gupta, William Marshall, Tianda Li, Kevin Leong, Dennis DeCoste, Sean Lie, Shreyas Saxena
https://arxiv.org/abs/2303.10464

최근 NLP에서는 Pre-training과 Fine-tuning 패러다임이 사용되는데 이는 직접 downstream task에 훈련시키는 것이 아니라, 큰 데이터셋에서 Cross-domain knowledge를 이용해 먼저 언어 모델을 pre-training하고, 이후 task-specific 데이터에서 fine-tuning을 수행한다.
하지만, 모델과 데이터셋의 크기를 확장하면 성능 향상은 있지만 연산 비용도 크게 증가한다. 따라서 본 논문에서는 unstructured weight sparsity를 이용해 모델의 capacity를 pre-training과 fine-tuning 단계에서 분리하여 Sparse Pre-training and Dense Fine-tuning (SPDF)을 제안.
이를 통해, 1.3B parameter GPT-3 XL 모델에서 75% sparsity를 도출해 pre-training FLOPs를 2.5배 줄일 수 있었으며, downstream task의 정확도에도 큰 손실 없이 기존 dense 모델과 비슷한 결과를 보였줌.

![fig1](./img/spdf/fig1.png)
![table1](./img/spdf/table1.png)
![table2](./img/spdf/table2.png)
