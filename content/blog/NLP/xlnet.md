---
title: 'XLNet: Generalized Autoregressive Pretraining for Language Understanding'
date: 2019-06-01
category: 'NLP'
draft: false
---

> **XLNet: Generalized Autoregressive Pretraining for Language Understanding**  
Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, Quoc V. Le  
https://arxiv.org/abs/1906.08237

# 1. Introduction
Unsupervised Representation Learning은 Large-scale의 corpora를 통해 Pre-train하고, Downstream task에 대해 fine-tune을 수행함. 이러한 Unsupervised Representation Learning은 NLP Domain에서 좋은 성능을 보여주었으며 이에 관한 연구 중 Autoregressive Language(AR) Modeling과 Auto-encoding(AE)이 가장 좋은 결과를 보여줌.  

AR language modeling은 previous token을 통해 sentence의 probability distribution을 추정.
$$
p(X)=p({ x }_{ 1 })p({ x }_{ 2 }|{ x }_{ 1 }),\cdots ,p({ x }_{ 1 },\cdots ,{ x }_{ T-1 })
$$ 
즉, AR language modeling은 현재 token을 계산하기 위해 $1$부터  $T-1$까지의 token에 대하여 conditional probability를 통하여 probability distribution을 추정하기 때문에 양방향 context를 modeling하는데는 적합하지 않다. AR language modeling은 density distribution을 estimation하면서 language modeling을 수행하는데 이에 반해 BERT와 같은 AE기반의 language modeling은 오류가 있는 데이터로 부터 원본 데이터를 재구성하는 것에 목표로 한다. BERT는 input sequence에서 특정 부분에 $[MASK]$를 적용하여 input sequence를 복원하도록 학습하는데, bidirectional context를 사용하기 때문에 성능을 향상시킬 수 있었다. 그러나 pre-training에 사용하는 $[MASK]$가 fine-tune 단계에서는 사용되지 않아 pre-train과 fine-tune간에 불일치 문제가 존재한다. 또한, BERT는 predicted token이 input sequence에서 mask처리되어 있기 때문에 AR language modeling에서 곱의 규칙을 통해 얻을 수 있는 joint probability를 계산할 수 없다. joint probability를 사용하지 않는다는 것은 input sequence로 부터 predicted token들이 서로 독립적이라고 추정하게되며, 이는 고차원(high-order), 장거리 의존성(long-term dependency)이 있는 자연어에서 지나치게 단순화 된다고 볼 수 있다.  

이를 통해 본 연구에서는 기존의 language pre-training의 장단점에 직면하여 AR language modeling과 AE를 최대한 활용하는 generalized autoregressive method를 제안한다.
* 기존의 AR model들은 previous tokens를 통해 modeling을 하였지만 XLNet은 인수분해 순서의 모든 순열에 대해 log-likelihood를 최대화 할 수 있게 학습한다. 이를 통해 각 위치의 context는 왼쪽과 오른쪽의 token으로 구성될 수 있으며 각 위치는 양방향 문맥을 capture하여 학습한다.
* XLNet은 generalized AR language model이기 때문에 pre-train과 fine-tune간에 차이가 존재하지 않는다. 곱의 법칙을 통해 token들의 joint probability를 계산하기 때문에 BERT에서 발생한 token들을 독립적으로 추정하게 되는 문제를 발생시키지 않는다.
* 추가적으로 XLNet은 pre-training을 위한 architecture design을 개선한다.
    1. **segment recurrence mechanism**과 Transformer-XL의 **relative encoding scheme**을 pre-training에 적용하여 길이가 긴 sequence에 대해서 성능을 향상시킨다.
    2. XLNet과 같은 permutation-based language modeling에 Transformer-XL architecture를 naive하게 적용하는 것은 어렵기 때문에 Transformer-XL network를 reparameterize(모델 재구성?)하는 방법을 제안한다.