---
title: '[논문리뷰] An Image is Worth 16X16 Words: Transformers for Image Recognition at Scale'
date: 2020-10-13
category: 'Vision'
draft: false
---

> **An Image is Worth 16X16 Words: Transformers for Image Recognition at Scale**  
Anonymous authors  
https://openreview.net/pdf?id=YicbFdNTTy

# Abstract
 Transformer architecture는 NLP task에서는 표준이 되었지만 computer vison task에서는 아직 제한적이다.
Computer vision task에서 attention은 CNN과 함께 적용되거나 전체구조를 유지하면서 CNN의 특성 요소를 대체하는데 사용된다.  
본 논문에서는 CNN에 대한 이러한 의존이 필요하지 않으며 image patch의 seqeuence가 transformer에 적용될 때 image classification task에서 잘 수행될 수 있음을 보여준다.  
많은 양의 데이터에 대해 사전 학습을 수행하고 여러가지 recognition benchmark(ImageNet, CIFAR-100, VTAB 등)에 대해 transfer learning을 수행하면 Vision Transformer는 훨씬 적은 computational resource를 가지며 동시에 SotA CNN과 비교하여 더 우수한 결과를 얻을 수 있다.

# 1. Introduction
Transformers는 NLP에서 선택되는 모델이 되었다.
대부분의 접근 방식은 large text corpus에서 pre-train을 수행하고 task-specific dataset에 대해 fine-tuning을 수행하는 것이다(BERT).
Transformers의 계산 효율성 및 확장성으로 100B 이상의 parameter를 사용하여 전례없는 크기의 모델을 학습 할 수 있게 되었다.

그러나 computer vision에서는 CNN이 여전히 많이 사용된다.
NLP 성공에 영감을 받아 여러 연구들에서 CNN과 유사한 architecture를 self-attention과 결합하려고 시도하며([Wang et al., 2018](https://arxiv.org/abs/1906.01787); [Carion et al., 2020](https://arxiv.org/abs/2005.12872)) 일부는 CNN을 완전히 대체한다([Ramachandran et al., 2019](https://arxiv.org/abs/1906.05909); [Wang et al. , 2020a](https://arxiv.org/abs/2003.07853)).  
후자의 연구들은 이론적으로는 효율적이지만 specialized attention pattern을 사용하기 때문에 최신 하드웨어 가속기에서는 아직 효과적으로 사용하기 어렵다. 따라서 large-scale image recognition task에서 ResNet-like architecture는 여전히 SotA.

NLP의 Transformer 성공에 영감을 받아, 가능한 최소한의 수정으로 Transformer를 이미지에 직접 적용하는 실험을한다. 이를 위해 image를 patch로 분할하고 이러한 패치의 linear embedding sequence를 Transformer에 대한 입력으로 feed한다. Image patch는 NLP 애플리케이션의 token(word)과 동일한 방식으로 처리된다.

이러한 모델은 ImageNet과 같은 중간 규모의 dataset에서 학습할때 적당한 결과를 산출하여 비슷한 크기의 ResNet보다 조금 아래의 정확도를 달성한다. 이는 겉보기에 실망스러운 결과를 예상 할 수 있다. Transformer는 translation equivariance 및 locality와 같은 CNN 고유의 inductive bias가 없기 때문에 불충분한 양의 data에 대해 학습할 때 일반화가 잘 되지 않는다.

그러나 large-scale dataset(14M-300M Images)에서 모델을 학습하면 이와 다르다. large-scale training이 inductive bias를 능가한다는 것을 알게 되었다. Transformer는 충분한 규모로 사전 학습되고 더 적은 데이터 포인트가있는 작업으로 전송 될 때 좋은 결과를 얻는다.  
JFT-300M 데이터 세트에 대해 사전 훈련 된 Vision Transformer는 여러 Image Recognition Benchmark에서 SotA에 접근하거나 이를 능가하여 ImageNet에서 88.36%, ImageNet-ReaL에서 90.77%, CIFAR-100에서 94.55%, 77.16%의 정확도를 달성했다.

# 2. Related Work
Transformer는 NMT(Neural Machine Translation)을 위한것이며 많은 NLP task에서 SotA를 달성했다.
Large Transformer-based model은 종종 large-scale corpus에 대해 pre-train을 수행하고 해당되는 task에 fine-tuning을 수행한다.
BERT는 denoising self-supervised pre-training task를 사용하는 반면 GPT 계열은 language modeling방식으로 pre-train을 수행한다.

Self-attention을 image에 naive하게 적용하려면 각 픽셀이 다른 모든 픽셀에 attention해야한다.
이는 픽셀수의 quadratic cost를 가지고 실제 input size로 확장되지 않는다.
따라서 image generation 측면에서 Transformer를 적용하기 위해 시도된 몇 가지 연구들이 있다.

가장 최근의 연구인 [iGPT](https://cdn.openai.com/papers/Generative_Pretraining_from_Pixels_V2.pdf)는 image resolution과 color sapce를 줄인 후 이미지에 대해 transformer를 적용한다. Model은 unsupervised 방식으로 학습되고 이후 fine-tuning을 수행하거나 linear를 수행하여 ImageNet에서 72%의 정확도를 달성하였다. 해당 연구는 SotA결과를 얻기 위해 추가적인 데이터에 의존한다.

[Sun et al.(2017)](https://arxiv.org/abs/1707.02968)은 CNN 성능이 dataset 크기에 따라 어떻게 확장되는지 연구하고 [Kolesnikov et al. (2020)](https://arxiv.org/abs/1912.11370); [Djolonga et al. (2020)](https://arxiv.org/abs/2007.08558)은 ImageNet-21k 및 JFT-300M과 같은 large-scale dataset에서 CNN transfer learning에 대한 경험적 탐색을 수행하며, 모두 본 논문이 초점을 두고 있다.

# 3. Method
## 3.1 Vision Transformer(ViT)
![fig1](./img/vit/fig1.png)


 이미지용 Transformer는 NLP용으로 설계된 architecture를 따르며 그림1과 같다.
Standard Transformer는 token embedding의 1D sequence를 입력으로 받는다.  
이미지를 처리하기 위해 $\mathbf{x}\in \mathbb{ R }^{ H\times W\times C }$ 를 flatten된 2D patch ${ x }_{ p }\in { R }^{ N\times \left( { P }^{ 2 }\cdot C \right)  }$ sequence로 재구성한다.  
$\left( H,W \right) $는 원본 이미지의 resolution이고 $\left( P,P \right) $ 는 각 이미지 patch의 resolution이다.
$N=HW/{ P }^{ 2 }$는 Transformer의 sequence length이다.  
Transformer는 모든 layer를 일정한 width를 사용하므로 학습 가능한 linear projection은 각 vectorized patch를 dimension $D$(식 1)에 mapping하며, 그 결과를 patch embedding이라고 한다.

BERT의 $[class]$ token과 유사하게 Transformer encoder의 output state $\left( { z }_{ 0 }^{ L } \right) $가 image representation $y$로 사용되는 embedded patch의 sequence $\left( { z }_{ 0 }^{ 0 }={ x }_{ class } \right) $ 앞에 학습 가능한 embedding을 추가한다(식 4).  
Pre-train 및 fine-tuning중에 classification head는 $\left( { z }_{ L }^{ 0 } \right) $ 에 추가된다.

Position embedding은 위치 정보를 유지하기 위해 patch embedding에 더해진다.  
Appendix C.3에서 position embedding의 2D-aware variants에 대해 탐색한다.

Transformer Encoder는 Multi-headed self-attention 및 MLP block으로 구성된다. Layernorm(LN)은 모든 block 이전에 적용되고 residual connection은 모든 block 이후에 적용된다.  
$$
\mathbf{ z }_{ 0 }=\left[ \mathbf{ x }_{ class };\mathbf{ x }_{ p }^{ 1 }\mathbf{E};\mathbf{ x }_{ p }^{ 2 }\mathbf{E};\cdots ;\mathbf{ x }_{ p }^{ N }\mathbf{E} \right] +\mathbf{ E }_{ pos },\quad \mathbf{E}\in \mathbb{ R }^{ \left( { p }^{ 2 }\cdot C \right) \times D },\mathbf{ E }_{ pos }\in \mathbb{ R }^{ \left( N+1 \right) \times D }\quad (1)
$$
$$
\mathbf{ z }_{ \ell  }^{ \prime  }=\mathrm{MSA}\left( \mathrm{LN}\left( \mathbf{ z }_{ \ell -1 } \right)  \right) +\mathbf{ z }_{ \ell -1 },\quad \ell =1\dots L\quad (2)
$$
$$
\mathbf{ z }_{ \ell  }=\mathrm{MLP}\left( \mathrm{LN}\left( \mathbf{ z }_{ \ell  }^{ \prime  } \right)  \right) +\mathbf{ z }_{ \ell  }^{ \prime  },\quad \ell =1\dots L\quad (3)
$$
$$
\mathbf{y}=\mathrm{LN}\left( \mathbf{ z }_{ L }^{ 0 } \right) \quad (4)
$$

**Hybrid Architecture.**  
이미지를 patch로 나누는 대신 ResNet의 중간 feature map에서 input sequence를 형성할 수 있다.  
Hybrid Model에서 patch embedding projection $\mathbf{E}$(식 1) 는 ResNet의 early stage로 대체된다.  
ResNet의 중간 2D feature map중 하나는 sequence로 flatten되고 transformer dimension에 projection된 다음 transformer의 input sequence로 feed된다.  
Classification input embedding 및 position embedding은 위에서 설명한대로 transformer에 대한 input에 추가된다.

## 3.2 Fine-tuning and Higher Resolution
일반적으로 large-scale dataset에 대해 ViT를 pre-train하고 downstream task에 대해 fine-tuning을 수행한다.
이를 위해 pre-trained prediction head를 제거하고 0으로 초기화된 $D\times K$ feedforward layer를 추가한다.
여기서 $K$는 downstream class의 개수이다.

Pre-train 보다 높은 resolution으로 fine-tuning하는것은 종종 도움이 된다.
더 높은 resolution의 이미지를 feed할 때 patch 크기를 동일하게 유지하므로 sequence length가 더 길어진다.
Vision Transformer는 임의의 sequence length를 처리할 수 있지만 pre-trained position embedding은 의미가 없을 수 있다.
따라서 원본 이미지에서의 위치에 따라 pre-trained position embedding의 2D interpolation을 수행한다.
Resolution 조정 및 patch 추출은 이미지의 2D 구조에 대한 inductive bias가 Vision Transformer에 수동으로 주입되는 유일한 지점이다.

# 4. Experiments

