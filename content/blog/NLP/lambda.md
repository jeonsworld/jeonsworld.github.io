---
title: '[논문리뷰] LaMDA: Language Models for Dialog Applications'
date: 2022-03-25
category: 'NLP'
draft: true
---

> **LaMDA: Language Models for Dialog Applications**  
Romal Thoppilan, Daniel De Freitas, Jamie Hall, Noam Shazeer, Apoorv Kulshreshtha, Heng-Tze Cheng, Alicia Jin, Taylor Bos, Leslie Baker, Yu Du, YaGuang Li, Hongrae Lee, Huaixiu Steven Zheng, Amin Ghafouri, Marcelo Menegali, Yanping Huang, Maxim Krikun, Dmitry Lepikhin, James Qin, Dehao Chen, Yuanzhong Xu, Zhifeng Chen, Adam Roberts, Maarten Bosma, Vincent Zhao, Yanqi Zhou, Chung-Ching Chang, Igor Krivokon, Will Rusch, Marc Pickett, Pranesh Srinivasan, Laichee Man, Kathleen Meier-Hellstern, Meredith Ringel Morris, Tulsee Doshi, Renelito Delos Santos, Toju Duke, Johnny Soraker, Ben Zevenbergen, Vinodkumar Prabhakaran, Mark Diaz, Ben Hutchinson, Kristen Olson, Alejandra Molina, Erin Hoffman-John, Josh Lee, Lora Aroyo, Ravi Rajakumar, Alena Butryna, Matthew Lamm, Viktoriya Kuzmina, Joe Fenton, Aaron Cohen, Rachel Bernstein, Ray Kurzweil, Blaise Aguera-Arcas, Claire Cui, Marian Croak, Ed Chi, Quoc Le  
https://arxiv.org/abs/2201.08239


# 1. Introduction
Language model pre-train은 NLP에서 점점 더 유망한 연구 접근법이다.
pre-train은 model size 및 dataset size를 증가시켜 더 나은 성능이나 새로운 기능을 달성할 수 있다.
예를들어, large-scale unlabeled text corpus에서 학습된 GPT-3(175B)는 few-shot learning에서 좋은 성능을 보여준다.


Large language model의 응용프로그램 중 하나인 대화모델은 transformer의 능력을 성공적으로 활용한다([Adiwardana, Daniel, et al.](https://arxiv.org/abs/2001.09977), [Roller, Stephen, et al.](https://arxiv.org/abs/2004.13637)).
General language model과 유사하게 [Adiwardana, Daniel, et al.](https://arxiv.org/abs/2001.09977) 연구에서는 대화모델도 모델크기 증가에 매우 적합함을 보여준다.
모델의 크기와 대화품질 사이에는 강한 correlation이 있다.

이러한 연구들에 영감을 받아 대화를 위한 transformer 기반의 모델인 LaMDA를 학습한다.
모델의 크기는 parameter 2B~137B까지이며 공개된 대화데이터 및 기타 공개 웹 문서(section 3)의 1.56T words 데이터셋을 통해 사전학습을 진행했다.
LaMDA는 단일모델을 사용하여 여러 task를 수행한다:
1. 가능성 높은 response을 생성
2. 안전성을 위해 필터링
3. external knowledge를 기반으로 re-rank하고 고품질의 response 선택


본 논문에서는 quality, safety, groundedness 세 가지 metric에서 LaMDA를 사용한 model scaling의 이점을 연구한다.
1. model scaling 만으로도 quality가 향상되지만 safety 및 groundedness는 human performance에 비해 떨어짐
2. model scaling 과 fine-tuning을 결합하면 모든 metric에서 LaMDA가 크게 향상되고 safety 및 groundedness 측면에서는 human performance 밑으로 유지되지만 crowdworker 수준과의 quality 차이는 좁힐수 있었음.

* quality는 sensibleness, specificity, interestingness를 기반으로 한다.
* safety는 model이 생성하는 response중 안전하지 않은 response를 줄이기 위해 추가했다.
* groundedness는 external knowledge에 기반한 response를 생성하기 위해 추가했다.
