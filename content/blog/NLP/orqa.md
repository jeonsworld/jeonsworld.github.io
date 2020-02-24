---
title: '[논문리뷰] Latent Retrieval for Weakly Supervised Open Domain Question Answering'
date: 2020-02-21
category: 'NLP'
draft: true
---

> **Latent Retrieval for Weakly Supervised Open Domain Question Answering**  
Kenton Lee, Ming-Wei Chang, Kristina Toutanova  
https://arxiv.org/abs/1906.00300


# 1. Introduction
MRC의 발전으로 Open-domain question answering(QA)가 관심받고 있는데, 이는 evidence는 input으로 제공되지 않고 open corpus에서 검색되어야 한다.
Open-QA는 실제 응용프로그램에 대해 보다 현실적인 시나리오를 제공한다.

현재 대부분의 접근 방식은 information retrieval(IR)을 통해 evidence candidate를 찾고 이를 통해 gold answer를 생성한다.
대표적으로 DrQA가 있는데 IR system을 사용하여 많은 양의 task를 수행해야 한다.

근본적으로 QA와 IR은 다르다.
IR은 lexical 및 semantic 매칭이 필요한 반면, 사용자들은 명확하게 알려지지 않은 정보를 질문하기 때문에 질문들은 정의가 불충분하고 더 많은 언어 이해를 필요로 한다.
IR system이 recall 상한을 받는 대신, QA data를 이용하여 검색하는 방법을 직접 배워야 한다.

본 논문에서는 Open Retrieval Question Answering(ORQA) System을 제안한다.
ORQA는 open corpus에서 evidence를 검색하는 방법을 배우고 question-answer pair로만 학습이 된다.

본 논문의 핵심은 unsupervised Inverse Cloze Task(ICT)를 통해 retriever를 pre-train하면 end-to-end 학습이 가능하다는 것이다.
ICT에서 sentence는 pseudo-question으로 처리되고 context는 pseudo-evidence로 처리된다.
ICT pre-training은 정답을 marginal log-likelihood로 최적화하여 ORQA를 end-to-end로 fine-tuning할 수 있도록 초기화할 수 있다.

다음과 같은 기존의 QA dataset을 통해 model을 평가한다.
1. SQuAD
2. TriviaQA
3. Natural Questions
4. WebQuestion
5. CuratedTrec


# 2. Overview
2절 에서는 prior work, baseline, 제안하는 model을 비교하는데 유용한 Open-QA 표기법을 설명한다.

## 2.1 Task
Open-QA에서 input $q$는 question string이고 output $a$는 answer string이다.
Reading comprehension과 달리 evidence는 task의 일부가 아닌 model로부터 검색된다.
표1의 reading comprehension 및 question answering의 변형에 의해 만들어진 가정을 비교한다.

![table1](./img/orqa/table1.png)

## 2.2 Formal Definitions
Model은 B의 evidence text block으로 분할되는 unstructured text corpus와 관련하여 정의된다.
답변 도출은 pair $(b,s)$이며 여기서 $1\le b\le B$는 evidence block의 index를 나타내고 $s$는 block $b$의 text span을 나타낸다.
Span의 시작 및 끝의 token index는 각각 $\mathsf{START}(s)$ 및 $\mathsf{END}(s)$로 나타낸다.

Model은 questions $q$에 대한 답변 도출 $(b,s)$를 나타내는 scoring function $S(b,s,q)$를 정의한다.
일반적으로 scoring function은 retrieval component ${ S }_{ retr }\left( b,q \right) $ 맟 reader component ${ S }_{ read }\left( b,s,q \right) $에 대해 decompose된다.
$$
S(b,s,q)={ S }_{ retr }(b,q)+{ S }_{ read }(b,s,q)
$$

Inference동안 model은 가장 높은 score를 가진 answer string을 출력한다.
$$
{ a }^{ * }=TEXT(\underset { b,s }{ argmax } S(b,s,q))
$$

$TEXT(b,s)$는 결정적으로 answer string에 대한 answer derivation $(b,s)$를 mapping한다.
Open-QA system의 주요 task는 scale을 처리하는 것이다.
영어 위키에 대한 실험에서 본 논문은 1,300만 이상의 evidence block $b$를 고려한다.

## 2.3 Existing Pipelined Models
기존의 검색 기반 Open-QA system에서 blackbox IR system은 먼저 closed evidence candidate를 선택한다.
예를 들어 DrQA의 retriever component의 점수는 다음과 같이 정의된다.

$$
{ S }_{ retr }\left( b,q \right) =\begin{cases} 0\quad b\in \mathsf{TOP}\left( k,\mathsf{TF-IDF}\left( q,b \right)  \right)  \\ -\infty \quad \mathsf{otherwise} \end{cases}
$$

DrQA 이후의 연구들은 대부분 TF-IDF를 사용하여 evidence를 찾고 reading comprehension 및 re-reanking을 수행하는 것에 focus를 둔다.
Reading comprehension component ${ S }_{ read }(b,s,q)$는 SQuAD dataset의 gold answer를 통해 학습된다.

위의 연구들과 비슷하게 우리의 접근방식에서 reader는 weak supervision을 통해 학습한다.
모호성은 검색 시스템에 의해 제거되고, cleaned result는 gold derivation으로 처리된다.



# 3. Open-Retrieval Question Answering (ORQA)
Retriever와 reader component가 joint learn되는 end-to-end model을 제안하는데 이를 ORQA(Open-Retrieval Question Answering)model 이라고 한다.
ORQA의 중요한 측면은 표현성이다.
이는 blackbox IR system으로부터 반환된 closed set로 제한되지 않고 open corpus에서 text를 검색할 수 있다.
ORQA의 점수 도출 방식은 그림1에서 확인할 수 있다.

![fig1](./img/orqa/fig1.png)

Transfer learning의 최근 발전에 따라 모든 scoring component는 unsupervised language modeling data로 부터 pre-train된 bidirectional transformer인 BERT에서 파생된다.
해당 task에서 rlevant abstraction은 다음 function으로 설명할 수 있다.
$$
\mathsf{BERT}({ x }_{ 1 },\left[ { x }_{ 2 } \right] =\left\{ \mathsf{CLS}:{ h }_{ CLS },1:{ h }_{ 1 },2:{ h }_{ 2 },\dots  \right\}
$$

BERT function은 하나 또는 두 개의 string input(${x}_{1}$ 및 선택적으로 ${x}_{2}$)을 argument로 사용한다.
CLS pooling token 또는 input token representation에 해당하는 vector를 반환한다.

**Retriever component:**
Retriever를 학습할 수 있도록 retrieval score를 question $q$와 evidence block $b$의 inner product로 정의한다.  
$$
{ h }_{ q }=\mathbf{{ W }_{ q }{ BERT }_{ Q }}(q)\mathsf{[CLS]}\\ { h }_{ b }=\mathbf{{ W }_{ b }{ BERT }_{ B }}(b)\mathsf{[CLS]}\\ { S }_{ retr }(b,q)={ h }_{ q }^{ \intercal  }{ h }_{ b }
$$

여기서 $\mathbf{{ W }}_{ q }$ 및 $\mathbf{{ W }}_{ b }$는 BERT output을 128 dimension vector로 projection하는 matrix이다.

**Reader component:**
Reader는 BERT에서 제안된 reading comprehension model의 span-based 변형이다.  
$$
{ h }_{ start }=\mathbf{{ BERT }}_{ R }(q,b)[\mathbf{START}(s)]\\ { h }_{ end }=\mathbf{{ BERT }}_{ R }(q,b)[\mathbf{END}(s)]\\ { S }_{ read }(b,s,q)=\mathbf{MLP}\left( \left[ { h }_{ start };{ h }_{ end } \right]  \right)
$$

Lee et al.,(2016)에서 span은 end point의 concatenation이며, 시작 및 종료 interaction을 가능하게 하는 multi-layer perceptron으로 score가 매겨진다.

**Inference & Learning Challenges:**
위에서 설명한 model은 개념적으로 간단하다.
그러나 (1)open dvidence corpus가 거대한 검색 공간을 제공하고(1,300만개가 넘는 evidence block) (2) 공간을 탐색하는 방법이 완전히 잠재되어 있으므로 standard teacher-forcing 접근방식이 적용되지 않아 inference 및 train이 까다롭다.
Latent-variable method는 모호성으로 인해 naive하게 적용하기 어렵다.
예를 들어 표2에 표시된 것처럼 Wikipedia의 많은 구절에는 "seven"이라는 관련이 없는 answer string이 포함된다.

![table2](./img/orqa/table2.png)

우리는 unsupervised pre-training을 통해 retriever를 신중하게 초기화하여 이러한 문제를 해결한다.
Pre-trained retriever를 사용하면 (1) Wikipedia의 모든 evidence block을 pre-encode하여 fine-tuning중에 동적이지만 빠른 top-k 검색을 수행할 수 있으며 (2) 모호성에서 멀어지게 검색에 편향을 줄 수 있다.