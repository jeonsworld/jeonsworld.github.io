---
title: 'Relative Position Encoding'
date: 2020-02-06
category: 'NLP'
draft: false
---

# Background
Transformer에서는 sinusoids 기반의 Position encoding을 사용하여 position 정보를 capture하였음. 또 다른 방법으로 BERT에서는 position embedding을 통해 position 정보를 capture함.

Self-Attention with Relative Position Representations(Peter el al., 2018)[[1]](https://arxiv.org/abs/1803.02155)에서는 input사이에 pair-wise relationship을 고려하도록 relative position representation을 제안.

기존의 BERT position embedding과 비교하면 BERT의 경우 sequence length가 512일 때, 0~511까지 sequence id를 통해 embedding을 통과. 이 때 각 위치의 vector값들은 항상 동일한 값. 이를 보완하기 위해 relative position representation은 각 token의 상대적인 위치를 나타냄.


# Implementation
huggingface T5 구현의 relative_position_encoding에 대한 구현은 다음과 같음.([reference](https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_t5.py#L232))
```python
def relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
    ret = 0
    n = -relative_position
    if bidirectional:
        num_buckets //= 2
        ret += (n < 0).to(torch.long) * num_buckets
        n = torch.abs(n)
    else:
        n = torch.max(n, torch.zeros_like(n))
    # now n is in the range [0, inf)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = n < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
    ).to(torch.long)
    val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
    ret += torch.where(is_small, n, val_if_large)
    return ret
```

* relative_position: relatvie position을 나타냄. 예시로 query, key의 length가 16이라고 가정할때 relative_position input은 다음과 같을 수 있음. 0 id는 현재 시점 $T$라고 볼 수 있다.
```bash
tensor([[  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
          14,  15],
        [ -1,   0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,
          13,  14],
        [ -2,  -1,   0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,
          12,  13],
        [ -3,  -2,  -1,   0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,
          11,  12],
        [ -4,  -3,  -2,  -1,   0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
          10,  11],
        [ -5,  -4,  -3,  -2,  -1,   0,   1,   2,   3,   4,   5,   6,   7,   8,
           9,  10],
        [ -6,  -5,  -4,  -3,  -2,  -1,   0,   1,   2,   3,   4,   5,   6,   7,
           8,   9],
        [ -7,  -6,  -5,  -4,  -3,  -2,  -1,   0,   1,   2,   3,   4,   5,   6,
           7,   8],
        [ -8,  -7,  -6,  -5,  -4,  -3,  -2,  -1,   0,   1,   2,   3,   4,   5,
           6,   7],
        [ -9,  -8,  -7,  -6,  -5,  -4,  -3,  -2,  -1,   0,   1,   2,   3,   4,
           5,   6],
        [-10,  -9,  -8,  -7,  -6,  -5,  -4,  -3,  -2,  -1,   0,   1,   2,   3,
           4,   5],
        [-11, -10,  -9,  -8,  -7,  -6,  -5,  -4,  -3,  -2,  -1,   0,   1,   2,
           3,   4],
        [-12, -11, -10,  -9,  -8,  -7,  -6,  -5,  -4,  -3,  -2,  -1,   0,   1,
           2,   3],
        [-13, -12, -11, -10,  -9,  -8,  -7,  -6,  -5,  -4,  -3,  -2,  -1,   0,
           1,   2],
        [-14, -13, -12, -11, -10,  -9,  -8,  -7,  -6,  -5,  -4,  -3,  -2,  -1,
           0,   1],
        [-15, -14, -13, -12, -11, -10,  -9,  -8,  -7,  -6,  -5,  -4,  -3,  -2,
          -1,   0]])
```

* bidirectional: 일반적으로 encoder는 True, decoder는 False. encoder의 경우는 현재 시점 $T$를 기준으로 앞, 뒤 양방향 position을 capture하고 decoder의 경우는 현재 시점 $T$를 기준으로 앞의 position만 capture.

* num_bucket: position encoding에 사용할 bucket의 갯수.

입력된 query, key length를 가진 sequence id에 대해 relative postion id를 만드는 과정이며 bidirectional의 경우는 bucket을 2로나눠 앞, 뒤의 position id로 사용. 12번 라인에서 bucket을 2로나눠 max_exact를 구하는데 max_exact의 값까지는 position의 값을 1씩 정확하게 증가시킴. 일정한 거리를 넘어서는 정확한 위치가 유용하지 않기 때문에 이렇게 하는것으로 생각됨.


## query, key length:16 bucket: 16일때의 output
### bidirectional: True
```bash
tensor([[ 0,  9, 10, 11, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13],
        [ 1,  0,  9, 10, 11, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13],
        [ 2,  1,  0,  9, 10, 11, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13],
        [ 3,  2,  1,  0,  9, 10, 11, 12, 12, 12, 12, 12, 12, 13, 13, 13],
        [ 4,  3,  2,  1,  0,  9, 10, 11, 12, 12, 12, 12, 12, 12, 13, 13],
        [ 4,  4,  3,  2,  1,  0,  9, 10, 11, 12, 12, 12, 12, 12, 12, 13],
        [ 4,  4,  4,  3,  2,  1,  0,  9, 10, 11, 12, 12, 12, 12, 12, 12],
        [ 4,  4,  4,  4,  3,  2,  1,  0,  9, 10, 11, 12, 12, 12, 12, 12],
        [ 4,  4,  4,  4,  4,  3,  2,  1,  0,  9, 10, 11, 12, 12, 12, 12],
        [ 4,  4,  4,  4,  4,  4,  3,  2,  1,  0,  9, 10, 11, 12, 12, 12],
        [ 5,  4,  4,  4,  4,  4,  4,  3,  2,  1,  0,  9, 10, 11, 12, 12],
        [ 5,  5,  4,  4,  4,  4,  4,  4,  3,  2,  1,  0,  9, 10, 11, 12],
        [ 5,  5,  5,  4,  4,  4,  4,  4,  4,  3,  2,  1,  0,  9, 10, 11],
        [ 5,  5,  5,  5,  4,  4,  4,  4,  4,  4,  3,  2,  1,  0,  9, 10],
        [ 5,  5,  5,  5,  5,  4,  4,  4,  4,  4,  4,  3,  2,  1,  0,  9],
        [ 5,  5,  5,  5,  5,  5,  4,  4,  4,  4,  4,  4,  3,  2,  1,  0]])
```

### bidirectional: False
```bash
tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [6, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [7, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [8, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0],
        [8, 8, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0],
        [8, 8, 8, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0],
        [9, 8, 8, 8, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0],
        [9, 9, 8, 8, 8, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 0],
        [9, 9, 9, 8, 8, 8, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0],
        [9, 9, 9, 9, 8, 8, 8, 8, 7, 6, 5, 4, 3, 2, 1, 0]])
```

이렇게 구해진 position id는 embedding을 통과하여 사용됨. T5구현에서는 (bucket_size, number_of_head)의 크기를 가진 embedding matrix 사용. 또한 relative position encoding이 수행되는 위치는 query, key의 matmul과 softmax사이라고 보면 됨.
