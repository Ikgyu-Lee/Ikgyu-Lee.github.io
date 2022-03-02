---
title: Batch Normalization
layout: post
Created: February 6, 2022 3:28 PM
tags:
    - DL
use_math: true
comments: true

---

NN은 학습해야할 parameter들이 너무 많다.

<div class="center">
  <figure>
    <a href="/images/2022/BatchNorm/t0.png"><img src="/images/2022/BatchNorm/t0.png" width="300"  ></a>
  </figure>
</div>

DNN은 hidden layer가 더 깊기 때문에 모든 layer에서의 작은 변화들이 hidden layer를 통과할 수록 아주 다른 값을 만들어 낼 수 있다.

<div class="center">
  <figure>
    <a href="/images/2022/BatchNorm/t1.png"><img src="/images/2022/BatchNorm/t1.png" width="400"  ></a>
  </figure>
</div>

이 때 변동이 기존에서 벗어나 변동이 심해지게 되는데 이것을 저자는 Internal Covariate Shift라고 하였다.

이것을 초기에 initialization을 잘 설정해주거나 learning rate를 작게 한다면, 어느정도 해결이 가능하지만, 전자는 적절한 initialization은 어렵고, 후자는 학습속도가 느려지게 된다.

이 문제를 Batch Normalization을 활용하여 개선할 수 있다.

![Untitled](/images/2022/BatchNorm/t2.png)

학습하는 과정에서 Batch별로 layer를 지날때 마다 나오는 ouput을 mean 0, standard deviation 1로 분포를 정규화하는 것을 말한다.

Train Step에서는 모든 feature에 정규화 해주면, feature가 동일한 scale이 되어 learning rate 결정에 유리하다. 만약 feature의 scale이 다르다면, feature마다 learning rate에 의해 weight마다 반응하는 정도가 다르다. 이렇게 되면, 큰 weight에 대해서는 gradient exploding이 작은 weight에 대해서는 gradient vanishing 문제가 발생한다. 하지만, Batch Norm을 하면, weight의 반응이 같아지기에 학습에 유리하다.

Test Step에서는 batch 단위 별로 Batch Normalization이 계속 변경되고 업데이트 되는데 추론 단계에서는 학습 단계에서 결정된 값을 고정하여 사용한다. 즉, 학습 과정 속에서 평균과 분산을 **이동 평균 또는 지수 평균에 의하여 고정**하고 이것을 추론 과정에서 사용한다.

Batch Norm의 장점은 Regularization Effect가 있어서 Dropout이 필요하지 않다는 것이다.

Batch Norm의 한계점은 Batch의 크기에 영향을 많이 받는다. 이를 개선하기 위해 Weight Normalization이나, Layer Normalization을 사용할 수 있다.

---

### Reference

[https://arxiv.org/abs/1502.03167](https://arxiv.org/abs/1502.03167)

[https://www.youtube.com/watch?v=TDx8iZHwFtM&list=PLlMkM4tgfjnJhhd4wn5aj8fVTYJwIpWkS](https://www.youtube.com/watch?v=TDx8iZHwFtM&list=PLlMkM4tgfjnJhhd4wn5aj8fVTYJwIpWkS)

[https://gaussian37.github.io/dl-concept-batchnorm/](https://gaussian37.github.io/dl-concept-batchnorm/)
