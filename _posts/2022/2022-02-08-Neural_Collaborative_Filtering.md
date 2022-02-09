---
title: Neural Collaborative Filtering
layout: post
Created: January 1, 2022 6:28 PM
tags:
    - Paper
    - Recommendation System
use_math: true

---



> 🧠 2017년 WWW에 발표된 'Neural Collaborative Filtering' 논문 요약 정리 및 코드 구현입니다.
> [Neural Collaborative Filtering Paper](https://arxiv.org/abs/1708.05031)
> [NCF Repository](https://github.com/IkGyu-Lee/NCF)



# 1. Abstract & Introduction

> NCF(Neural Collaborative Filtering) : replacing the inner product with a neural architecture that can learn an arbitrary function from data
>
>
> To supercharge NCF modelling with non-linearities, we propose to leverage a multi-layer perceptron to learn the user–item interaction function


- MF의 내적을 대체하는 용도로 Neural Architecture을 제안한다.
- supercharge을 위해 MLP를 활용한다.

> Despite the effectiveness of MF for collaborative filtering, it is well-known that its performance can be hindered by the simple choice of the interaction function — inner product.
>
>
> The inner product, which simply combines the multiplication of latent features linearly, may not be sufficient to capture the complex structure of user interaction data.


- MF를 개선시키기 위해 다양한 논문들이 나왔지만, MF가 사용하는 내적의 특성상 단순한 선택으로 성능을 저해시키며, 복잡한 구조에 충족되기 어렵다.

> We focus on [implicit feedback](https://www.notion.so/Explicit-vs-Implicit-Feedback-Data-9b2eac5db6ee442ba75d81e17fd47828)


- 해당 논문은 user가 직접(explicit) 평가하는 data가 아닌, implicit data에 중점을 둔다.

> We present a neural network architecture to model latent features of users and items and devise a general framework NCF for collaborative filtering based on neural networks.
>
>
> We show that MF can be interpreted as a specialization of NCF and utilize a multi-layer perceptron to endow NCF modelling with a high level of non-linearities.
>
> We perform extensive experiments on two real-world datasets to demonstrate the effectiveness of our NCF approaches and the promise of deep learning for collaborative filtering.

- 3가지 main contributions

# 2. Preliminaries

## 2.1 Learning from Implicit Data
<div class="center">
  <figure>
    <a href="/images/2022/NCF/t0.png"><img src="/images/2022/NCF/t0.png" width="600"  ></a>
  </figure>
</div>

- user's implicit feedback이기 때문에 explicit preference를 나타내지 않는다. 따라서 $y_{u,i} = 0$ 은 user와 item간의 interaction이 없다는 것을 의미한다.

> Moving one step forward, our NCF framework parameterizes the interaction function f using neural networks to estimate ˆyui. As such, it naturally supports both pointwise and pairwise learning.

- 본 논문에서는 NCF의 $\theta$를 학습하는데 [pointwise learning, pairwise learning](https://www.notion.so/Pointwise-vs-Pairwise-vs-Listwise-8661583de4f7418fb1914f96d2b66250) 2가지 방법 모두 다 사용하고자 한다.

## 2.2 Matrix Factorization

<div class="center">
  <figure>
    <a href="/images/2022/NCF/t1.png"><img src="/images/2022/NCF/t1.png" width="600"  ></a>
  </figure>
</div>

- $p_u$ : user latent vector
- $q_i$ : item latent vector

> MF models the two-way interaction of user and item latent factors, assuming each dimension of the latent space is independent of each other and linearly combining them with the same weight.

- MF는 **user latent vector**와 **item latent vector**의 inner product를 통해 interaction을 modeling한다. 각 latent space는 서로 독립적이며, 같은 weight로 linearly combining한다.

### Matrix Factorization's limit

<div class="center">
  <figure>
    <a href="/images/2022/NCF/t2.png"><img src="/images/2022/NCF/t2.png" width="600"  ></a>
  </figure>
</div>

- 본 논문에서는 [Jaccard coefficient](https://www.notion.so/Similarity-28496d0e6cbd4f70b3ce053cc2b08e76)를 활용하여 MF의 한계를 설명한다.

먼저, user 1,2,3만 고려했을 때의 유사도를 구하면 다음과 같다.

$s_{23}(0.66) > s_{12}(0.5) > s_{13}(0.4)$

이에 따라 latent space에 $p_1$, $p_2$, $p_3$를 나타낼 수 있다.

다음으로, user 4를 함께 고려하게 되었을 때의 유사도를 구하면 다음과 같다.

$s_{41}(0.6) > s_{43}(0.4) > s_{42}(0.2)$

user 4를 user 1과 가장 유사하면서 user 3보다 user 2가 덜 유사한 $p_4$를 나타낼 수 없다. 이것은 즉, MF의 inner product 한계를 나타낸다. 이는 user와 item 사이의 complex interaction을 low dimensional latent space로 나타냈기 때문이다.

> We note that one way to resolve the issue is to use a large number of latent factors K. However, it may adversely hurt the generalization of the model (e.g., overfitting the data), especially in sparse settings

- 이를 해결하기 위해 latent factor의 dimension을 높일 수 있지만, 이는 generalization of the model을 저해하게 된다. 즉, 성능이 떨어질 수 있다.
- 따라서 본 논문의 저자는 DNN을 이용해 해당 문제를 해결하고자 한다.

# 3. Neural Collaborative Filtering

user-item interaction function을 학습하기 위해 NCF는 non-linearity(비선형성)을 나타낼 수 있는 MLP를 사용한다.

## 3.1 General Framework

<div class="center">
  <figure>
    <a href="/images/2022/NCF/t3.png"><img src="/images/2022/NCF/t3.png" width="600"  ></a>
  </figure>
</div>
- Input Layer (Sparse)

input으로 각각 user와 item의 one-hot encoding을 한 vector를 사용한다. 이는 binarized sparse vector이다.

- Embedding Layer

input의 sparse vector를 dense vector로 mapping한다. 일반적인 embedding 방법과 동일하게 fully-connected layer를 사용한다.

- Neural CF Layers

user latent vector와 item latent vector를 concatenation한 vector를 input으로 DNN을 거친다. 마지막 hidden layer X의 dimension이 모델의 capability를 결정한다. 이를 통해 non-linearity data를 학습할 수 있다.

- Output Layer

Score $\hat{y}_{ui}$를 예측한다.

- Training

point-wise loss와 pair-wise loss(Bayesian Personalized Ranking, margin-based loss) 둘 다 학습이 가능하나, 본 논문에서는 $\hat{y}_{ui}$과 target value인 $y_{ui}$사이를 최소화하는 point-wise loss만 사용하여 학습한다.

- NCF model

<div class="center">
  <figure>
    <a href="/images/2022/NCF/t4.png"><img src="/images/2022/NCF/t4.png" width="600"  ></a>
  </figure>
</div>
$P$와 $Q$는 embedding layer의 Matrix $P\in \mathbb R^{M\times K}, Q\in \mathbb R^{N\times K}$이고, $\theta_{f}$는 interaction function $f$의 model parameter이다. function $f$은 multi-layer neural network이기에 표기하면 다음과 같다.

![Untitled](/images/2022/NCF/t5.png)

### Learning NCF

implicit data이기에 binary한 특징(bernoulli distribution)을 고려하여 Logistic이나 Probit function을 사용하여, $\hat{y}_{ui}$을 [0, 1]의 범위를 가지게 한다. 이에 따른 likelihood는 다음과 같다.

<div class="center">
  <figure>
    <a href="/images/2022/NCF/t6.png"><img src="/images/2022/NCF/t6.png" width="600"  ></a>
  </figure>
</div>
이어 loss function은 아래와 같으며, binary cross-entropy loss를 사용하는 것을 알 수 있다. 해당 model은 $L$의 값을 최소화하는 파라미터를 찾게 된다.

![Untitled](/images/2022/NCF/t7.png)

optimizer로는 SGD를 사용하며, unobserved interaction에 대한 item을 negative sample로 사용하였다.

## 3.2 Generalized Matrix Factorization (GMF)

> MF can be interpreted as a special case of our NCF framework
>
- NCF의 special case로 MF를 적용시킬 수 있다.

latent vector $P^Tv^U_{u}, Q^Tv^I_{i}$를 각각 $p_u, q_i$라고 표현, $a_{out}$은 activation function을 의미하며, $h$는 output layer의 가중치를 의미한다. $\odot$은 element-wise product(Hadamard product)를 의미한다. output layer에 project하면 다음과 같다.

<div class="center">
  <figure>
    <a href="/images/2022/NCF/t8.png"><img src="/images/2022/NCF/t8.png" width="600"  ></a>
  </figure>
</div>
$a_{out}$이 indentity function, $h$는 uniform vector라면, 기존의 MF와 동일하다. 본 논문에서는 $a_{out}$을 sigmoid function( $\sigma(x) = 1/(1+e^{-x})$ ), $h$를 log loss를 사용해 GMF를 학습하였다.

## 3.3 Multi-Layer Perceptron (MLP)

<div class="center">
  <figure>
    <a href="/images/2022/NCF/t9.png"><img src="/images/2022/NCF/t9.png" width="600"  ></a>
  </figure>
</div>
$\phi_1$는 concatenate 함수, $W_x$는 weight matrix, $b_x$는 bias vector, $a_x$는 activation function을 의미한다.

단순한 vector의 concatenation으로 user와 item 사이의 상호작용을 설명하지 않아, CF modeling의 효과를 주기에는 충분하지 않다. 이 때문에 MLP를 사용했다.

## 3.4 Fusion of GMF and MLP (NeuMF)

NCF framework를 기반으로 한 GMF와 MLP를 fuse한 것이 Neural Matrix Factorization(NeuMF)이다.

<div class="center">
  <figure>
    <a href="/images/2022/NCF/t10.png"><img src="/images/2022/NCF/t10.png" width="600"  ></a>
  </figure>
</div>

- GMF Layer는 user와 item vector간의 element-wise product를 진행하여, latent feature interaction을 위해 linear kernel을 apply.
- MLP Layer X는 user와 item vector를 concatenate하여 hidden layer(MLP Layer i)를 진행한다. data에서 interaction function을 학습하기 위해 non-linear kernel을 apply.
- NeuMF Layer는 GMF와 MLP의 output을 concatenate하여 $\hat{y}_{ui}$(score)를 예측한다.
- GMF와 MLP Layer를 각각 먼저 pre-training한 다음, last hidden layer를 concat하여 NeuMF를 학습.
- GMF, MLP, NeuMF without pretraining은 Adam, NeuMF with pretraining은 SGD를 Optimizer로 사용.

![Untitled](/images/2022/NCF/t11.png)

GMF, MLP 둘 다 같은 embedding size를 가져야하기 때문에 performance에 limit이 있을 수 있으며, optimal embedding size가 많이 변화하는 dataset은 optimal ensemble을 낼 수 없을 지도 모른다.

### 3.4.1 Pre-training

단순히 GMF와 MLP를 concat하여 objective function으로 Adam을 사용하는 것은 gradient-based optimization 방법들의 특성상 local optima로 빠질 수 있다. 실제로 딥러닝 모델의 수렴과 성능에 initialization이 중요한 역할을 한다고 알려져 있다. 따라서 무작위의 initialization을 하여 NeuMF를 학습하는 과정에서 parameter들을 update하는 것보다 convergence(수렴)되어 있는 GMF와 MLP의 parameter들을 가져와 사용하는 것이 더 performace가 좋다.

# 4. Experiments

실험은 3가지의 질문을 바탕으로 진행한다.

Q1) Do our proposed NCF methods outperform the state-of-the-art implicit collaborative filtering methods?

- Performance comparison

Q2) How does our proposed optimization framework (log loss with negative sampling) work for the recommendation task?

- Log loss with Negative sampling

Q3) Are deeper layers of hidden units helpful for learning from user-item interaction data?

- Is deep learning helpful?

### 4.1 Experimental Settings

![Untitled](/images/2022/NCF/t12.png)

- Dataset : MovieLens와 Pinterest dataset 사용하며, 최소 20개의 rating이나 pin을 남긴 user의 data만으로 학습을 진행한다.
- Evaluation Protocols : Hit Ratio, NDCG
- Baselines : NCF method(GMF, MLP, NeuMF)와 비교하기위해 baseline으로 ItemKNN, BPR, eALS를 사용한다.
- Parameter Settings :
    - Binary Log loss function(BCE)
    - Sampling four negative instances per positive instance
    - randomly initialized model parameters with (Gaussian Distribution, mean=0, std=0.01)
    - optimizer : Adam(GMF, MLP), SGD(NeuMF)
    - batch size : [128, 256, 512, 1024]
    - learning rate : [0.0001, 0.0005, 0.001, 0.005]
    - predictive factors and evaluated the factors : [8, 16, 32, 64]
    - NeuMF with pre-training, α was set to 0.5
- Ex) predictive factor가 8이고, model architecture가 “32 → 16 → 8”인 model을 만든다고 가정을 하자. embedding size는 16이 되는데, 그 이유는 user와 item vector에 대해 concat을 해주기 때문에 embedding 과정에서 첫 번째 hidden layer input의 절반으로 embedding해야 concat했을 때, size가 같아진다.

### 4.2 Performance comparison

![Untitled](/images/2022/NCF/t13.png)

NeuMF의 성능이 가장 좋으며, 단일 모델의 경우에는 오히려 MLP보다 GMF의 성능이 더 좋다.

### 4.3 Log loss with Negative sampling

![Untitled](/images/2022/NCF/t14.png)

Loss 값이 줄어드는 것을 통해 interaction function을 학습하는데 적절하다는 것을 알 수 있다.

### 4.4 Is deep learning helpful?

![Untitled](/images/2022/NCF/t15.png)

Layer의 depth가 늘어날 수록 성능이 개선되고 있음을 알 수 있다.
