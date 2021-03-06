---
title: Neural Collaborative Filtering
layout: post
Created: January 1, 2022 6:28 PM
tags:
    - Recommendation System
    - Paper
use_math: true
comments: true

---



> ๐ง  2017๋ WWW์ ๋ฐํ๋ 'Neural Collaborative Filtering' ๋ผ๋ฌธ ์์ฝ ์ ๋ฆฌ ๋ฐ ์ฝ๋ ๊ตฌํ์๋๋ค.
>
> [Neural Collaborative Filtering Paper](https://arxiv.org/abs/1708.05031)
>
> [NCF Repository](https://github.com/IkGyu-Lee/NCF)



# 1. Abstract & Introduction

> NCF(Neural Collaborative Filtering) : replacing the inner product with a neural architecture that can learn an arbitrary function from data
>
>
> To supercharge NCF modelling with non-linearities, we propose to leverage a multi-layer perceptron to learn the userโitem interaction function


- MF์ ๋ด์ ์ ๋์ฒดํ๋ ์ฉ๋๋ก Neural Architecture์ ์ ์ํ๋ค.
- supercharge์ ์ํด MLP๋ฅผ ํ์ฉํ๋ค.

> Despite the effectiveness of MF for collaborative filtering, it is well-known that its performance can be hindered by the simple choice of the interaction function โ inner product.
>
>
> The inner product, which simply combines the multiplication of latent features linearly, may not be sufficient to capture the complex structure of user interaction data.


- MF๋ฅผ ๊ฐ์ ์ํค๊ธฐ ์ํด ๋ค์ํ ๋ผ๋ฌธ๋ค์ด ๋์์ง๋ง, MF๊ฐ ์ฌ์ฉํ๋ ๋ด์ ์ ํน์ฑ์ ๋จ์ํ ์ ํ์ผ๋ก ์ฑ๋ฅ์ ์ ํด์ํค๋ฉฐ, ๋ณต์กํ ๊ตฌ์กฐ์ ์ถฉ์กฑ๋๊ธฐ ์ด๋ ต๋ค.

> We focus on [implicit feedback](https://www.notion.so/Explicit-vs-Implicit-Feedback-Data-9b2eac5db6ee442ba75d81e17fd47828)


- ํด๋น ๋ผ๋ฌธ์ user๊ฐ ์ง์ (explicit) ํ๊ฐํ๋ data๊ฐ ์๋, implicit data์ ์ค์ ์ ๋๋ค.

> We present a neural network architecture to model latent features of users and items and devise a general framework NCF for collaborative filtering based on neural networks.
>
>
> We show that MF can be interpreted as a specialization of NCF and utilize a multi-layer perceptron to endow NCF modelling with a high level of non-linearities.
>
> We perform extensive experiments on two real-world datasets to demonstrate the effectiveness of our NCF approaches and the promise of deep learning for collaborative filtering.

- 3๊ฐ์ง main contributions

# 2. Preliminaries

## 2.1 Learning from Implicit Data
<div class="center">
  <figure>
    <a href="/images/2022/NCF/t0.png"><img src="/images/2022/NCF/t0.png" width="600"  ></a>
  </figure>
</div>

- user's implicit feedback์ด๊ธฐ ๋๋ฌธ์ explicit preference๋ฅผ ๋ํ๋ด์ง ์๋๋ค. ๋ฐ๋ผ์ $y_{u,i} = 0$ ์ user์ item๊ฐ์ interaction์ด ์๋ค๋ ๊ฒ์ ์๋ฏธํ๋ค.

> Moving one step forward, our NCF framework parameterizes the interaction function f using neural networks to estimate หyui. As such, it naturally supports both pointwise and pairwise learning.

- ๋ณธ ๋ผ๋ฌธ์์๋ NCF์ $\theta$๋ฅผ ํ์ตํ๋๋ฐ [pointwise learning, pairwise learning](https://www.notion.so/Pointwise-vs-Pairwise-vs-Listwise-8661583de4f7418fb1914f96d2b66250) 2๊ฐ์ง ๋ฐฉ๋ฒ ๋ชจ๋ ๋ค ์ฌ์ฉํ๊ณ ์ ํ๋ค.

## 2.2 Matrix Factorization

<div class="center">
  <figure>
    <a href="/images/2022/NCF/t1.png"><img src="/images/2022/NCF/t1.png" width="600"  ></a>
  </figure>
</div>

- $p_u$ : user latent vector
- $q_i$ : item latent vector

> MF models the two-way interaction of user and item latent factors, assuming each dimension of the latent space is independent of each other and linearly combining them with the same weight.

- MF๋ **user latent vector**์ **item latent vector**์ inner product๋ฅผ ํตํด interaction์ modelingํ๋ค. ๊ฐ latent space๋ ์๋ก ๋๋ฆฝ์ ์ด๋ฉฐ, ๊ฐ์ weight๋ก linearly combiningํ๋ค.

### Matrix Factorization's limit

<div class="center">
  <figure>
    <a href="/images/2022/NCF/t2.png"><img src="/images/2022/NCF/t2.png" width="600"  ></a>
  </figure>
</div>

- ๋ณธ ๋ผ๋ฌธ์์๋ [Jaccard coefficient](https://www.notion.so/Similarity-28496d0e6cbd4f70b3ce053cc2b08e76)๋ฅผ ํ์ฉํ์ฌ MF์ ํ๊ณ๋ฅผ ์ค๋ชํ๋ค.

๋จผ์ , user 1,2,3๋ง ๊ณ ๋ คํ์ ๋์ ์ ์ฌ๋๋ฅผ ๊ตฌํ๋ฉด ๋ค์๊ณผ ๊ฐ๋ค.

$s_{23}(0.66) > s_{12}(0.5) > s_{13}(0.4)$

์ด์ ๋ฐ๋ผ latent space์ $p_1$, $p_2$, $p_3$๋ฅผ ๋ํ๋ผ ์ ์๋ค.

๋ค์์ผ๋ก, user 4๋ฅผ ํจ๊ป ๊ณ ๋ คํ๊ฒ ๋์์ ๋์ ์ ์ฌ๋๋ฅผ ๊ตฌํ๋ฉด ๋ค์๊ณผ ๊ฐ๋ค.

$s_{41}(0.6) > s_{43}(0.4) > s_{42}(0.2)$

user 4๋ฅผ user 1๊ณผ ๊ฐ์ฅ ์ ์ฌํ๋ฉด์ user 3๋ณด๋ค user 2๊ฐ ๋ ์ ์ฌํ $p_4$๋ฅผ ๋ํ๋ผ ์ ์๋ค. ์ด๊ฒ์ ์ฆ, MF์ inner product ํ๊ณ๋ฅผ ๋ํ๋ธ๋ค. ์ด๋ user์ item ์ฌ์ด์ complex interaction์ low dimensional latent space๋ก ๋ํ๋๊ธฐ ๋๋ฌธ์ด๋ค.

> We note that one way to resolve the issue is to use a large number of latent factors K. However, it may adversely hurt the generalization of the model (e.g., overfitting the data), especially in sparse settings

- ์ด๋ฅผ ํด๊ฒฐํ๊ธฐ ์ํด latent factor์ dimension์ ๋์ผ ์ ์์ง๋ง, ์ด๋ generalization of the model์ ์ ํดํ๊ฒ ๋๋ค. ์ฆ, ์ฑ๋ฅ์ด ๋จ์ด์ง ์ ์๋ค.
- ๋ฐ๋ผ์ ๋ณธ ๋ผ๋ฌธ์ ์ ์๋ DNN์ ์ด์ฉํด ํด๋น ๋ฌธ์ ๋ฅผ ํด๊ฒฐํ๊ณ ์ ํ๋ค.

# 3. Neural Collaborative Filtering

user-item interaction function์ ํ์ตํ๊ธฐ ์ํด NCF๋ non-linearity(๋น์ ํ์ฑ)์ ๋ํ๋ผ ์ ์๋ MLP๋ฅผ ์ฌ์ฉํ๋ค.

## 3.1 General Framework

<div class="center">
  <figure>
    <a href="/images/2022/NCF/t3.png"><img src="/images/2022/NCF/t3.png" width="600"  ></a>
  </figure>
</div>
- Input Layer (Sparse)

input์ผ๋ก ๊ฐ๊ฐ user์ item์ one-hot encoding์ ํ vector๋ฅผ ์ฌ์ฉํ๋ค. ์ด๋ binarized sparse vector์ด๋ค.

- Embedding Layer

input์ sparse vector๋ฅผ dense vector๋ก mappingํ๋ค. ์ผ๋ฐ์ ์ธ embedding ๋ฐฉ๋ฒ๊ณผ ๋์ผํ๊ฒ fully-connected layer๋ฅผ ์ฌ์ฉํ๋ค.

- Neural CF Layers

user latent vector์ item latent vector๋ฅผ concatenationํ vector๋ฅผ input์ผ๋ก DNN์ ๊ฑฐ์น๋ค. ๋ง์ง๋ง hidden layer X์ dimension์ด ๋ชจ๋ธ์ capability๋ฅผ ๊ฒฐ์ ํ๋ค. ์ด๋ฅผ ํตํด non-linearity data๋ฅผ ํ์ตํ  ์ ์๋ค.

- Output Layer

Score $\hat{y}_{ui}$๋ฅผ ์์ธกํ๋ค.

- Training

point-wise loss์ pair-wise loss(Bayesian Personalized Ranking, margin-based loss) ๋ ๋ค ํ์ต์ด ๊ฐ๋ฅํ๋, ๋ณธ ๋ผ๋ฌธ์์๋ $\hat{y}_{ui}$๊ณผ target value์ธ $y_{ui}$์ฌ์ด๋ฅผ ์ต์ํํ๋ point-wise loss๋ง ์ฌ์ฉํ์ฌ ํ์ตํ๋ค.

- NCF model

<div class="center">
  <figure>
    <a href="/images/2022/NCF/t4.png"><img src="/images/2022/NCF/t4.png" width="600"  ></a>
  </figure>
</div>
$P$์ $Q$๋ embedding layer์ Matrix $P\in \mathbb R^{M\times K}, Q\in \mathbb R^{N\times K}$์ด๊ณ , $\theta_{f}$๋ interaction function $f$์ model parameter์ด๋ค. function $f$์ multi-layer neural network์ด๊ธฐ์ ํ๊ธฐํ๋ฉด ๋ค์๊ณผ ๊ฐ๋ค.

![Untitled](/images/2022/NCF/t5.png)

### Learning NCF

implicit data์ด๊ธฐ์ binaryํ ํน์ง(bernoulli distribution)์ ๊ณ ๋ คํ์ฌ Logistic์ด๋ Probit function์ ์ฌ์ฉํ์ฌ, $\hat{y}_{ui}$์ [0, 1]์ ๋ฒ์๋ฅผ ๊ฐ์ง๊ฒ ํ๋ค. ์ด์ ๋ฐ๋ฅธ likelihood๋ ๋ค์๊ณผ ๊ฐ๋ค.

<div class="center">
  <figure>
    <a href="/images/2022/NCF/t6.png"><img src="/images/2022/NCF/t6.png" width="600"  ></a>
  </figure>
</div>
์ด์ด loss function์ ์๋์ ๊ฐ์ผ๋ฉฐ, binary cross-entropy loss๋ฅผ ์ฌ์ฉํ๋ ๊ฒ์ ์ ์ ์๋ค. ํด๋น model์ $L$์ ๊ฐ์ ์ต์ํํ๋ ํ๋ผ๋ฏธํฐ๋ฅผ ์ฐพ๊ฒ ๋๋ค.

![Untitled](/images/2022/NCF/t7.png)

optimizer๋ก๋ SGD๋ฅผ ์ฌ์ฉํ๋ฉฐ, unobserved interaction์ ๋ํ item์ negative sample๋ก ์ฌ์ฉํ์๋ค.

## 3.2 Generalized Matrix Factorization (GMF)

> MF can be interpreted as a special case of our NCF framework
>
- NCF์ special case๋ก MF๋ฅผ ์ ์ฉ์ํฌ ์ ์๋ค.

latent vector $P^Tv^U_{u}, Q^Tv^I_{i}$๋ฅผ ๊ฐ๊ฐ $p_u, q_i$๋ผ๊ณ  ํํ, $a_{out}$์ activation function์ ์๋ฏธํ๋ฉฐ, $h$๋ output layer์ ๊ฐ์ค์น๋ฅผ ์๋ฏธํ๋ค. $\odot$์ element-wise product(Hadamard product)๋ฅผ ์๋ฏธํ๋ค. output layer์ projectํ๋ฉด ๋ค์๊ณผ ๊ฐ๋ค.

<div class="center">
  <figure>
    <a href="/images/2022/NCF/t8.png"><img src="/images/2022/NCF/t8.png" width="600"  ></a>
  </figure>
</div>
$a_{out}$์ด indentity function, $h$๋ uniform vector๋ผ๋ฉด, ๊ธฐ์กด์ MF์ ๋์ผํ๋ค. ๋ณธ ๋ผ๋ฌธ์์๋ $a_{out}$์ sigmoid function( $\sigma(x) = 1/(1+e^{-x})$ ), $h$๋ฅผ log loss๋ฅผ ์ฌ์ฉํด GMF๋ฅผ ํ์ตํ์๋ค.

## 3.3 Multi-Layer Perceptron (MLP)

<div class="center">
  <figure>
    <a href="/images/2022/NCF/t9.png"><img src="/images/2022/NCF/t9.png" width="600"  ></a>
  </figure>
</div>
$\phi_1$๋ concatenate ํจ์, $W_x$๋ weight matrix, $b_x$๋ bias vector, $a_x$๋ activation function์ ์๋ฏธํ๋ค.

๋จ์ํ vector์ concatenation์ผ๋ก user์ item ์ฌ์ด์ ์ํธ์์ฉ์ ์ค๋ชํ์ง ์์, CF modeling์ ํจ๊ณผ๋ฅผ ์ฃผ๊ธฐ์๋ ์ถฉ๋ถํ์ง ์๋ค. ์ด ๋๋ฌธ์ MLP๋ฅผ ์ฌ์ฉํ๋ค.

## 3.4 Fusion of GMF and MLP (NeuMF)

NCF framework๋ฅผ ๊ธฐ๋ฐ์ผ๋ก ํ GMF์ MLP๋ฅผ fuseํ ๊ฒ์ด Neural Matrix Factorization(NeuMF)์ด๋ค.

<div class="center">
  <figure>
    <a href="/images/2022/NCF/t10.png"><img src="/images/2022/NCF/t10.png" width="600"  ></a>
  </figure>
</div>

- GMF Layer๋ user์ item vector๊ฐ์ element-wise product๋ฅผ ์งํํ์ฌ, latent feature interaction์ ์ํด linear kernel์ apply.
- MLP Layer X๋ user์ item vector๋ฅผ concatenateํ์ฌ hidden layer(MLP Layer i)๋ฅผ ์งํํ๋ค. data์์ interaction function์ ํ์ตํ๊ธฐ ์ํด non-linear kernel์ apply.
- NeuMF Layer๋ GMF์ MLP์ output์ concatenateํ์ฌ $\hat{y}_{ui}$(score)๋ฅผ ์์ธกํ๋ค.
- GMF์ MLP Layer๋ฅผ ๊ฐ๊ฐ ๋จผ์  pre-trainingํ ๋ค์, last hidden layer๋ฅผ concatํ์ฌ NeuMF๋ฅผ ํ์ต.
- GMF, MLP, NeuMF without pretraining์ Adam, NeuMF with pretraining์ SGD๋ฅผ Optimizer๋ก ์ฌ์ฉ.

![Untitled](/images/2022/NCF/t11.png)

GMF, MLP ๋ ๋ค ๊ฐ์ embedding size๋ฅผ ๊ฐ์ ธ์ผํ๊ธฐ ๋๋ฌธ์ performance์ limit์ด ์์ ์ ์์ผ๋ฉฐ, optimal embedding size๊ฐ ๋ง์ด ๋ณํํ๋ dataset์ optimal ensemble์ ๋ผ ์ ์์ ์ง๋ ๋ชจ๋ฅธ๋ค.

### 3.4.1 Pre-training

๋จ์ํ GMF์ MLP๋ฅผ concatํ์ฌ objective function์ผ๋ก Adam์ ์ฌ์ฉํ๋ ๊ฒ์ gradient-based optimization ๋ฐฉ๋ฒ๋ค์ ํน์ฑ์ local optima๋ก ๋น ์ง ์ ์๋ค. ์ค์ ๋ก ๋ฅ๋ฌ๋ ๋ชจ๋ธ์ ์๋ ด๊ณผ ์ฑ๋ฅ์ initialization์ด ์ค์ํ ์ญํ ์ ํ๋ค๊ณ  ์๋ ค์ ธ ์๋ค. ๋ฐ๋ผ์ ๋ฌด์์์ initialization์ ํ์ฌ NeuMF๋ฅผ ํ์ตํ๋ ๊ณผ์ ์์ parameter๋ค์ updateํ๋ ๊ฒ๋ณด๋ค convergence(์๋ ด)๋์ด ์๋ GMF์ MLP์ parameter๋ค์ ๊ฐ์ ธ์ ์ฌ์ฉํ๋ ๊ฒ์ด ๋ performace๊ฐ ์ข๋ค.

# 4. Experiments

์คํ์ 3๊ฐ์ง์ ์ง๋ฌธ์ ๋ฐํ์ผ๋ก ์งํํ๋ค.

Q1) Do our proposed NCF methods outperform the state-of-the-art implicit collaborative filtering methods?

- Performance comparison

Q2) How does our proposed optimization framework (log loss with negative sampling) work for the recommendation task?

- Log loss with Negative sampling

Q3) Are deeper layers of hidden units helpful for learning from user-item interaction data?

- Is deep learning helpful?

### 4.1 Experimental Settings

![Untitled](/images/2022/NCF/t12.png)

- Dataset : MovieLens์ Pinterest dataset ์ฌ์ฉํ๋ฉฐ, ์ต์ 20๊ฐ์ rating์ด๋ pin์ ๋จ๊ธด user์ data๋ง์ผ๋ก ํ์ต์ ์งํํ๋ค.
- Evaluation Protocols : Hit Ratio, NDCG
- Baselines : NCF method(GMF, MLP, NeuMF)์ ๋น๊ตํ๊ธฐ์ํด baseline์ผ๋ก ItemKNN, BPR, eALS๋ฅผ ์ฌ์ฉํ๋ค.
- Parameter Settings :
    - Binary Log loss function(BCE)
    - Sampling four negative instances per positive instance
    - randomly initialized model parameters with (Gaussian Distribution, mean=0, std=0.01)
    - optimizer : Adam(GMF, MLP), SGD(NeuMF)
    - batch size : [128, 256, 512, 1024]
    - learning rate : [0.0001, 0.0005, 0.001, 0.005]
    - predictive factors and evaluated the factors : [8, 16, 32, 64]
    - NeuMF with pre-training, ฮฑ was set to 0.5
- Ex) predictive factor๊ฐ 8์ด๊ณ , model architecture๊ฐ โ32 โ 16 โ 8โ์ธ model์ ๋ง๋ ๋ค๊ณ  ๊ฐ์ ์ ํ์. embedding size๋ 16์ด ๋๋๋ฐ, ๊ทธ ์ด์ ๋ user์ item vector์ ๋ํด concat์ ํด์ฃผ๊ธฐ ๋๋ฌธ์ embedding ๊ณผ์ ์์ ์ฒซ ๋ฒ์งธ hidden layer input์ ์ ๋ฐ์ผ๋ก embeddingํด์ผ concatํ์ ๋, size๊ฐ ๊ฐ์์ง๋ค.

### 4.2 Performance comparison

![Untitled](/images/2022/NCF/t13.png)

NeuMF์ ์ฑ๋ฅ์ด ๊ฐ์ฅ ์ข์ผ๋ฉฐ, ๋จ์ผ ๋ชจ๋ธ์ ๊ฒฝ์ฐ์๋ ์คํ๋ ค MLP๋ณด๋ค GMF์ ์ฑ๋ฅ์ด ๋ ์ข๋ค.

### 4.3 Log loss with Negative sampling

![Untitled](/images/2022/NCF/t14.png)

Loss ๊ฐ์ด ์ค์ด๋๋ ๊ฒ์ ํตํด interaction function์ ํ์ตํ๋๋ฐ ์ ์ ํ๋ค๋ ๊ฒ์ ์ ์ ์๋ค.

### 4.4 Is deep learning helpful?

![Untitled](/images/2022/NCF/t15.png)

Layer์ depth๊ฐ ๋์ด๋  ์๋ก ์ฑ๋ฅ์ด ๊ฐ์ ๋๊ณ  ์์์ ์ ์ ์๋ค.
