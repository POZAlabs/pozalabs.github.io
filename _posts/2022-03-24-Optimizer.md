---
title: “Optimizer”
layout: post
date: 2021-03-16
headerImage: False
tag:
- Perceptron
- MLP
- Optimizer
- Gradient Descent
//star: true
category: blog
author: kwanho
description: Optimizer summary
---

# Optimizer
## - cost에 따라 어떻게 hyper-parameter를 업데이트할것인가?

이번 포스팅에서 다룰 기술은 바로 Optimizer입니다.

좋은 딥러닝 개발자가 되기위해서는 기본적인 딥러닝 구조를 기초부터 알아가는 과정이 매우 중요합니다. 딥러닝 모델로 데이터를 입력받아 각각의 layer를 통과하고 cost function을 통해서 cost를 감소시키면서 학습데이터에 맞게 모델을 조정하는 과정이 딥러닝 모델 학습의 대략적인 과정이라고 할 수 있습니다.

그러면 여기서 의문점. 그래서 cost는 어떻게 감소시키는데?

여기서 사용되는 알고리즘이 바로 Optimizer! Optimizer는 딥러닝에서 중요한 부분중의 하나인 cost를 줄이는 과정을 담당하고 있으며 좋은 딥러닝 개발자가 되기위해서 기본적으로 알아야할 알고리즘 중에 하나이기에 이번 포스팅의 주제로 선정하였습니다.

이번 포스팅은 아래와 같이 진행됩니다.
1. What is optimizer?
2. Gradient descent
3. 문제점
5. optimizer의 종류
6. 최신 optimizer
7. 정리
8. reference

## 1. What is Optimizer?
딥러닝을 처음 접하는 초심자들을 위해 간단한 비유를 한다면 필자는 종종 데이터를 몸, 딥러닝 모델을 옷에 비유할 수 있습니다. 여기서 optimizer는 옷을 수선하는 방식이라고 생각하면 편할 것같습니다.

데이터(몸)에 맞는 딥러닝 모델(옷)을 만들기 위해서는 딥러닝 모델의 layer와 이외의 알고리즘에 존재하는 hyper-parameter를 데이터에 맞게 최적화 해야한다. 그래서 supervised learning을 기준으로 설명하면 target data(견본 옷)가 존재하고 input data(몸의 실루엣 정보)가 존재하면 딥러닝 모델을 통과한 input data와 target data의 사이의 오차를 계산하는 cost function을 먼저 결정하고 이러한 cost function의 cost를 줄이기 위해 hyper-parameter를 수정해야 하는데 여기서 사용되는 것이 바로 optimizer이다. 이렇듯 optimizer가 모델에 끼치는 영향은 

Optimizer는 cost function에서 계산되는 cost를 극소점으로 가까이 하기 위해, hyper-parameter를 업데이트하게되는데 어떻게 업데이트 할것인지 결정하는 알고리즘이다.

---

## 2.Gradient Descent

위키피디아에 gradient descent를 검색하면
“In mathematics gradient descent is a first-order iterative optimization algorithm for fining a local minimum of differentiable function.”이라고 나온다. 직역하면 “미분가능한 함수의 극소값을 찾는 1차 반복 최적화 알고리즘”이라고 나온다. 앞서 설명햇듯, 딥러닝 모델은 학습데이터를 반복적으로 모델에 학습시켜 데이터에 맞게 딥러닝 모델의 parameter를 수정하여 데이터에 딥러닝 모델을 최적화 하여 생성된다고 하였습니다.

(빠른 진행은 위해 앞으로 gradient descent를 GD라 부르겠습니다.)

여기서 반복, 최적을 강조하였듯 두가지가 optimizer의 가장 중요한 성능이라고 할 수 있습니다. 그래서 초기 Optimizer는 GD를 기반으로 개발되었습니다. 매우 기본적이며 이해가 쉬운 방법이기에 대부분의 블로그나 책에서 GD를 예시로 설명하는 것이 바로 그 이유이며 필자도 동일하게 GD를 예시로 optimizer에 대한 전반적인 설명을 할 것입니다.

그러면 본격적으로 GD가 무엇이냐?
Gradient Descent, 경사하강법이라고 하는 이 방법은 w를 미지수로 가지는 J(w)를 최소화 하는 방법입니다. 

(W,b를 통해 계산된 target, output의 오차의 평균 수식)
(GD수식)

위의 두번째 수식이 cost function입니다. 계속 언급했듯, 딥러닝 모델 학습 데이터에 모델을 반복적인 학습으로 최적화 시키는 작업이라고 하였고, 최적화라고 할 수 있는 인자가 cost라고 할  수 있고 또 이런 값을 최소로 만든다면, 딥러닝 모델이 데이터에 최적화 되었다고 얘기할 수 있을 것입니다.

그래서 학습시 cost function에 optimizer를 적용하여 모델 파라미터 수정을 빠르게 할것인지 천천히 진행할것인지 혹은 어떤식으로 조정할 것인지 결정 할 수있습니다.

설명은 간단히 하기 위해 bias를 제외하고 weight와 cost f에 대해서만 설명하겠습니다. 최초 weight값이 존재하며 이를 업데이트 하기 위해서는 learning rate와 cost, weight의 미분 값를 곱하여 업데이트 해줍니다. 이런식으로 전체데이터에 대해 반복적으로 weight를 조정하면서 데이터에 모델을 최적화 시키는 겁니다.
(bias도 동일합니다.)

Q. 최초학습 때, w, b는 어떻게 설정하는데?
초기 딥러닝 모델들은 일반적으로 w,b를 random값으로 설정하여 최초 학습을 시작하였습니다.

---

## 3.문제점 
그러나 딥러닝을 조금이라도 해본 분들이라면 눈치 챘을 수도 있겠지만, 아무도 GD를 사용하지 않습니다. 왜냐? 보다 좋은 optimizer가 많이 개발되었기 때문입니다.

### 그러면 GD는 어떤 문제가 있을까요?
### 첫번째, 학습속도가 매우매우 느립니다.
GD는 한번 파라미터를 업데이트 할때, 학습데이터 모두에 대해 cost를 계산한 뒤에 업데이트합니다.

### 그래서 이게 왜 문제가 되는데???

GD는 컴퓨터가 개발되기도 전에 나온 이론이고 해당 기술을 머신러닝 목적으로 사용했을 때에도 대략 80~90년대에 적용했었습니다. 그러나 이전에 GD가 사용될 떄는 데이터의 양이 매우 적었고 GD를 적용하는데 아무 문제가 되지 않았습니다. 왜냐하면 얼마되지도 않는 데이터를 쭉 훓어보고 한번 업데이트하면 그만 이였으니까요. 그러나 딥러닝에 GD를 적용하는 것은 또 다른문제입니다. 왜냐하면 딥러닝은 기본적으로 무수히 많은 종류와 많은 양의 데이터를 이용하여 모델을 학습하기 때문입니다. 그렇기에 전체데이터를 다 훓어보고 한번 업데이트를 하게 되면 적당한 성능의 모델을 개발하기위해 천문학적인 리소스가 필요하게 될것입니다.


### 두번째, local minimum에 빠질 수 있다.
Learning rate를 어떻게 설정하냐에 따라 다르겠지만 일반적인 상황을 가정했을 때, 전체 데이터를 보고 조금씩 weight optimization을 진행하게 되는데, 그림에서 알수 있듯이 convex한 부분 근처에서는 미분값이 작아지면서 weight값이 조금씩 업데이트됨을 알 수 있다. 그런데 만약 극소점이 아닌 곳에서 미분값이 0이 되어버린다면? 학습은 더이상 진행되지 않게 된다. 그러면 이 모델은 자동적으로 under-fitting이 될 수 밖에 없다.

## 4. Optimizer의 종류

위에서 GD의 문제점에 대해 얘기했다. Stochastic gradient descent부터 많은 딥러닝 개발자들이 무지성으로 사용하는 Adam까지 많은 종류의 Optmizer가 GD의 문제점을 보완하기 위해 개발되었다.


**Batch Gradient Descent**

BGD? 이게 머야?라고 생각 할 수 있습니다. 일반적으로 우리가 python에 사용하는 GD가 BGD입니다.

무슨 말이야?
딥러닝은 컴퓨터 자세하게는 GPU로 학습하게 됩니다. 학습하기 위해서는 데이터를 GPU의 메모리에 올리고 메모리에서 데이터를 가져와서 학습하게 되는데 데이터의 양이 많아 지게되면 모든 학습데이터를 GPU의 메모리에 올릴 수가 없습니다. 그래서 나온 개념이 batch. 간단하게 데이터를 batch size만큼 쪼개어서 각 batch마다 데이터를 구하여 모든 batch의 데이터로 cost function을 구하여 데이터를 update하는 방식으로, GPU장비의 한계로 개발된 기술로 일반적으로 말하는 GD와 BGD는 동일한 기술입니다.

**Stochastic Gradient Descent**

SGD는 GD와 수식은 동일하며 단지 업데이트 하는 방식이 다른 기술입니다. 앞서 GD의 단점으로 학습속도가 매우 느리다고 했습니다. 이를 보완하기 위해 나온 기술로, BGD를 설명할때 batch단위로 끊어서 업데이르틀 한다고 하였는데 SGD는 매 batch마다 cost function을 재설정하여 업데이트 하는 방식으로 데이터가 많지 않다면 그만큼 함수가 간단해 질것이고 다시말해 업데이트의 변동성이 매우 커진다는 말입니다. 그래서 매 batch마다 업데이트가 많이 일어나게 되기 때문에 GD보다 학습속도가 빠르게 됩니다. 그러나 여기에도 문제가 있습니다.

첫번째, 학습 방향이 너무 급격하게 변한다.
두번째, 하나의 parameter에 너무 의존적이다.
SGD는 learning rate에 따라 빠르게 변할수도 늦게 변할 수도 있기에 learning rate에 너무 의존적이게 되어 learning rate마다 성능차이가 심합니다. 그래서 학습중간 마다 learning rate를 조정해야합니다.

**Momentum Optimizer**

수식 : $$V_t = m \times$$



**Nesterov Accelerated Gradient**

**Adaptive Gradient Descent**

**Root Mean Square Propagation Optimizer**

**Adaptive Momentum Optimizer**


## 5. Brand-new Optimizer

## 6.정리

## 7.reference





![activation1.png](/assets/images/Activation_function/activation1.png)

Momentum optimizer
SGD의 경우, learning rate에 따라서 이동이 가능하지만 local minimum에 빠질수 있다. 그래서 여기서 관성의 개념을 도입한다.
weight를 구할때 이전 weight값에 momentum값을 곱하여 업데이트 될때마다 이전의 값을 줄이고 업데이트 되는 힘을 많이 받겠다는 의미로
(일반적으로 0.9로 momentum 상수를 많이 사용)
그래서 local minimum에 빠질 가능성이 SGD보다 현저히 줄어든다.
단점 : 최적 parameter를 구했는데 지나치게된다면? 엉뚱한 parameter를 찾을 수 있다.

NAG
Momentum의 문제점이 무엇이냐? Minimum근처가 sharp했다고 가정하자, 그러면 minimum point를 지날수 있고 지난 뒤에도 fluctuation해서 minimum까지 오는 시간이 오래 걸리 수 있다. 그래서 momentum을 미분값을 구하는 point에서 구해서 관성 통해서 빠르게 수렴함과 동시에 기울기가 변하는 위치에서 momentum보다 제동력이 뛰어나다

Adaptive gradient
각 parameter에 개별 기준을 적용. 계속해서 변했던 parameter는 변화량을 줄이고 변하지 않았던 parameter에 가중치를 크게 주겠다. 
근데 학습이 어느정도 진행되면 대부분의 parameter가 변동이 된 사항이라 업데이트가 매우매우 느리다.
w식을 보면 learning rate만큼 뺀 값만큼 update하게 되는데 메인 식에서 이를 보게되면 많이 업데이트되면 update가 느리게 되고 update가 적었으면 많이 변동함을 알 수 있다. -> 학습이 어느정도 진행되었는데 최적값을 찾지 못하였다면? -> 업데이트는 느려지고 학습시간이 동일하다면 그만큼 성능이 떨어지게 된다.

RMSprop
위 식에서 원 parameter와 update parameter에 각각 가중치를 더해서 학습이 길어지더라도 최소한의 step은 이동가능

Adam optimizer
근본적인 방식은 momentum을 통해 관성을 사용하고  update가 많이 되서 안정화된 parameter는 update를 늦추겠다는 컨셉

