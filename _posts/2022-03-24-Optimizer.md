---
title: Optimizer
layout: post
date: 2022-03-30
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

이번 포스팅은 아래와 같은 순서로 진행합니다.
1. What is optimizer?
2. Gradient descent
3. 문제점
5. optimizer의 종류
6. 최신 optimizer 및 이슈
7. 정리
8. reference

## 1. What is Optimizer?
딥러닝을 처음 접하는 초심자들을 위해 간단한 비유를 한다면 필자는 종종 데이터를 몸, 딥러닝 모델을 옷에 비유할 수 있습니다. 여기서 optimizer는 옷을 수선하는 방식이라고 생각하면 편할 것같습니다.

데이터(몸)에 맞는 딥러닝 모델(옷)을 만들기 위해서는 딥러닝 모델의 layer와 이외의 알고리즘에 존재하는 hyper-parameter를 데이터에 맞게 최적화 해야한다. 그래서 supervised learning을 기준으로 설명하면 target data(견본 옷)가 존재하고 input data(몸의 실루엣 정보)가 존재하면 딥러닝 모델을 통과한 input data와 target data의 사이의 오차를 계산하는 cost function을 먼저 결정하고 이러한 cost function의 cost를 줄이기 위해 hyper-parameter를 수정해야 하는데 여기서 사용되는 것이 바로 optimizer이다. 이렇듯 optimizer는 모델 성능에 많은 영향을 끼칩니다.  
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
도대체 무슨 말이야?  

딥러닝은 컴퓨터 자세하게는 GPU로 학습하게 됩니다. 학습하기 위해서는 데이터를 GPU의 메모리에 올리고 메모리에서 데이터를 가져와서 학습하게 되는데 데이터의 양이 많아 지게되면 모든 학습데이터를 GPU의 메모리에 올릴 수가 없습니다. 그래서 나온 개념이 batch. 간단하게 데이터를 batch size만큼 쪼개어서 각 batch마다 데이터를 구하여 모든 batch의 데이터로 cost function을 구하여 데이터를 update하는 방식으로, GPU장비의 한계로 개발된 기술로 일반적으로 말하는 GD와 BGD는 동일한 기술입니다.

**Stochastic Gradient Descent**

SGD는 GD와 수식은 동일하며 단지 업데이트 하는 방식이 다른 기술입니다. 앞서 GD의 단점으로 학습속도가 매우 느리다고 했습니다. 이를 보완하기 위해 나온 기술로, BGD를 설명할때 batch단위로 끊어서 업데이르틀 한다고 하였는데 SGD는 매 batch마다 cost function을 재설정하여 업데이트 하는 방식으로 데이터가 많지 않다면 그만큼 함수가 간단해 질것이고 다시말해 업데이트의 변동성이 매우 커진다는 말입니다. 그래서 매 batch마다 업데이트가 많이 일어나게 되기 때문에 GD보다 학습속도가 빠르게 됩니다. 그러나 여기에도 문제가 있습니다.

첫번째, 학습 방향이 너무 급격하게 변한다.  
두번째, 하나의 parameter에 너무 의존적이다.
SGD는 learning rate에 따라 빠르게 변할수도 늦게 변할 수도 있기에 learning rate에 너무 의존적이게 되어 learning rate마다 성능차이가 심합니다. 그래서 학습중간 마다 learning rate를 조정해야합니다.

**Momentum Optimizer**

모멘텀은 위의 SGD에서 발생하는 문제점을 해결하기 위해 개발된 기술이다. 개념은 간단하다. SGD와 MO의 수식을 비교해보자. 미분 term을 비교하면 MO의 경우, 이전에 사용했던 미분값을 추가로 더해줌으로써 이전 step에서 미분값이 컸다면 큰값을 업데이트해주며, 미분값이 작다면 작은값을 업데이트 해주게된다.  

이런 의문을 가질수 있다. SGD랑 다른게 없잖아?  
그렇게 생각할 수 있습니다. 그러나 미분값에 이전에 사용했던 미분값을 넣게되어, 갑자기 함수의 기울기가 급격하게 변한다 하더라도 이전에 계산해놓은 값으로 인해, local minimum에 빠질 확률이 줄어듭니다. 그렇기에 SGD와 비교해 학습방향이 다소 완만해지며, learning rate에 크게 영향을 받지않고 학습을 할 수 있습니다.


**Nesterov Accelerated Gradient**

NAG는 MO을 개량하기 위해 개발된 기술입니다. 만약 global minimum근처가 매우 sharp하다는 가정을 해봅시다. 그렇다면 MO의 경우에는 관성 term의 영향으로 인해 global minimum근처를 계속해서 왔다갔다하면서 학습시간이 불필요하게 늘어납니다. 이를 해결하기위해 NAG가 개발되었습니다. MO와 NAG의 수식을 비교해봅시다. MO의 경우, 미분텀에 과거의 미분값을 넣어서 현재의 미분값을 계산하게됩니다. 그러나 NAG의 경우에는 현재의 미분값을 넣어서 weight를 업데이트하게 됩니다. 다시말해 MO는 과거의 미분값으로 현재의 미분값을 구하여 weight를 업데이트하지만 NAG의 경우에는 과거의 momentum값으로 업데이트되어 이동한 위치의 미분값을 구하여 업데이트를 하게됩니다.

정리하면 MO은 과거로 현재의 값을 찾는다면 NAG는 과거값으로 미래값을 구한다고 생각하면 이해가 쉬울것입니다.

여기까지 momentum term을 이용하여 optimizer를 개발한 기술입니다.

**Adaptive Gradient Descent**

지금부터는 learning rate를 조절해가면서 학습하는 방법에 대해 설명하겠습니다. AGD는 SGD에서 파생되었으며, learning rate update term을 보게되면 과거의 미분값이 크면 클 수 록 learning rate를 줄이는 식으로 학습함을 알 수 있습니다. 이 방법이 왜 유용하냐? 최근 딥러닝은 과거의 딥러닝 모델보다 더 많고 더 다양한 hyper-parameter를 지니게 됩니다. 그러나 모든 parameter를 동일하게 학습하게된다면 좋은 성능의 모델 생성이 힘들 수 있습니다. 또한 학습이 완료된 parameter가 있다고 하여도 동일하게 학습되면 잘 설정되어있던 parameter도 변할수 있습니다. 그렇기 때문에 learning rate term을 두어, 학습이 많이 진행된 parameter에 대해서는 learning rate를 줄여 학습을 더디게 하고 학습이 거의 진행되지 않은 parameter에 대해서는 learning rate를 크게 만들어 학습속도를 빠르게 가져가는 것입니다.
그러나 이런 Adagrad에도 문제점이 존재한다. 학습이 많이 된 parameter는 learning rate를 줄여 학습속도를 줄이는 컨셉은 좋으나, 학습이 오래되어 버린다면 뒤의 learning rate term이 계속해서 증가하여 나중에는 값이 업데이트되지 않는 현상이 발생한다.

**Root Mean Square Propagation Optimizer**

RMSprop는 위의 Adagrad의 단점을 보완하기 위하여 개발되었습니다. 개념은 간단합니다. learning rate term에서 미분텀의 영향을 크게하여 학습이 오래되어도 Adagrad에 비해 학습이 원활하게 되도록 하는 기술입니다.

**Adaptive Momentum Optimizer**

지금까지 이 optimizer를 위해 달려왔다고 해도 과언이 아닙니다. 바로 대부분의 딥러닝 개발자들이 무지성으로 사용하는 유명한 Adam optimizer. 생각없이 그냥 사용하여도 좋은 이유는 이미 몇년전부터 많은 실험을 통해 그 성능과 효과가 입증이 되었기 때문입니다. 그러나 알고 쓰는 것과 모르고 쓰는 것은 하늘과 땅차이! 그러므로 Adam에 대해서 설명하겠습니다. 많은 개발자들이 사용해서 어려운 개념이라고 생각하기 쉽지만 이 기술도 매우 간단합니다. 바로 이전에 설명드렸던 momentum과 RMSprop의 개념을 짬뽕한 기술입니다. momentum, learning rate term을 각각 계산해서 m, learning rate를 step마다 계산하여 parameter를 update합니다.


## 5. Brand-new Optimizer and Essue

제목 : AngularGrad: A New Optimization Technique for Angular Convergence of Convolutional Neural Networks
요약 : SGD부터 Adam까지 수식을 보든 결과를 보든 parameter-Cost function그래프에서 수직방향으로 영향을 많이 받기 때문에, 계곡사이에 흐르는 강의 모양을 한 function에서는 수직방향의 영향을 받아 극소점까지의 수렴이 매우 느리며 극소점 근처에서도 oscilating하는 현상이 발생하여 수직방향의 성분을 줄여 안정적으로 극소점을 찾아가는 기술을 개발하였습니다.
그러나 저는 다소 회의적입니다. 왜냐면 해당 논문이 2021 IEEE에 등재될정도로 우수한 논문임에는 이견이 없지만 그러한 case가 흔치 않기에 그러한 function의 극소점을 찾는 것을 사전에 알고 접근하지 않는 이상 Adam을 사용하는것이 오히려 generall하게 우수한 성능을 보장할 것이라고 생각합니다.

제목 : SGD가 Adam보다 일반화면에서 오히려 우수하다?!
링크 : https://medium.com/geekculture/a-2021-guide-to-improving-cnns-optimizers-adam-vs-sgd-495848ac6008
요약 : 다양한 실험을 통해 Adam의 성능이 증명되었으나 굳이 저렇게 optimizer를 복잡하게 가져갈 필요할까? 오히려 over-fitting이 가속화된다고 주장합니다. 일반적으로 Adam은 초기값 설정 안해도 성능이 좋다고 하였는데 실험결과를 보니 초기값을 제대로 설정하냐 안하냐에 따라 Adam의 성능차이가 존재하였습니다. 다시말해 초기값을 어떻게 잡냐에 따라 Adam이 오히려 SGD보다 일반화에 실패하는 경우가 있다. 그러나 다른 논문에서는 딥러닝이 더욱더 딥해지면서 더 많은 parameter를 사용하기 때문에 SGD보다 Adam이 좋다고 주장합니다. 그래서 필자는 딥한 모델을 학습시에는 Adam이 항상 SGD보다 좋다고 주장하고, 얉은 모델에서는 굳이 Adam을 고집할 필요가 있을까? 라고 질문을 던졌습니다. 이렇듯 딥러닝에 대한 개발이 오래전부터 활발히 진행됬음에도 불구하고 여전히 basic한 기술에 대해 언쟁이 오가는 만큼 기초를 잘 잡고 있어야 변하는 흐름에 빠르게 대응이 가능할 것같습니다.

## 6.정리

지금까지 optimizer의 개념과 gradient descent를 통해 어떤 식으로 사용되는지 확인하였고 동시에 optmizer의 기술흐름에 대해 알아보았습니다. 요약하면 optimizer는 cost function에서 weight, bias를 이용하여 이러한 parameter를 어떤식으로 수정해 나갈것인지 결정하는 알고리즘이며 이중 많이 언급되는 gradient descent는 optimizer의 개념을 이해하기 쉬운 기술로, 최초 parameter에서 cost function을 통해 값을 계산하고 이때의 미분값과 learning rate로 중학교시간에 배우는 극소점을 찾아가는 과정과 동일하며 GD의 느린 학습속도를 보완하기 위해 나온 SGD. 그리고 여기서 momentum과 learning rate개념을 적용하여 개발된 MO와 Adagrad 그리고 극소점 근방이 sharp하거나 oscilating이 심하면 학습이 오래걸리는 MO의 문제점을 보완하기 위해 개발된 NAG. 그리고 학습이 길어지면 학습이 거의 되지않는 Adagrad의 문제점을 보완하기 위해 개발된 RMSprop. momentum과 learning rate term의 개념을 모두 종합하여 개발된 Adam이렇게 정리 할 수 있겠습니다.


## 7.reference





![SGD.png](/Users/pkh/desktop/SGD.png)
