# AI-X-Deeplearing

# Title : Arcface 논문에 따른 얼굴 인식 모델 구현

# Members
      김민수 수학과 maengjja@naver.com
      김윤기 수학과 dbsrl7665@naver.com
      
# Index
## Ⅰ. Proposal
### · Motivation
### · Goal
## Ⅱ. Datasets
## Ⅲ. Methodlogy
### · CNN기반의 ResNet구조
### · 손실함수 : Arcface loss function 사용
## Ⅳ. Evaluation & Analysis
## Ⅴ. Related Work
## Ⅵ. Conclusion


# Ⅰ. Proposal(option A)
## Motivation
MIT가 2022년에 선정한 10대 기술 중 하나로, 비밀번호를 대체하는 기술 중 생체인증이 떠오르고 있습니다. 그 이유는 생체 인증이 비밀번호를 대체하면서 핸드폰 잠금해제를 하거나 계좌이체를 할 때 사용자 입장에서 비밀번호를 외우거나, 입력을 해야하는 등 불편함이 생기지 않고 센서에 인식만 시켜주면 되는 편리함이 생기게 되기 때문입니다. 이러한 생체인증 중에서, 저희는 얼굴 인식에 주목을 하였습니다. 그 이유는 다른 인식 기기 없이 카메라만으로 동작 가능하고 터치 같은 추가적인 동작없이 얼굴을 인식이 가능한 올바른 위치에 두기만 해도 인식이 가능하여 편리함에 목적을 둔 생체 인증에 잘 부합한다고 생각을 하였습니다. 저희는 딥러닝을 활용한 얼굴인식으로 주제를 잡고 그에 대한 fundamental한 work중 하나인, arcface논문을 참고하여 얼굴 인식 모델을 구현하려고 하였습니다.

## Goal
저희의 최종적인 목표는 딥러닝 얼굴인식 모델에 사용되는 구조적인 것들의 이해와 구현에 있습니다. 구조로는 CNN(Convolution neural network), Res Net과 Arcface Loss가 있습니다. 모델 구현을 위해서 얼굴인식에 자주 사용되는 데이터셋 중 하나인 LFW dataset의 일부를 활용하여, 모델을 train하고 데이터 셋의 나머지 부분을 활용하여 모델을 test할 것입니다.

# Ⅱ. Datasets
LFW Face Database를 사용하겠습니다.

해당 데이터 셋은 5748명의 사람들의 13000장 이상의 사진들로 이루어져있습니다. 각각의 사진 속 인물의 이름이 라벨로 붙어있고 해당 사진의 확장자는 JPEG으로 되어있습니다. 또한 해당 사진들의 픽셀은 62*47로 이루어져있습니다.

예시로는 다음과 같습니다.

<img width = "100" alt = "image" src \"https://github.com/yunki00/AI-X-Deeplearing9/assets/132141925/30db4a93-6be3-4c7a-89ca-cf1f3dba1125">

<img width = "100" alt = "image" src \"https://github.com/yunki00/AI-X-Deeplearing9/assets/132141925/92d63c2c-a6bc-40fe-9205-75c4a4187d85">


https://www.kaggle.com/datasets/atulanandjha/lfwpeople

# Ⅲ. Methodlogy
## CNN기반의 ResNet구조
### CNN
일반적인 순방향 신경망을 사용하여 이미지를 학습하기 위해서는 공간데이터를 1차원으로 변환하기 때문에 정확한 패턴을 인식하기 어려워집니다. 또한 고차원 데이터별로 크기가 더 커질 수록 모델의 파라미터 수도 급격하게 증가하게 됩니다. 이미지의 패턴을 효과적으로 인식하기 위해서 하나의 픽셀만 바라보는 것이 아니라 하나의 픽셀 주위의 픽셀과의 연관성을 봐야합니다. 따라서 다음 사진과 같은 convolution filter를 사용하여 많은 계층을 만든 것을 CNN(Convolution Neural Network)이라고 합니다.

convolution이라는 단어는 합성곱이라는 뜻으로 CNN에서 사용하는 이미지에 대한 convolution 연산은 원래 이미지 데이터와 convolution filter의 두 행렬의 각각의 성분을 곱한 뒤 더한 값입니다. 따라서 아래 사진과 같이 나타납니다.

1.

<img width="321" alt="image" src="

https://github.com/yunki00/AI-X-Deeplearing9/assets/132141925/de20eb11-54ea-488c-8c90-69f7247046da">

2. 

<img width="364" alt="image" src="https://github.com/yunki00/AI-X-Deeplearing9/assets/132141925/8c5b2003-e761-4d5f-a4a3-921dc3fb5b92">

위 과정을 필터를 옮겨가며 반복하면 아래와 같이 출력데이터가 나오게 됩니다.

3. 

<img width="526" alt="image" src="https://github.com/yunki00/AI-X-Deeplearing9/assets/132141925/c509cba4-325d-45fe-bbf8-8d369fb972f4">

위 사진은 기존 이미지의 데이터를 그대로 convolution filter와 연산한 것입니다. 위와 같이 연산하면 다음계층으로 전달 될 이미지 데이터의 크기가 다음과 같이 정해지게 됩니다.

O = N - F + 1

(N : 입력데이터의 크기, F : 필터(kernel)의 크기, O : 출력데이터 크기 )

따라서 우리의 필요에 따라 stride의 크기를 다르게 하거나 padding 또는 pooling연산을 이용하여 output 데이터를 우리가 원하는 크기로 만들 수 있습니다. 
#### padding : convolution 연산을 하기 전 이미지 데이터 주변에 특정 값을 채워 크기를 늘리는 방법입니다.
아래 사진은 빈칸에 특정 값 0을 채워 연산한 것입니다.

4. 

<img width="391" alt="pad" src="https://github.com/yunki00/AI-X-Deeplearing9/assets/132141925/3288a64b-89a3-478b-a7df-86c3995f7457">

#### pooling : 입력데이터를 일정 크기의 영역으로 나누고 그 안에 있는 데이터들의 요약통계량(평균, 최댓값, 최솟값, 가중합산)을 사용하여 데이터의 크기를 줄이는 방법입니다. 우리의 코드에서는 average pooling을 사용합니다.
5. 

<img width="515" alt="pool" src="https://github.com/yunki00/AI-X-Deeplearing9/assets/132141925/c652fcc5-8042-4c6d-a391-99fb4ecb832c">

#### stride : stride는 보폭이란 의미로 필터를 적용하는 간격입니다. stride를 크게 하면 다음 그림과 같이 출력 데이터의 크기가 감소하게 됩니다. 우리의 코드에서는 stride 2를 사용하여 downsample을 하거나 stride 1 사용합니다. 
6. 

<img width="329" alt="stride" src="https://github.com/yunki00/AI-X-Deeplearing9/assets/132141925/c3bdad17-630a-4aa6-a95e-ff992547a466">

#### loss function : 학습 중에 알고리즘이 얼마나 잘못 예측하는 정도를 확인하기 위한 함수입니다. 우리의 코드에서는 Arcface loss function을 사용합니다. 또한 해당 부분은 코드의 header부분에 위치하고 있습니다. 

#### gradient descent : 손실함수(loss function)의 최소 지점을 찾기 위해 경사가 가장 가파른 곳을 찾아서 현재 위치에서 그 방향으로 내려가는 방법으로 손실함수의 최솟값을 찾을 수 있게 도와주는 역할을 합니다.
7. 

<img width="299" alt="image" src="https://github.com/yunki00/AI-X-Deeplearing9/assets/132141925/5f95066c-5aaa-4d9f-ad14-d48b4eff93c9">

#### Activation fucntion : 입력 신호의 총합을 출력 신호로 변환하는 함수로서 입력신호의 총합이 활성화를 일으키는지 아닌지 정하는 역할을 합니다. 우리의 코드에서는 ReLU을 사용합니다. 

8. 

<img width="232" alt="image" src="https://github.com/yunki00/AI-X-Deeplearing9/assets/132141925/3082b8b7-4cb1-4350-ac4e-aff2ca6e97e4">

#### batch normalization : 각 layer에 따라서 입력 값의 분포가 편향되어 있을 수 있습니다. 이러한 편향된 분포는 activation을 사용하면 0이 나오거나 지나치게 편향된 값이 나올 수 있으므로 데이터 분포를 정규화합니다.

9. 

<img width="443" alt="image" src="https://github.com/yunki00/AI-X-Deeplearing9/assets/132141925/3bef11c2-e62e-4f2c-9f2a-777226ac343d">
<br/><br/>
CNN의 구조를 간략하게 나타내면 다음과 같습니다.

10.

<img width="561" alt="image" src="https://github.com/yunki00/AI-X-Deeplearing9/assets/132141925/03215c1f-99a9-4d95-bf78-ac6c5784b641">

### ResNet
기본적인 CNN구조를 사용하면 layer가 일정 수준 이상으로 깊어지면 자유 파라미터들의 개수가 증가하여 학습의 하락을 유발합니다. 이것을 방지하기 위해 layer에 input x의 identity인 x를 더해줘서 연산을 수행하는 ResNet을 사용하여 layer를 깊게 만들어도 학습의 하락이 이뤄지지 않고 잘 학습되는 것을 확인 할 수 있습니다. 우리의 코드에서는 ResNet18을 사용합니다.

11.

![image](https://github.com/yunki00/AI-X-Deeplearing9/assets/132141925/c16a19b6-0e59-42e5-a46a-d9b29f86a39e)

## 손실함수 : Arcface loss function 사용
기존 ResNet을 이용한 얼굴인식 모델의 손실함수인 Softmax loss functuion

12. 

<img width="272" alt="image" src="https://github.com/yunki00/AI-X-Deeplearing9/assets/132141925/0168ce00-cb8a-4254-84fe-525acce54037">

에서 내적연산을 각도를 포함한 연산으로 다시 적어보면 다음과 같이 됩니다.

13.

<img width="346" alt="image" src="https://github.com/yunki00/AI-X-Deeplearing9/assets/132141925/c98f7106-2c65-4110-9b16-32abf581a904">

다음 연산에서 클래스 간 각도에 직접적 margin penalty(m)를 더 해서 계산한 것을 Arcface loss function

14. 

<img width="411" alt="image" src="https://github.com/yunki00/AI-X-Deeplearing9/assets/132141925/6a9e2def-d443-4032-9a65-4c2b4e29e50b">

이라고 합니다. 그렇게 되면 서로 다른 클래스 간 각도의 차이를 더해 주어 더 큰 격차를 만들 수 있어 분변력을 증가시키고 학습을 안정화할 수 있기 때문에 우리는 손실함수로서 Arcface loss function을 사용하겠습니다. Softmax function하고 Arcface funtion을 그림으로 비교해 보면 다음과 같습니다.

15. 

<img width="303" alt="image" src="https://github.com/yunki00/AI-X-Deeplearing9/assets/132141925/a7e2cff1-d07c-4207-95b6-886cc82bc424">

# Ⅳ. Evaluation & Analysis

컴퓨터 사양

      AMD EPYC 7543P (2.8GHZ)	

      NVIDIA A100 X 2	

      512GB RAM

      pyorch 사용

#### Hyper parameter setting

Dataset & Training Loop


      img_shape = (3,112,112)
      
      batch_size = 1024
      
      epoch = 300

margins for arcface

      emb_size = 512
      
      scale = 64
      
      margin = 0.5
      
emb_size는 embedding size인데 이는 이미지가 모델을 거치면 실수형 데이터가 되는데, 그 실수형 데이터를 해당 차원의 벡터에 해당합니다.

Optimizer

      lr_backbone = 0.01
      
      lr_header = 0.01
      
      weight_decay = 1e-3
      
      
The number of classes :  5748

Image shape :  (3, 112, 112)

The number of images :  13233

The number of train images :  10560

The number of test images :  2673





# Ⅴ. Related Work
ArcFace: Additive Angular Margin Loss for Deep Face Recognition : 이미지 12, 13, 14, 15

Deep Residual Learning for Image Recognition : 이미지 11

https://www.technologyreview.com/2022/02/23/1045416/10-breakthrough-technologies-2022

https://github.com/whitesoonguh/CnA_Arcface

https://www.kaggle.com/datasets/atulanandjha/lfwpeople

https://yjjo.tistory.com/8 : 이미지 1, 2, 3

https://amber-chaeeunk.tistory.com/24 : 이미지 4, 6, 10

Do it! 딥러닝 교과서 : 이미지 5

https://process-mining.tistory.com/175 : 이미지 7

https://www.researchgate.net/figure/Commonly-used-activation-functions-a-Sigmoid-b-Tanh-c-ReLU-and-d-LReLU_fig3_335845675 : 이미지 8

https://gaussian37.github.io/dl-concept-batchnorm/ : 이미지 9
# Ⅵ. Conclusion

