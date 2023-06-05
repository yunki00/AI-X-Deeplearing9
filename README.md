# AI-X-Deeplearing

# Title : Arcface 논문에 따른 얼굴 인식 모델 구현

# Members
      김민수 수학과 maengjja@naver.com
      김윤기 수학과 dbsrl7665@naver.com
      
# Index
### Ⅰ. Proposal
### Ⅱ. Datasets
### Ⅲ. Methodlogy
### Ⅳ. Evaluation & Analysis
### Ⅴ. Related Work
### Ⅵ. Conclusion


# Ⅰ. Proposal(option A)
## Motivation
MIT가 2022년에 선정한 10대 기술 중 하나로, 비밀번호를 대체하는 기술 중 생체인증이 떠오르고 있습니다. 그 이유는 생체인증이 가져다주는 편리함에 있는데, 생체 인증이 비밀번호를 대체하면서 핸드폰 잠금해제를 하거나 은행 계좌이체를 할 때 사용자 입장에서 비밀번호를 외우거나 입력을 해야하는 불편함을 느끼지 않고 센서에 인식만 시켜주면 되기 때문에 편리함이 생기게 됩니다. 이러한 생체인증 중에서, 저희는 얼굴 인식에 주목을 하였습니다. 그 이유는 다른 인식 기기 없이 카메라만으로 동작 가능하고 터치같은 추가적인 동작없이 얼굴을 카메라에 올바른 위치에 두기만 해도 인식이 가능하여 편리함이라는 것에 목적을 둔 생체 인증에 잘 부합한다고 생각을 하였습니다. 저희는 딥러닝을 활용한 얼굴인식으로 주제를 잡고 그에 대한 fundamental한 work중 하나로, arcface논문을 참고하여 얼굴 인식 모델을 구현하려고 하였습니다.

## Goal
저희의 최종적인 목표는 딥러닝 얼굴인식 모델에 사용되는 CNN(Convolution neural network)과 ResNet에 대한 이해와 arcface와 그 이전의 얼굴인식 모델과 다른 점(Loss Function)에 대한 이해에 있습니다. 최종적으로 얼굴인식에 자주 사용되는 데이터셋 중 하나인 LFW dataset의 일부를 활용하여, 모델을 train하고 데이터 셋의 나머지 부분을 활용하여 모델을 test할 것입니다.

# Ⅱ. Datasets
LFW Face Database를 사용하겠습니다.

해당 데이터 셋은 1680명의 사람들의 13000장 이상의 사진들로 이루어져있습니다. 각각의 사진 속 인물의 이름이 라벨로 붙어있고 해당 사진의 확장자는 JPEG으로 되어있습니다. 또한 해당 사진들의 픽셀은 62*47로 이루어져있습니다.

https://www.kaggle.com/datasets/atulanandjha/lfwpeople

# Ⅲ. Methodlogy
## CNN기반의 ResNet구조
### CNN
일반적인 순방향 신경망을 사용하여 이미지를 학습하기 위해서는 공간데이터를 1차원으로 변환하기 때문에 정확한 패턴을 인식하기 어려워집니다. 또한 고차원 데이터별로 크기가 더 커질 수록 모델의 파라미터 수도 급격하게 증가하게 됩니다. 이미지의 패턴을 효과적으로 인식하기 위해서 하나의 픽셀만 바라보는 것이 아니라 하나의 픽셀 주위의 픽셀과의 연관성을 봐야합니다. 따라서 다음 사진과 같은 convolution filter를 사용하여 많은 계층을 만든 것을 CNN(Convolution Neural Network)이라고 합니다.

convolution이라는 단어는 합성곱이라는 뜻으로 CNN에서 사용하는 이미지에 대한 convolution 연산은 원래 이미지 데이터와 convolution filter의 두 행렬의 각각의 성분을 곱한 뒤 더한 값입니다. 따라서 아래 사진과 같이 나타납니다.
<img width="364" alt="image" src="https://github.com/yunki00/AI-X-Deeplearing9/assets/132141925/8c5b2003-e761-4d5f-a4a3-921dc3fb5b92">

<img width="517" alt="image" src="https://github.com/yunki00/AI-X-Deeplearing9/assets/132141925/3e8f70a0-ba19-4fc7-b0bf-ab0aa8828e31">

위 사진은 기존 이미지의 데이터를 그대로 convolution filter를 사용하면 다음계층으로 전달 될 이미지 데이터의 크기가 다음과 같이 정해지게 됩니다.
O = N - F + 1

(N : 입력데이터의 크기 F : convolution filter의 크기 O : 출력데이터 크기 )

따라서 우리의 필요에 따라 stride의 크기를 다르게 하거나 padding 또는 pooling연산을 이용하여 output 데이터를 우리가 원하는 크기로 만들 수 있습니다. 
#### padding : convolution 연산을 하기 전 이미지 데이터 주변에 특정 값을 채워 크기를 늘리는 방법
![image](https://github.com/yunki00/AI-X-Deeplearing9/assets/132141925/d369aed6-c355-4dc6-b7a1-10a4fa07cca1)

#### pooling : 입력데이터를 일정 크기의 영역으로 나누고 그 안에 있는 데이터들의 요약통계량(평균, 최댓값, 최솟값, 가중합산)을 사용하여 데이터의 크기를 줄이는 방법
<img width="687" alt="image" src="https://github.com/yunki00/AI-X-Deeplearing9/assets/132141925/79c0a817-a007-4747-a2b5-7b7d3ccf8118">

#### stride : stride는 보폭이란 의미로 필터를 적용하는 간격입니다. 스트라이드를 크게 하면 다음 그림과 같이 출력 데이터의 크기가 감소하게 됩니다.
![image](https://github.com/yunki00/AI-X-Deeplearing9/assets/132141925/ccb709bf-7f87-4ffd-aede-668b2dcd07c6)


### ResNet
기본적인 CNN구조에 layer가 일정 수준 이상으로 깊어지면 결과가 더 안 좋게 나오게 됩니다. 그래서 ResNet구조를 이용해 망이 깊어져도 더 좋은결과가 나올 수 있게 사용할 것입니다.

## 손실함수 : Arcface loss function 사용
기존 Softmax loss functuion인

<img width="272" alt="image" src="https://github.com/yunki00/AI-X-Deeplearing9/assets/132141925/0168ce00-cb8a-4254-84fe-525acce54037">

에서 분변력을 증가시키고 학습을 안정화하기 위해 각도에 직접적 margin penalty를 더 해준 Arcface loss funcyion을 사용할 것입니다. 이 loss function은 다음과 같습니다.

<img width="411" alt="image" src="https://github.com/yunki00/AI-X-Deeplearing9/assets/132141925/6a9e2def-d443-4032-9a65-4c2b4e29e50b">

Softmax function하고 Arcface funtion을 그림으로 비교해 보면 다음과 같습니다.

<img width="303" alt="image" src="https://github.com/yunki00/AI-X-Deeplearing9/assets/132141925/a7e2cff1-d07c-4207-95b6-886cc82bc424">

# Ⅳ. Evaluation & Analysis
# Ⅴ. Related Work
ArcFace: Additive Angular Margin Loss for Deep Face Recognition

https://www.technologyreview.com/2022/02/23/1045416/10-breakthrough-technologies-2022

https://github.com/whitesoonguh/CnA_Arcface

https://www.kaggle.com/datasets/atulanandjha/lfwpeople

https://amber-chaeeunk.tistory.com/24

# Ⅵ. Conclusion
