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
현재 우리 곁에는 얼굴인식 기술이 많이 사용되고 있습니다. 우리가 매일 사용하고 있는 핸드폰에서부터 얼굴인식 기술을 활용하여 비밀번호를 대체하고 있습니다. 또한 MIT가 2022년 선정한 10대 기술 중 생체인증과 관련된(비밀번호를 대체하는 새로운 인증기술)주제가 있을만큼 얼굴인식은 중요한 기술이 되어가고 있습니다. 이러한 얼굴인식 모델을 만들기 위해 Arcface 논문을 참고하여 얼굴인식 모델을 구현해보려 합니다.

## Goal
최종적으로 Arcface 논문을 참고하여 얼굴인식 모델을 구현과 정확도 측정을 목표로합니다.

# Ⅱ. Datasets
LFW Face Database를 사용하겠습니다.

해당 데이터 셋은 1680명의 사람들의 13000장 이상의 사진들로 이루어져있습니다. 각각의 사진 속 인물의 이름이 라벨로 붙어있고 해당 사진의 확장자는 JPEG으로 되어있습니다. 또한 해당 사진들의 픽셀은 62*47 입니다.

https://www.kaggle.com/datasets/atulanandjha/lfwpeople

# Ⅲ. Methodlogy
## CNN기반의 ResNet구조
CNN : 

## 손실함수 : Arcface loss function 사용
<img width="411" alt="image" src="https://github.com/yunki00/AI-X-Deeplearing9/assets/132141925/6a9e2def-d443-4032-9a65-4c2b4e29e50b">

기존 Softmax loss functuion인 <img width="272" alt="image" src="https://github.com/yunki00/AI-X-Deeplearing9/assets/132141925/0168ce00-cb8a-4254-84fe-525acce54037">
에서 분력력을 증가시키고 학습을 안정화하기 위해 각도에 직접적 margin penalty를 더해줬다.

Softmax function하고 Arcface funtion을 그림으로 비교해 보면 다음과 같다.
<img width="303" alt="image" src="https://github.com/yunki00/AI-X-Deeplearing9/assets/132141925/a7e2cff1-d07c-4207-95b6-886cc82bc424">

# Ⅳ. Evaluation & Analysis
# Ⅴ. Related Work
ArcFace: Additive Angular Margin Loss for Deep Face Recognition

https://github.com/whitesoonguh/CnA_Arcface

https://www.kaggle.com/datasets/atulanandjha/lfwpeople
# Ⅵ. Conclusion
