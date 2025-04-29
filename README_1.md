# Fine-Tuning 기반 ResNet50 모델 비교 실험: 설계 및 주의사항

본 프로젝트에서는 DeepXplore를 이용한 테스트를 수행하기 위해 CIFAR-10 데이터셋에 대해 두 개의 ResNet50 모델을 fine-tuning 방식으로 구성하였다. 이에 따라 발생할 수 있는 주요 문제들과 그에 대한 대응 전략을 정리한다.

---

## Fine-Tuning 과정에서 발생할 수 있는 주요 이슈

### 1. 모델 간 학습 편향 차이 (Non-uniform Training)

문제 설명  
두 모델이 동일한 구조를 가지고 있더라도, 학습 조건(optimizer, learning rate, epoch 수, data augmentation 등)이 조금만 달라져도 decision boundary가 크게 달라질 수 있다.  
그 결과, DeepXplore가 탐지한 의심 샘플이 실제 오류가 아닌 학습 편차에 기인할 수 있다.

해결 방안  
- 동일한 CIFAR-10 데이터셋 사용  
- 동일한 ResNet50 base architecture 유지  
- fine-tuning은 일부 차이만 두기  
  - 마지막 1~2개 layer만 fine-tune  
  - dropout 비율만 다르게 적용

---

### 2. Overfitting 위험

문제 설명  
fine-tuning 과정에서 일부 모델이 특정 클래스에 과적합(overfitting)되면, 두 모델 간 결과 차이가 모델의 성능 차이보다는 특성 차이로 해석될 수 있다. 이는 DeepXplore에서 진짜 오류 탐지 여부를 혼동시킬 수 있다.

해결 방안  
- fine-tuning 후 validation accuracy 비교  
- confusion matrix를 확인하여 overfitting 여부 파악  
- 성능이 극단적으로 다른 모델은 비교에서 제외

---

### 3. 구조 변경에 따른 gradient 왜곡

문제 설명  
DeepXplore는 gradient 정보를 활용해 neuron coverage를 최대화하는 입력을 생성한다.  
fine-tuning 과정에서 구조가 변경되면 gradient 흐름 자체가 달라져 coverage 기반 테스트 결과가 왜곡될 수 있다.

해결 방안  
- 구조 변경은 최소화  
- feature extractor는 freeze  
- classifier head만 fine-tune  
  - 예: 마지막 fully connected layer 등

---

## 실험 설계 요약

| 항목 | Model A | Model B |
|------|---------|---------|
| Base Network | ResNet50 | ResNet50 |
| Fine-Tuning 범위 | 마지막 2개 Layer | 마지막 1개 Layer + Dropout 변경 |
| Optimizer | Adam | Adam |
| Epoch | 20 | 20 |
| Freeze 여부 | Feature layer 전체 freeze | 동일 |
| Dataset | CIFAR-10 | CIFAR-10 |

---

본 실험에서는 모델 간 학습 편차를 최소화하고, DeepXplore에서 생성된 input이 실제 오류를 반영하도록 설계하였다.  
더 많은 정보는 model, scripts 하위 디렉터리에 있는 주석 및 설정 파일을 참고하기 바란다.
