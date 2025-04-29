import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys
import requests
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as torch_models
from collections import OrderedDict
import copy
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

# load_models 함수: 마지막 1-2개 layer만 fine-tuning 적용
def load_models():
    """
    마지막 1-2개 layer만 fine-tuning하는 ResNet-50 모델 두 개를 로드합니다.
    개선된 버전: CIFAR-10에 더 적합한 구조로 변경
    
    Returns:
        model_list: 로드된 모델 리스트
        model_names: 모델 이름 리스트
    """
    # device 설정 추가
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_list = []
    model_names = []
    
    try:
        # 기본 ResNet 모델 클래스 정의 (CIFAR-10에 맞게 조정)
        class CIFAR10_ResNet50(torch.nn.Module):
            def __init__(self, num_classes=10):
                super(CIFAR10_ResNet50, self).__init__()
                # 기본 ResNet-50 모델 로드 (가중치는 나중에 로드)
                resnet = torch_models.resnet50(pretrained=True)
                
                # ResNet 첫 레이어를 CIFAR-10에 맞게 조정
                # 7x7 커널을 3x3으로 변경하고 stride도 줄임
                resnet.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                resnet.bn1 = torch.nn.BatchNorm2d(64)
                
                # 첫 번째 MaxPool 제거 (작은 이미지에 맞게)
                resnet.maxpool = torch.nn.Identity()
                
                # Dropout 추가로 일반화 성능 향상
                self.dropout = torch.nn.Dropout(0.3)
                
                # 출력 레이어를 CIFAR-10 클래스 수에 맞게 조정
                # 중간 FC 레이어 추가로 표현력 향상
                in_features = resnet.fc.in_features
                resnet.fc = torch.nn.Sequential(
                    torch.nn.Linear(in_features, 512),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.5),
                    torch.nn.Linear(512, num_classes)
                )
                
                self.resnet = resnet
                
            def forward(self, x):
                x = self.resnet.conv1(x)
                x = self.resnet.bn1(x)
                x = F.relu(x)
                # Maxpool 생략 (self.resnet.maxpool 대신 Identity)
                
                x = self.resnet.layer1(x)
                x = self.resnet.layer2(x)
                x = self.resnet.layer3(x)
                x = self.resnet.layer4(x)
                
                x = F.adaptive_avg_pool2d(x, (1, 1))
                x = torch.flatten(x, 1)
                x = self.dropout(x)  # 추가된 Dropout
                x = self.resnet.fc(x)
                
                return x
            
            # ResNet 서브모듈에 쉽게 접근하기 위한 프로퍼티
            @property
            def layer1(self):
                return self.resnet.layer1
                
            @property
            def layer2(self):
                return self.resnet.layer2
                
            @property
            def layer3(self):
                return self.resnet.layer3
                
            @property
            def layer4(self):
                return self.resnet.layer4
                
            @property
            def fc(self):
                return self.resnet.fc
                
            @fc.setter
            def fc(self, new_fc):
                self.resnet.fc = new_fc
        
        # --------------------------------------------
        # 모델 1: Partial Fine-tuning + Class imbalance
        # --------------------------------------------
        model1 = CIFAR10_ResNet50(num_classes=10)
        
        # 마지막 1-2개 layer만 학습하도록 설정 - 모든 파라미터 우선 freeze
        for param in model1.parameters():
            param.requires_grad = False
            
        # 마지막 layer(layer4)와 FC layer만 학습하도록 설정 (개선: layer3도 추가)
        for param in model1.layer3.parameters():
            param.requires_grad = True
        for param in model1.layer4.parameters():
            param.requires_grad = True
        for param in model1.fc.parameters():
            param.requires_grad = True
            
        # Class imbalance 조절 설정 (weighted loss를 위한 가중치 저장)
        class_weights_model1 = torch.ones(10)
        # 특정 클래스(0, 3, 7번)에 가중치 부여
        class_weights_model1[0] = 2.0  # airplane
        class_weights_model1[3] = 1.5  # cat
        class_weights_model1[7] = 1.8  # horse
        
        # 가중치 loss를 사용하기 위한 설정 저장
        model1.class_weights = class_weights_model1
        
        # --------------------------------------------
        # 모델 2: Noisy data + Domain-specific augmentation
        # --------------------------------------------
        model2 = CIFAR10_ResNet50(num_classes=10)
        
        # 마찬가지로 마지막 1-2개 layer만 학습하도록 설정
        for param in model2.parameters():
            param.requires_grad = False
            
        # 마지막 layer(layer4)와 FC layer만 학습하도록 설정 (개선: layer3도 추가)
        for param in model2.layer3.parameters():
            param.requires_grad = True
        for param in model2.layer4.parameters():
            param.requires_grad = True
        for param in model2.fc.parameters():
            param.requires_grad = True
        
        # Noisy data fine-tune를 위한 설정
        model2.noise_level = 0.05  # 입력에 추가할 노이즈 레벨
        
        # Domain-specific augmentation 설정 개선
        model2.domain_transforms = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(32, scale=(0.8, 1.0))  # 추가된 변환
        ])
        
        # 두 모델 모두 device로 이동
        model1 = model1.to(device)
        model2 = model2.to(device)
        
        # 모델을 평가 모드로 설정
        model1.eval()
        model2.eval()
        
        # 모델과 이름 추가
        model_list = [model1, model2]
        model_names = ["ResNet50_Partial_ClassImbalance", "ResNet50_Noisy_Augmented"]
        
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        return [], []
    
    return model_list, model_names

# 모델 평가 함수: accuracy와 class별 성능 분석 추가
def evaluate_model_detailed(model, dataloader, class_names=None):
    """
    모델의 정확도를 평가하고 클래스별 성능을 분석합니다.
    
    Args:
        model: 평가할 모델
        dataloader: 테스트 데이터 로더
        class_names: 클래스 이름 리스트 (confusion matrix 시각화용)
    
    Returns:
        accuracy: 전체 정확도
        class_accuracies: 클래스별 정확도
        conf_matrix: 혼동 행렬
    """
    # device 설정 추가
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model.eval()  # 평가 모드로 설정
    
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    
    # 클래스별 정확도 계산을 위한 변수
    class_correct = [0] * 10
    class_total = [0] * 10
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            # 전체 정확도 계산용
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 클래스별 정확도 계산용
            for i in range(len(labels)):
                label = labels[i].item()
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1
            
            # 혼동 행렬 계산용
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 전체 정확도 계산
    accuracy = 100 * correct / total
    
    # 클래스별 정확도 계산
    class_accuracies = [100 * class_correct[i] / max(1, class_total[i]) for i in range(10)]
    
    # 혼동 행렬 계산
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    print(f'전체 정확도: {accuracy:.2f}%')
    print('클래스별 정확도:')
    for i in range(10):
        class_name = class_names[i] if class_names else f"Class {i}"
        print(f'  {class_name}: {class_accuracies[i]:.2f}%')
    
    return accuracy, class_accuracies, conf_matrix

# 혼동 행렬 시각화 함수
def plot_confusion_matrix(conf_matrix, class_names=None, title='Confusion Matrix'):
    """
    혼동 행렬을 시각화합니다.
    
    Args:
        conf_matrix: 혼동 행렬
        class_names: 클래스 이름 리스트
        title: 그래프 제목
    """
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(conf_matrix))]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title)
    plt.tight_layout()
    plt.show()

# 모델 비교 함수
def compare_models(model_list, model_names, validation_loader, test_loader, class_names=None):
    """
    여러 모델의 성능을 비교하고 시각화합니다.
    
    Args:
        model_list: 모델 리스트
        model_names: 모델 이름 리스트
        validation_loader: 검증 데이터셋 로더
        test_loader: 테스트 데이터셋 로더
        class_names: 클래스 이름 리스트
    """
    if class_names is None:
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                      'dog', 'frog', 'horse', 'ship', 'truck']
    
    # 결과 저장용 변수
    validation_accuracies = []
    test_accuracies = []
    class_accuracies_list = []
    conf_matrices = []
    
    # 각 모델별 평가
    for i, (model, name) in enumerate(zip(model_list, model_names)):
        print(f"\n===== 모델 {i+1}: {name} 평가 =====")
        
        # 검증 데이터셋으로 평가
        print("검증 데이터셋 평가:")
        val_acc, val_class_acc, _ = evaluate_model_detailed(model, validation_loader, class_names)
        validation_accuracies.append(val_acc)
        
        # 테스트 데이터셋으로 평가
        print("\n테스트 데이터셋 평가:")
        test_acc, test_class_acc, conf_mat = evaluate_model_detailed(model, test_loader, class_names)
        test_accuracies.append(test_acc)
        class_accuracies_list.append(test_class_acc)
        conf_matrices.append(conf_mat)
        
        # 혼동 행렬 시각화
        plot_confusion_matrix(conf_mat, class_names, f'{name} Confusion Matrix')
    
    # 모델 간 정확도 비교 시각화
    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    index = np.arange(len(model_list))
    
    plt.bar(index, validation_accuracies, bar_width, label='Validation Accuracy')
    plt.bar(index + bar_width, test_accuracies, bar_width, label='Test Accuracy')
    
    plt.xlabel('모델')
    plt.ylabel('정확도 (%)')
    plt.title('모델 간 정확도 비교')
    plt.xticks(index + bar_width/2, model_names, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # 클래스별 성능 비교 시각화
    plt.figure(figsize=(12, 8))
    x = np.arange(len(class_names))
    width = 0.8 / len(model_list)
    
    for i, (model_name, class_accs) in enumerate(zip(model_names, class_accuracies_list)):
        plt.bar(x + i * width, class_accs, width, label=model_name)
    
    plt.xlabel('클래스')
    plt.ylabel('정확도 (%)')
    plt.title('클래스별 정확도 비교')
    plt.xticks(x + width * (len(model_list) - 1) / 2, class_names, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # 모델 간 성능 차이 분석
    print("\n===== 모델 간 성능 차이 분석 =====")
    for i in range(len(model_list)):
        for j in range(i+1, len(model_list)):
            print(f"\n{model_names[i]} vs {model_names[j]}")
            print(f"전체 정확도 차이: {test_accuracies[i] - test_accuracies[j]:.2f}%")
            
            # 클래스별 정확도 차이가 큰 순서대로 정렬
            diff = [class_accuracies_list[i][k] - class_accuracies_list[j][k] for k in range(len(class_names))]
            sorted_indices = np.argsort(np.abs(diff))[::-1]
            
            print("주요 클래스별 정확도 차이:")
            for idx in sorted_indices[:3]:  # 상위 3개 클래스만 표시
                print(f"  {class_names[idx]}: {diff[idx]:.2f}% ({model_names[i]} {'+' if diff[idx]>0 else ''}{diff[idx]:.2f}%)")

def verify_models_for_deepxplore(model_list, model_names):
    """
    DeepXplore 테스트 전에 모델들의 구조를 검증합니다.
    
    Args:
        model_list: 모델 리스트
        model_names: 모델 이름 리스트
    """
    print("\n===== DeepXplore 테스트 전 모델 구조 검증 =====")
    
    # 학습 가능한 레이어 정보를 저장할 리스트 초기화
    trainable_layers_list = []
    
    for i, (model, name) in enumerate(zip(model_list, model_names)):
        print(f"\n모델 {i+1}: {name}")
        
        # 학습 가능한 파라미터 확인
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params_count = sum(p.numel() for p in trainable_params)
        
        print(f"총 파라미터: {total_params:,}")
        print(f"학습 가능한 파라미터: {trainable_params_count:,} ({trainable_params_count/total_params*100:.2f}%)")
        
        # 학습 가능한 레이어 확인 - 더 상세한 정보 제공
        print("학습 가능한 레이어:")
        trainable_layers = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"  - {name}")
                trainable_layers.append(name.split('.')[0] if '.' in name else name)
        
        # 유사한 구조 확인을 위한 레이어 비교
        if i > 0:
            prev_trainable_layers = set(trainable_layers_list[i-1])
            curr_trainable_layers = set(trainable_layers)
            diff_layers = prev_trainable_layers.symmetric_difference(curr_trainable_layers)
            if diff_layers:
                print(f"⚠️ 주의: 모델 {i}와 모델 {i+1}의 학습 가능한 레이어 구조가 다릅니다.")
                print(f"  차이가 있는 레이어: {diff_layers}")
                print("  이로 인해 DeepXplore 테스트 결과에 영향이 있을 수 있습니다.")
            else:
                print(f"✓ 모델 {i}와 동일한 학습 가능 레이어 구조를 가지고 있습니다.")
        
        trainable_layers_list.append(trainable_layers)
    
    print("\nDeepXplore 호환성 참고사항:")
    print("- 모든 모델은 동일한 네트워크 구조를 가지고 있지만 일부 파라미터만 fine-tune되었습니다.")
    print("- Gradient 흐름은 고정된 레이어에서는 변경되지 않지만, fine-tune된 레이어에서는 달라질 수 있습니다.")
    print("- 뉴런 커버리지 측정 시 미세한 차이가 있을 수 있으나, 구조적 호환성은 유지됩니다.")
    
    return trainable_layers_list  # 학습 가능한 레이어 정보 반환


# CIFAR-10 데이터셋 로드 및 설정 예시
def prepare_cifar10_dataloaders():
    """
    CIFAR-10 데이터셋을 로드하고 더 강력한 데이터 증강을 적용한 데이터 로더를 준비합니다.
    
    Returns:
        train_loader: 학습 데이터 로더
        validation_loader: 검증 데이터 로더
        test_loader: 테스트 데이터 로더
        class_names: 클래스 이름 리스트
    """
    # 데이터 변환 설정 개선
    # 테스트용 변환
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    # 학습용 변환 - 더 강력한 증강 적용
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        transforms.RandomErasing(p=0.2)  # 랜덤 지우기 추가
    ])
    
    # 전체 학습 데이터셋 로드
    train_full = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    
    # 학습 및 검증 데이터셋 분할 (80:20)
    train_size = int(0.8 * len(train_full))
    validation_size = len(train_full) - train_size
    train_dataset, validation_dataset = torch.utils.data.random_split(train_full, [train_size, validation_size])
    
    # 검증용 데이터셋은 test_transform 적용 (증강 없이)
    validation_dataset.dataset.transform = test_transform
    
    # 테스트 데이터셋 로드
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    
    # 데이터 로더 생성 - 배치 크기 조정
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    return train_loader, validation_loader, test_loader, class_names

# 새로 추가: 모델 학습 함수
def train_model(model, train_loader, val_loader, device, epochs=5, lr=0.001):
    """
    모델을 학습시키는 함수
    
    Args:
        model: 학습할 모델
        train_loader: 학습 데이터 로더
        val_loader: 검증 데이터 로더
        device: 학습 장치 (CPU/GPU)
        epochs: 에폭 수
        lr: 학습률
    
    Returns:
        학습된 모델
    """
    # 모델을 학습 모드로 설정
    model.train()
    
    # 손실 함수 설정 - 클래스 불균형이 있는 경우 가중치 적용
    if hasattr(model, 'class_weights'):
        weights = model.class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        print("클래스 가중치 적용된 손실 함수 사용")
    else:
        criterion = nn.CrossEntropyLoss()
    
    # 최적화기 설정 - 가중치 감쇠 및 모멘텀 적용
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], 
                                lr=lr, momentum=0.9, weight_decay=5e-4)
    
    # 학습률 스케줄러 - 성능 개선에 중요
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_acc = 0.0
    for epoch in range(epochs):
        running_loss = 0.0
        model.train()  # 학습 모드 설정
        
        # 학습 루프
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Noisy data 설정이 있는 모델인 경우
            if hasattr(model, 'noise_level'):
                noise = torch.randn_like(inputs) * model.noise_level
                inputs = inputs + noise
                inputs = torch.clamp(inputs, 0, 1)  # 값 범위 제한
            
            # Domain-specific augmentation 설정이 있는 모델인 경우
            if hasattr(model, 'domain_transforms'):
                # 배치 단위로 적용하기 어려워 순회하며 적용
                for i in range(inputs.shape[0]):
                    # CPU로 이동하여 변환 후 다시 디바이스로
                    img = inputs[i].cpu()
                    img = transforms.ToPILImage()(img)
                    img = model.domain_transforms(img)
                    inputs[i] = transforms.ToTensor()(img).to(device)
            
            # 그래디언트 초기화
            optimizer.zero_grad()
            
            # 순전파
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 역전파 및 최적화
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # 에폭당 학습률 갱신
        scheduler.step()
        
        # 검증 루프
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Val Acc: {val_acc:.2f}%')
        
        # 최고 성능 모델 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # 실제 프로젝트에서는 여기서 모델 저장 코드 추가
    
    print(f'최종 검증 정확도: {best_val_acc:.2f}%')
    return model

# DeepXplore 클래스 구현
class DeepXplore:
    def __init__(self, model_list, model_names, testloader, classes, device=None):
        """
        DeepXplore 클래스 초기화
        
        Args:
            model_list: 테스트할 모델 리스트
            model_names: 모델 이름 리스트
            testloader: 테스트 데이터 로더
            classes: 클래스 이름 리스트
            device: 연산 장치 (CPU/GPU)
        """
        # device 설정 추가
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"DeepXplore using device: {self.device}")
        
        self.models = model_list
        self.model_names = model_names
        self.testloader = testloader
        self.classes = classes
        self.neuron_thresholds = {}  # 뉴런 활성화 임계값 저장
        
        # 각 모델별로 뉴런 활성화 임계값 초기화
        for i, model in enumerate(model_list):
            self.neuron_thresholds[i] = {}
            
    def measure_neuron_coverage(self, model_idx, inputs, threshold=0.5):
        """
        특정 모델의 뉴런 커버리지를 측정합니다.
        
        Args:
            model_idx: 모델 인덱스
            inputs: 입력 데이터
            threshold: 활성화로 간주할 임계값
            
        Returns:
            활성화된 뉴런 비율
        """
        model = self.models[model_idx]
        
        # 뉴런 활성화를 저장할 딕셔너리 초기화
        activations = {}
        
        # 학습 가능한 레이어 이름 추출
        fine_tuned_layers = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                layer_name = name.split('.')[0] if '.' in name else name
                fine_tuned_layers.append(layer_name)
        
        # 중복 제거
        fine_tuned_layers = list(set(fine_tuned_layers))
        
        # 활성화 값을 수집하기 위한 후크 함수
        def hook_fn(module, input, output):
            # ReLU 이후의 활성화 고려
            if isinstance(module, torch.nn.ReLU):
                # 특정 기준(threshold) 이상의 활성화만 고려
                activations[module] = (output > threshold).float().mean().item()
        
        # 모든 ReLU 레이어에 후크 등록
        hooks = []
        module_info = {}  # 모듈별 fine-tuned 여부 저장
        
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.ReLU):
                # 레이어 이름을 기록하여 fine-tuned vs frozen 레이어 구분
                layer_name = name.split('.')[0] if '.' in name else name
                is_fine_tuned = any(layer_name.startswith(ft_layer) for ft_layer in fine_tuned_layers)
                
                # 후크 등록
                hook = module.register_forward_hook(hook_fn)
                hooks.append(hook)
                module_info[module] = is_fine_tuned
        
        # 모델 추론 실행
        with torch.no_grad():
            model(inputs)
        
        # 후크 제거 및 커버리지 계산
        fine_tuned_activations = {}
        frozen_activations = {}
        
        for module in activations:
            if module in module_info:
                if module_info[module]:
                    fine_tuned_activations[module] = activations[module]
                else:
                    frozen_activations[module] = activations[module]
        
        # 모든 후크 제거
        for hook in hooks:
            hook.remove()
        
        # 활성화된 뉴런 비율 계산 - fine-tuned vs frozen 구분
        coverage_overall = sum(activations.values()) / len(activations) if activations else 0.0
        coverage_fine_tuned = sum(fine_tuned_activations.values()) / len(fine_tuned_activations) if fine_tuned_activations else 0.0
        coverage_frozen = sum(frozen_activations.values()) / len(frozen_activations) if frozen_activations else 0.0
        
        # 반환 값 향상: 전체, fine-tuned, frozen 레이어별 커버리지
        return {
            'overall': coverage_overall,
            'fine_tuned': coverage_fine_tuned,
            'frozen': coverage_frozen
        }
    
    def find_diff_inputs(self, num_samples=100):
        """
        테스트셋에서 모델 간 예측이 다른 입력을 찾습니다.
        
        Args:
            num_samples: 검사할 샘플 수
            
        Returns:
            의심스러운 입력 목록
        """
        print(f"테스트셋에서 차이점이 있는 입력 검색 중...")
        suspicious_inputs = []
        count = 0
        
        with torch.no_grad():
            for inputs, labels in self.testloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # 각 이미지에 대해 모델 간 예측 비교
                for i in range(inputs.size(0)):
                    input_img = inputs[i:i+1]
                    true_label = labels[i].item()
                    
                    # 각 모델의 예측 결과
                    predictions = []
                    probs = []
                    for model in self.models:
                        output = model(input_img)
                        probabilities = F.softmax(output, dim=1)
                        pred = output.argmax(dim=1).item()
                        prob = probabilities[0, pred].item()
                        predictions.append(pred)
                        probs.append(prob)
                    
                    # 모델 간 예측이 같은지 확인
                    if len(set(predictions)) > 1:
                        suspicious_inputs.append({
                            'input': input_img.cpu().squeeze(0),
                            'true_label': true_label,
                            'predictions': predictions,
                            'probabilities': probs,
                            'class_name': self.classes[true_label],
                            'pred_class_names': [self.classes[p] for p in predictions]
                        })
                
                count += inputs.size(0)
                if count >= num_samples or len(suspicious_inputs) >= 20:
                    break
        
        print(f"{count}개 샘플 중 {len(suspicious_inputs)}개의 의심스러운 입력 발견")
        return suspicious_inputs
    
    def generate_adversarial_inputs(self, num_samples=10, max_iter=20, step_size=0.01):
        """
        적대적 입력을 생성하여 모델 간 차이를 극대화합니다.
        
        Args:
            num_samples: 생성할 샘플 수
            max_iter: 최대 반복 횟수
            step_size: 업데이트 단계 크기
            
        Returns:
            생성된 적대적 입력 목록
        """
        print(f"적대적 입력 생성 중...")
        adversarial_inputs = []
        sample_count = 0
        
        # 테스트셋에서 샘플 얻기
        for inputs, labels in self.testloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            for i in range(inputs.size(0)):
                if len(adversarial_inputs) >= num_samples:
                    break
                    
                # 원본 이미지 복사
                input_img = inputs[i:i+1].clone().detach().requires_grad_(True)
                true_label = labels[i].item()

                # 그래디언트 기반 탐색 수행
                for iteration in range(max_iter):
                    # 모델의 예측 결과
                    outputs = []
                    for model in self.models:
                        output = model(input_img)
                        outputs.append(output)
                    
                    # 차분 손실: 모델 간 출력 차이 최대화
                    diff_loss = 0
                    for i in range(len(outputs)):
                        for j in range(i+1, len(outputs)):
                            # 출력 차이의 절대값 최대화
                            diff_loss -= F.mse_loss(F.softmax(outputs[i], dim=1), 
                                                  F.softmax(outputs[j], dim=1))
                    
                    # 그래디언트 계산 및 업데이트
                    diff_loss.backward()
                    
                    # 입력 업데이트
                    with torch.no_grad():
                        grad = input_img.grad.sign()  # FGSM 방식 사용
                        input_img.data = input_img.data - step_size * grad
                        
                        # 유효한 이미지 범위로 클리핑
                        input_img.data = torch.clamp(input_img.data, -2.0, 2.0)  # 정규화된 범위 내로 제한
                        
                        # 그래디언트 초기화
                        input_img.grad.zero_()
                
                # 변형된 입력으로 예측
                predictions = []
                probs = []
                with torch.no_grad():
                    for model in self.models:
                        output = model(input_img)
                        probabilities = F.softmax(output, dim=1)
                        pred = output.argmax(dim=1).item()
                        prob = probabilities[0, pred].item()
                        predictions.append(pred)
                        probs.append(prob)
                
                # 모델 간 예측이 다른 경우만 저장
                if len(set(predictions)) > 1:
                    adversarial_inputs.append({
                        'input': input_img.detach().cpu().squeeze(0),
                        'true_label': true_label,
                        'predictions': predictions,
                        'probabilities': probs,
                        'class_name': self.classes[true_label],
                        'pred_class_names': [self.classes[p] for p in predictions],
                        'iterations': max_iter
                    })
                
                sample_count += 1
                
                # 충분한 샘플을 찾았거나 너무 많은 시도를 했다면 중단
                if len(adversarial_inputs) >= num_samples or sample_count >= num_samples * 5:
                    break
            
            if len(adversarial_inputs) >= num_samples or sample_count >= num_samples * 5:
                break
        
        print(f"{sample_count}개 시도 중 {len(adversarial_inputs)}개의 적대적 입력 생성 성공")
        return adversarial_inputs
    
    def visualize_results(self, suspicious_inputs, title="의심스러운 입력 시각화", save_path=None):
        """
        의심스러운 입력들을 시각화합니다.
        
        Args:
            suspicious_inputs: 의심스러운 입력 목록
            title: 그림 제목
            save_path: 저장 경로
        """
        num_to_show = min(10, len(suspicious_inputs))
        if num_to_show == 0:
            print("시각화할 입력이 없습니다.")
            return
        
        # 이미지 역정규화 함수
        def denormalize(tensor):
            mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
            std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
            return tensor * std + mean
        
        # subplot 크기 계산
        rows = (num_to_show + 1) // 2 if num_to_show > 1 else 1
        cols = min(2, num_to_show)
        
        plt.figure(figsize=(12, 5 * rows))
        plt.suptitle(title, fontsize=16)
        
        for i in range(num_to_show):
            sample = suspicious_inputs[i]
            img = denormalize(sample['input'])
            
            plt.subplot(rows, cols, i + 1)
            plt.imshow(img.permute(1, 2, 0).numpy())
            
            # 제목 구성
            subtitle = f"실제: {sample['class_name']}\n"
            for j, (pred, model_name, prob) in enumerate(zip(
                sample['pred_class_names'], 
                self.model_names, 
                sample.get('probabilities', [1.0] * len(self.model_names))
            )):
                subtitle += f"{model_name}: {pred} ({prob:.2f})\n"
            
            plt.title(subtitle)
            plt.axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # 상단 제목을 위한 공간 확보
        
        if save_path:
            plt.savefig(save_path)
            print(f"시각화 결과가 '{save_path}'로 저장되었습니다.")
        
        plt.show()
    
    def analyze_results(self, suspicious_inputs):
        """
        의심스러운 입력들을 분석합니다.
        
        Args:
            suspicious_inputs: 의심스러운 입력 목록
            
        Returns:
            분석 결과 딕셔너리
        """
        analysis = {
            'total_count': len(suspicious_inputs),
            'class_distribution': {},
            'disagreement_patterns': {},
            'error_rate': 0
        }
        
        if not suspicious_inputs:
            print("분석할 입력이 없습니다.")
            return analysis
        
        # 클래스별 분포 분석
        for sample in suspicious_inputs:
            true_class = sample['class_name']
            if true_class not in analysis['class_distribution']:
                analysis['class_distribution'][true_class] = 0
            analysis['class_distribution'][true_class] += 1
        
        # 불일치 패턴 분석
        for sample in suspicious_inputs:
            pattern = tuple(sample['pred_class_names'])
            if pattern not in analysis['disagreement_patterns']:
                analysis['disagreement_patterns'][pattern] = 0
            analysis['disagreement_patterns'][pattern] += 1
        
        # 오류율 분석: 적어도 하나의 모델이 오답을 예측한 비율
        error_count = 0
        for sample in suspicious_inputs:
            true_label = sample['true_label']
            predictions = sample['predictions']
            if any(pred != true_label for pred in predictions):
                error_count += 1
        
        analysis['error_rate'] = error_count / len(suspicious_inputs) if suspicious_inputs else 0
        
        return analysis

# 메인 실행 함수
def run_deepxplore():
    """DeepXplore를 실행하고 결과를 분석합니다."""
    # device 설정
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 데이터 로더 생성
    train_loader, validation_loader, test_loader, class_names = prepare_cifar10_dataloaders()
    
    # 모델 로드
    print("\n=== 모델 로드 중 ===")
    model_list, model_names = load_models()
    if not model_list:
        print("모델 로드에 실패했습니다. 종료합니다.")
        return
    
    # 모델 훈련 추가 - 핵심 개선점
    print("\n=== 모델 훈련 시작 ===")
    for i, (model, name) in enumerate(zip(model_list, model_names)):
        print(f"\n=== 모델 {i+1}: {name} 훈련 ===")
        model_list[i] = train_model(model, train_loader, validation_loader, device, epochs=10, lr=0.01)
    
    # sklearn confusion_matrix 임포트 확인
    try:
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
    except ImportError:
        print("sklearn 또는 seaborn 패키지가 설치되어 있지 않습니다.")
        print("pip install scikit-learn seaborn으로 설치하세요.")
        return
    
    # 모델 구조 검증 (DeepXplore 테스트 전)
    trainable_layers_info = verify_models_for_deepxplore(model_list, model_names)
    
    # 모델 비교 및 혼동 행렬 분석
    print("\n=== 모델 검증 및 혼동 행렬 분석 ===")
    compare_models(model_list, model_names, validation_loader, test_loader, class_names)
    
    # DeepXplore 인스턴스 생성
    print("\n=== DeepXplore 초기화 ===")
    deep_xplore = DeepXplore(model_list, model_names, test_loader, class_names, device)
    
    # 1. 테스트셋에서 모델 간 예측이 다른 입력 찾기
    print("\n=== 차분 테스트: 자연적 불일치 찾기 ===")
    natural_diff_inputs = deep_xplore.find_diff_inputs(num_samples=500)
    
    # 2. 적대적 입력 생성
    print("\n=== 적대적 입력 생성 ===")
    adversarial_inputs = deep_xplore.generate_adversarial_inputs(num_samples=10, max_iter=30)
    
    # 모든 의심스러운 입력 합치기
    all_suspicious_inputs = natural_diff_inputs + adversarial_inputs
    
    # 결과 시각화 및 분석
    if natural_diff_inputs:
        print("\n=== 자연적 불일치 시각화 ===")
        deep_xplore.visualize_results(natural_diff_inputs, 
                                      title="모델 간 자연적 예측 불일치",
                                      save_path="natural_disagreements.png")
        
        # 자연적 불일치의 클래스별 분포 분석 - 과적합 패턴 발견
        class_distribution = {}
        for sample in natural_diff_inputs:
            true_class = sample['class_name']
            if true_class not in class_distribution:
                class_distribution[true_class] = {'count': 0, 'models': {m: 0 for m in model_names}}
            
            class_distribution[true_class]['count'] += 1
            
            # 각 모델별로 예측 결과 기록 (올바른 예측 vs 잘못된 예측)
            for i, (pred, model_name) in enumerate(zip(sample['predictions'], model_names)):
                if pred == sample['true_label']:
                    class_distribution[true_class]['models'][model_name] += 1
        
        # 클래스별 과적합/과소적합 패턴 분석
        print("\n=== 클래스별 과적합/과소적합 패턴 분석 ===")
        for class_name, data in sorted(class_distribution.items(), key=lambda x: x[1]['count'], reverse=True):
            print(f"\n클래스: {class_name} (불일치 샘플 수: {data['count']}개)")
            print("  모델별 정확도:")
            for model_name, correct_count in data['models'].items():
                accuracy = correct_count / data['count'] * 100 if data['count'] > 0 else 0
                print(f"  - {model_name}: {accuracy:.1f}% 정확도 ({correct_count}/{data['count']})")
            
            # 과적합/과소적합 패턴 감지
            accuracies = [data['models'][m] / data['count'] * 100 for m in model_names]
            if max(accuracies) - min(accuracies) > 30:  # 모델 간 정확도 차이가 30% 이상이면 과적합/과소적합 의심
                best_model = model_names[accuracies.index(max(accuracies))]
                worst_model = model_names[accuracies.index(min(accuracies))]
                print(f"  ⚠️ 과적합/과소적합 의심: {best_model}은 {max(accuracies):.1f}%, {worst_model}은 {min(accuracies):.1f}%")
    
    # 결과 저장
    results = {
        'model_names': model_names,
        'natural_disagreements': natural_diff_inputs,
        'adversarial_inputs': adversarial_inputs,
    }
    
    try:
        torch.save(results, 'deepxplore_results.pth')
        print("\n분석 결과가 'deepxplore_results.pth'로 저장되었습니다.")
    except Exception as e:
        print(f"\n결과 저장 중 오류 발생: {e}")
    
    return results

# 메인 실행 코드
if __name__ == "__main__":
    print("DeepXplore 테스트 시작...")
    results = run_deepxplore()
    print("DeepXplore 테스트 완료!")