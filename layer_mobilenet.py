import torch
import numpy as np
import random
import torchvision.models as models
import torch.nn as nn
from utils.EarlyStopping import *
from utils.LRScheduler import *
from utils.train_eval_util import evaluate, train
# 저장된 모델 로드
import torchvision.models as models
import datetime

from utils.loader import get_loaders

random_seed = 2024  # 시드(seed) 고정
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"current device: {device}")

train_loader, test_loader, valid_loader = get_loaders()

print("loading complete 🔥")


# 하이퍼 파라미터 설정
class Parameters:
    def __init__(self, batch_size, learning_rate):
        self.description = "Mobilenet for Image Classification"
        # 에포크 수
        self.epochs = 50
        # 배치 크기
        self.batch_size = batch_size
        # 학습률
        self.learning_rate = learning_rate
        # 훈련된 모델 경로
        self.model_name = f"Mobilenet_layers"

def setup_model_fc(model):
    model.classifier = nn.Sequential(
        nn.Linear(in_features=model.last_channel, out_features=6, bias=True),  # 추가된 중간 레이어
    ).to(device)

for i in range(18, -1, -1):
    model = models.mobilenet_v2(pretrained=True).to(device)
    print(f"current layer: {i}")

    setup_model_fc(model)

    for param in model.parameters():
        param.requires_grad = False # 모든 파라미터를 학습

    # Ensure the final layer's parameters require gradients
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    for param in model.features[i:].parameters():
        param.requires_grad = True


    print("model setup completed")

    args = Parameters(8, 1e-4)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    early_stopping = EarlyStopping(patience=7, min_delta=1e-5)

    scheduler = LRScheduler(optimizer=optimizer, patience=5, min_lr=1e-10, factor=0.5)


    best_valid_loss = float('inf')

    # 학습률 스케줄러 정의
    train_losses = []
    valid_losses = []

    for epoch in range(args.epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        valid_loss, valid_acc, _, _ = evaluate(model, valid_loader, criterion, device)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        print(f"Epoch [{epoch+1}/{args.epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}")

        scheduler(valid_loss)

        if (early_stopping(valid_loss)):
            print("early stopped! ⚡️")
            break

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), f"./{args.model_name}.pth")

    model = models.mobilenet_v2(pretrained=True).to(device)

    setup_model_fc(model)

    model.load_state_dict(torch.load( "./" + args.model_name + ".pth" ))

    # train 세트 평가
    train_loss, train_acc, train_preds, train_labels = evaluate(model, train_loader, criterion, device)
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

    # valid 세트 평가
    valid_loss, valid_acc, valid_preds, valid_labels = evaluate(model, valid_loader, criterion, device)
    print(f"Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}")

    # test 세트 평가
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    # 현재 시간 출력
    now = datetime.datetime.now()
    now = now + datetime.timedelta(hours=9)
    now = now.strftime('%Y-%m-%d %H:%M:%S')
    print(now)