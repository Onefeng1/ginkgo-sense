# -*- coding: utf-8 -*-
"""
模型训练脚本
白果图像分类模型训练与评估
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
import json
import time

from config import MODEL_CONFIG, TRAIN_CONFIG, PREPROCESS_CONFIG


def get_data_loaders(data_dir, batch_size):
    """构建数据加载器"""
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(PREPROCESS_CONFIG['mean'], PREPROCESS_CONFIG['std']),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(PREPROCESS_CONFIG['mean'], PREPROCESS_CONFIG['std']),
    ])

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"训练集: {len(train_dataset)} 张图片, {len(train_loader)} 批次")
    print(f"验证集: {len(val_dataset)} 张图片, {len(val_loader)} 批次")
    print(f"类别: {train_dataset.classes}")

    return train_loader, val_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })

    return total_loss / len(loader), 100. * correct / total


def validate(model, loader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return total_loss / len(loader), 100. * correct / total


def main():
    parser = argparse.ArgumentParser(description='白果图像分类模型训练')
    parser.add_argument('--data_dir', type=str, default='data', help='数据目录')
    parser.add_argument('--epochs', type=int, default=TRAIN_CONFIG['epochs'])
    parser.add_argument('--batch_size', type=int, default=TRAIN_CONFIG['batch_size'])
    parser.add_argument('--lr', type=float, default=TRAIN_CONFIG['learning_rate'])
    parser.add_argument('--save_dir', type=str, default='weights', help='模型保存目录')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 数据
    train_loader, val_loader = get_data_loaders(args.data_dir, args.batch_size)

    # 模型
    model = models.resnet50(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, MODEL_CONFIG['num_classes']),
    )
    model.to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=TRAIN_CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 训练循环
    best_acc = 0
    history = []

    print(f"\n开始训练，共 {args.epochs} 个epoch")
    print("=" * 60)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 40)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
        })

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))
            print(f"  -> 保存最佳模型 (Acc: {val_acc:.2f}%)")

    # 保存训练历史
    with open(os.path.join(args.save_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print(f"训练完成! 最佳验证准确率: {best_acc:.2f}%")
    print(f"模型已保存至: {args.save_dir}/best_model.pth")


if __name__ == '__main__':
    main()
