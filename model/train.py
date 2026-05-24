# -*- coding: utf-8 -*-
"""
GinkgoSense 模型训练脚本
支持多backbone选择、CBAM注意力、混合精度训练、Cosine学习率调度
"""
import os
import sys
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import MODEL_CONFIG, TRAIN_CONFIG, PREPROCESS_CONFIG, DATA_DIR, WEIGHTS_DIR
from model.arch import GinkgoResNet, build_model
from model.evaluator import ClassificationEvaluator
from utils.augment import GinkgoAugmentation
from utils.logger import TrainingLogger


def get_args():
    parser = argparse.ArgumentParser(description='GinkgoSense 训练')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR, help='数据目录')
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet18', 'resnet34', 'resnet50', 'resnet101'])
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=TRAIN_CONFIG['epochs'])
    parser.add_argument('--batch_size', type=int, default=TRAIN_CONFIG['batch_size'])
    parser.add_argument('--lr', type=float, default=TRAIN_CONFIG['learning_rate'])
    parser.add_argument('--weight_decay', type=float, default=TRAIN_CONFIG['weight_decay'])
    parser.add_argument('--use_cbam', action='store_true', default=True)
    parser.add_argument('--no_cbam', action='store_true')
    parser.add_argument('--pretrained', action='store_true', default=True)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=WEIGHTS_DIR)
    parser.add_argument('--log_dir', type=str, default='logs/training')
    parser.add_argument('--early_stopping', type=int, default=TRAIN_CONFIG['early_stopping'])
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的权重路径')
    return parser.parse_args()


def main():
    args = get_args()
    use_cbam = not args.no_cbam

    # 设备
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"使用设备: {device}")

    # 日志
    logger = TrainingLogger(log_dir=args.log_dir)
    logger.log_config(vars(args))

    # 数据增强
    train_transform = GinkgoAugmentation(size=224, is_train=True)
    val_transform = GinkgoAugmentation(size=224, is_train=False)

    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')

    if os.path.exists(train_dir):
        train_dataset = datasets.ImageFolder(train_dir, transform=train_transform.transform)
        val_dataset = datasets.ImageFolder(val_dir, transform=val_transform.transform)
    else:
        print(f"警告: 未找到数据目录 {train_dir}，使用FakeData演示")
        train_dataset = datasets.FakeData(size=200, image_size=(3, 224, 224), num_classes=args.num_classes, transform=train_transform.transform)
        val_dataset = datasets.FakeData(size=50, image_size=(3, 224, 224), num_classes=args.num_classes, transform=val_transform.transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 模型
    config = {
        'model_type': 'resnet',
        'backbone': args.backbone,
        'num_classes': args.num_classes,
        'pretrained': args.pretrained,
        'use_cbam': use_cbam,
    }
    model = build_model(config)
    model.to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {params:,}")

    # 恢复训练
    start_epoch = 0
    best_val_acc = 0.0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt.get('model_state_dict', ckpt))
        start_epoch = ckpt.get('epoch', 0)
        best_val_acc = ckpt.get('best_val_acc', 0.0)
        print(f"恢复训练: epoch={start_epoch}, best_acc={best_val_acc:.4f}")

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 学习率调度
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 混合精度
    use_amp = device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # 训练循环
    patience_counter = 0
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"\n开始训练: {args.epochs} epochs, batch_size={args.batch_size}, lr={args.lr}")
    print("=" * 70)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # 验证
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        # 日志
        logger.log_epoch(epoch + 1, train_loss, val_loss, train_acc, val_acc, current_lr)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            save_path = os.path.join(args.save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'config': config,
            }, save_path)
            print(f"  ✓ 保存最佳模型: {save_path} (acc={val_acc:.4f})")
        else:
            patience_counter += 1

        # 保存最新模型
        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(args.save_dir, f'checkpoint_epoch{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
            }, save_path)

        # 早停
        if patience_counter >= args.early_stopping:
            print(f"\n早停: {args.early_stopping} 个epoch无提升")
            break

    # 保存训练历史
    logger.save_history()
    print(f"\n训练完成! 最佳验证准确率: {best_val_acc:.4f}")
    print(f"模型权重保存在: {args.save_dir}")


if __name__ == '__main__':
    main()
