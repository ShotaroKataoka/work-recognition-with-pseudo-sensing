import os
import yaml
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from data.dataset import VideoFrameDataset
from models.model import get_model
from utils.transforms import get_transforms
from utils.mypath import Path
from utils.logger import get_logger

def main():
    # 設定ファイルの読み込み
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ロギングの設定
    log_dir = './logs'
    logger = get_logger(log_dir)

    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    num_classes = config['num_classes']
    clip_length = config['clip_length']
    num_workers = config['num_workers']

    # パスの取得
    dataset_root = Path.db_root_dir(config['dataset'])

    # データセットとデータローダーの作成
    train_dataset = VideoFrameDataset(
        root_dir=dataset_root,
        clip_length=clip_length,
        transform=get_transforms(config, is_train=True)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    # モデルの定義
    model = get_model(num_classes=num_classes)
    model = model.to(device)

    # 損失関数とオプティマイザ
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    logger.info('Start Training')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for clips, labels in train_loader:
            clips = clips.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(clips)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * clips.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)
            total_samples += labels.size(0)

        epoch_loss = running_loss / total_samples
        epoch_acc = correct_predictions.double() / total_samples

        logger.info(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')

        # モデルの保存
        checkpoint_dir = './checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pth'))

    logger.info('Training Finished')

if __name__ == '__main__':
    main()
