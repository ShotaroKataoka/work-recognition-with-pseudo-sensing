import os
import yaml
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from data.dataset import VideoFrameDataset
from models.model import R3D_TwoStream_Model
from utils.transforms import get_transforms
from utils.mypath import Path
from utils.logger import get_logger
from dotenv import load_dotenv, find_dotenv
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse
import shutil

# .env ファイルから環境変数を読み込む
load_dotenv(find_dotenv())

def validate(model, val_loader, criterion, device, num_classes):
    model.eval()
    val_running_loss = 0.0
    val_correct_predictions = 0
    val_total_samples = 0

    with torch.no_grad():
        val_loader_tqdm = tqdm(val_loader, desc='Validation', unit='batch')  # tqdmの追加
        for clips, labels, _, sensors in val_loader_tqdm:
            clips = clips.to(device)
            labels = labels.to(device)
            sensors = sensors.view(sensors.shape[0], 1, sensors.shape[1], sensors.shape[2], sensors.shape[3]).to(device)

            outputs = model(clips, sensors)
            outputs = outputs.reshape(-1, num_classes)
            labels = labels.view(-1)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item() * labels.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct_predictions += torch.sum(preds == labels.data)
            val_total_samples += labels.size(0)

            # バッチごとの損失と精度を計算
            batch_loss = loss.item()
            batch_acc = torch.sum(preds == labels.data).double() / labels.size(0)

            # tqdmの表示を更新
            val_loader_tqdm.set_postfix({'Val Loss': batch_loss, 'Val Acc': batch_acc.item()})

    val_loss = val_running_loss / val_total_samples
    val_acc = val_correct_predictions.double() / val_total_samples

    model.train()  # モデルをトレーニングモードに戻す
    return val_loss, val_acc.item()

def main():
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='Training script for action recognition')
    parser.add_argument('--exp_name', type=str, required=True, help='Name of the experiment')
    args = parser.parse_args()
    experiment_name = args.exp_name

    # 設定ファイルの読み込み
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ログとチェックポイントのディレクトリを実験名に基づいて設定
    log_dir = os.path.join('./logs', experiment_name)
    checkpoint_dir = os.path.join('./checkpoints', experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger = get_logger(log_dir)
    writer = SummaryWriter(log_dir=log_dir)
    
    # 設定ファイルをコピーして保存
    shutil.copy('config.yaml', os.path.join(log_dir, 'config.yaml'))

    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    learning_rate = float(config['learning_rate'])
    num_classes = config['num_classes']
    clip_length = config['clip_length']
    num_workers = config['num_workers']
    log_interval = config['log_interval']
    save_interval = config['save_interval']
    validation_interval = config['validation_interval']

    # パスの取得
    dataset_root = Path.db_root_dir(config['dataset'])

    # データセットとデータローダーの作成
    train_dataset = VideoFrameDataset(
        root_dir=dataset_root,
        data_split='train',
        clip_length=clip_length,
        transform=get_transforms(config, is_train=True)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, prefetch_factor=4, shuffle=True, num_workers=num_workers, pin_memory=True)

    val_dataset = VideoFrameDataset(
        root_dir=dataset_root,
        data_split='val',
        clip_length=clip_length,
        clip_skip=4,
        transform=get_transforms(config, is_train=False)
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, prefetch_factor=4, shuffle=False, num_workers=num_workers, pin_memory=True)

    # モデルの定義
    model = R3D_TwoStream_Model(num_classes=num_classes, pretrained=True)
    model = model.to(device)

    # 損失関数とオプティマイザ
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    logger.info('Start Training')

    global_step = 0  # 追加

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        with tqdm(train_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch+1}/{num_epochs}")
            for clips, labels, _, sensors in tepoch:
                clips = clips.to(device)  # (batch_size, C, T, H, W)
                labels = labels.to(device)  # (batch_size, T)
                sensors = sensors.view(sensors.shape[0], 1, sensors.shape[1], sensors.shape[2], sensors.shape[3]).to(device)

                optimizer.zero_grad()
                outputs = model(clips, sensors)  # (batch_size, T, num_classes)

                # 出力とラベルをフラット化して損失を計算
                outputs = outputs.reshape(-1, num_classes)  # (batch_size * T, num_classes)
                labels = labels.view(-1)  # (batch_size * T)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * labels.size(0)
                _, preds = torch.max(outputs, 1)
                correct_predictions += torch.sum(preds == labels.data)
                total_samples += labels.size(0)

                global_step += 1  # ステップ数を更新

                # バッチごとの損失と精度を計算
                batch_loss = loss.item()
                batch_acc = torch.sum(preds == labels.data).double() / labels.size(0)

                # tqdm の表示を更新
                tepoch.set_postfix({'Loss': batch_loss, 'Acc': batch_acc.item()})

                # ロギング
                if global_step % log_interval == 0:
                    writer.add_scalar('Loss/train', batch_loss, global_step)
                    writer.add_scalar('Accuracy/train', batch_acc.item(), global_step)

                # 検証
                if global_step % validation_interval == 0:
                    val_loss, val_acc = validate(model, val_loader, criterion, device, num_classes)
                    writer.add_scalar('Loss/val', val_loss, global_step)
                    writer.add_scalar('Accuracy/val', val_acc, global_step)
                    logger.info(f'Step {global_step}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

                # モデルの保存
                if global_step % save_interval == 0:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'model_step_{global_step}.pth'))
                    logger.info(f'Model saved at step {global_step}')

        # エポック終了時のログ
        epoch_loss = running_loss / total_samples
        epoch_acc = correct_predictions.double() / total_samples
        logger.info(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')

    logger.info('Training Finished')
    writer.close()  # 追加

if __name__ == '__main__':
    main()
