# predict.py

import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from data.dataset import VideoFrameDataset
from models.model import R3D_Model
from utils.transforms import get_transforms
from utils.mypath import Path
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description='Inference script for action recognition')
    parser.add_argument('--experiment_name', type=str, required=True, help='Name of the experiment')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--output_file', type=str, default='predictions.txt', help='File to save predictions')
    args = parser.parse_args()
    experiment_name = args.experiment_name
    checkpoint_path = args.checkpoint
    output_file = args.output_file

    # 実験の設定ファイルを読み込む
    config_path = os.path.join('./logs', experiment_name, 'config.yaml')
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_classes = config['num_classes']
    clip_length = config['clip_length']
    num_workers = config['num_workers']

    # データセットのルートディレクトリを取得
    dataset_root = Path.db_root_dir(config['dataset'])

    # テストデータセットとデータローダーの作成
    test_dataset = VideoFrameDataset(
        root_dir=dataset_root,
        data_split='test',  # 'test' データを使用
        clip_length=clip_length,
        clip_skip=7,
        transform=get_transforms(config, is_train=False)
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)

    # モデルの定義
    model = R3D_Model(num_classes=num_classes, pretrained=False)
    model = model.to(device)

    # チェックポイントの読み込み
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)

    model.eval()
    predictions = []

    with torch.no_grad():
        for clips, labels, frame_ids in tqdm(test_loader):
            clips = clips.to(device)
            labels = labels.view(-1).numpy()  # 正解ラベルを取得
            outputs = model(clips)
            outputs = outputs.reshape(-1, num_classes)
            _, preds = torch.max(outputs, 1)
            preds = preds.cpu().numpy()
            frame_ids = frame_ids  # フレーム識別子のリスト

            for fid, pred_label, true_label in zip(frame_ids, preds.tolist(), labels.tolist()):
                predictions.append((fid, pred_label, true_label))

    # 予測結果をファイルに保存
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write('FrameID\tPredictedLabel\tTrueLabel\n')
        for fid, pred_label, true_label in predictions:
            f.write(f"{fid}\t{pred_label}\t{true_label}\n")

    print(f"Predictions saved to {output_file}")

if __name__ == '__main__':
    main()
