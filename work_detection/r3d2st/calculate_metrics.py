# calculate_metrics.py

import argparse
from sklearn.metrics import accuracy_score, f1_score, classification_report

def main():
    parser = argparse.ArgumentParser(description='Calculate metrics from predictions')
    parser.add_argument('--predictions_file', type=str, required=True, help='Path to the predictions.txt file')
    args = parser.parse_args()
    predictions_file = args.predictions_file

    true_labels = []
    pred_labels = []

    with open(predictions_file, 'r') as f:
        next(f)  # ヘッダー行をスキップ
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 3:
                continue
            fid, pred_label, true_label = parts
            pred_labels.append(int(pred_label))
            true_labels.append(int(true_label))

    # 評価指標を計算
    accuracy = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average='weighted')
    report = classification_report(true_labels, pred_labels)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score (weighted): {f1:.4f}")
    print("\nClassification Report:")
    print(report)

if __name__ == '__main__':
    main()
