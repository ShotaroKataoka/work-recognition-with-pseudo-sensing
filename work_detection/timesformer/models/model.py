import torch
import torch.nn as nn
from timesformer_pytorch import TimeSformer

def get_timesformer_model(img_size=224, num_classes=12, num_frames=16, pretrained=False):
    model = TimeSformer(
        dim=768,
        image_size=img_size,
        patch_size=16,
        num_frames=num_frames,
        num_classes=num_classes,
        depth=12,
        heads=12,
        dim_head=64,
        attn_dropout=0.1,
        ff_dropout=0.1
    )
    if pretrained:
        # 事前学習済みモデルのロード（必要に応じて）
        pass
    return model
