import torch
import torch.nn as nn
import torchvision
from torchvision.models.video import R3D_18_Weights

class R3D_Model(nn.Module):
    def __init__(self, num_classes=12, pretrained=True):
        super(R3D_Model, self).__init__()
        if pretrained:
            weights = R3D_18_Weights.DEFAULT
        else:
            weights = None
        self.model = torchvision.models.video.r3d_18(weights=weights)# in_features を取得
        in_features = self.model.fc.in_features
        # 最終全結合層を除去
        self.model.fc = nn.Identity()
        # 新たな全結合層
        self.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        # x: (batch_size, C, T, H, W)
        batch_size, C, T, H, W = x.size()
        x = self.model.stem(x)  # (batch_size, 64, T/2, H/2, W/2)
        x = self.model.layer1(x)  # 時間次元: T/2
        x = self.model.layer2(x)  # 時間次元: T/4
        x = self.model.layer3(x)  # 時間次元: T/8
        x = self.model.layer4(x)  # 時間次元: T/8

        # 時間次元が過度に縮小されないように、layer4 のプーリングを無効化
        # または、layer4 を使用しないことも検討
        # ここでは、layer4 を使用しない例を示します
        # x = self.model.layer3(x)  # 時間次元: T/8

        # 空間次元を平均化
        x = x.mean(dim=[3, 4])  # (batch_size, 512, T_out)

        # 時間次元をフレームごとに展開
        x = x.permute(0, 2, 1)  # (batch_size, T_out, 512)
        x = x.reshape(-1, x.size(-1))  # (batch_size * T_out, 512)
        outputs = self.fc(x)  # (batch_size * T_out, num_classes)
        outputs = outputs.view(batch_size, -1, outputs.size(-1))  # (batch_size, T_out, num_classes)

        # フレーム数に合わせて線形補間
        outputs = nn.functional.interpolate(outputs.permute(0, 2, 1), size=T, mode='linear', align_corners=False)
        outputs = outputs.permute(0, 2, 1)  # (batch_size, T, num_classes)
        return outputs
