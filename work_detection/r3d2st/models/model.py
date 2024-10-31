import torch
import torch.nn as nn
import torchvision
from torchvision.models.video import R3D_18_Weights

class SensorStream(nn.Module):
    def __init__(self, input_channels, sensor_channel):
        super(SensorStream, self).__init__()
        # Define a simple 3D CNN for the sensor data
        self.conv1 = nn.Conv3d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)  # Reduces T by half
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)  # Reduces T by half
        self.conv3 = nn.Conv3d(32, sensor_channel, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)  # Further reduces T by half

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = nn.functional.relu(x)
        x = self.pool3(x)
        x = x.mean(dim=[3,4])  # Global Average Pooling over spatial dimensions
        x = x.permute(0, 2, 1)  # (batch_size, T_out, sensor_channel)
        x = x.reshape(-1, x.size(-1))  # (batch_size * T_out, sensor_channel)
        return x

class R3D_TwoStream_Model(nn.Module):
    def __init__(self, num_classes=12, pretrained=True, sensor_input_channels=1, sensor_channel=64):
        super(R3D_TwoStream_Model, self).__init__()
        # Main video stream
        if pretrained:
            weights = R3D_18_Weights.DEFAULT
        else:
            weights = None
        self.video_model = torchvision.models.video.r3d_18(weights=weights)
        in_features = self.video_model.fc.in_features
        self.video_model.fc = nn.Identity()

        # Sensor stream
        self.sensor_stream = SensorStream(sensor_input_channels, sensor_channel)

        # New fully connected layer
        total_in_features = in_features + sensor_channel
        self.fc = nn.Linear(total_in_features, num_classes)

    def forward(self, x_video, x_sensor):
        # x_video: (batch_size, C, T, H, W)
        # x_sensor: (batch_size, C_s, T, H_s, W_s)
        batch_size, C, T, H, W = x_video.size()
        # Process video stream
        x_v = self.video_model.stem(x_video)  # (batch_size, 64, T/2, H/2, W/2)
        x_v = self.video_model.layer1(x_v)
        x_v = self.video_model.layer2(x_v)
        x_v = self.video_model.layer3(x_v)
        x_v = self.video_model.layer4(x_v)
        x_v = x_v.mean(dim=[3,4])  # (batch_size, 512, T_out)
        x_v = x_v.permute(0,2,1)  # (batch_size, T_out, 512)
        x_v = x_v.reshape(-1, x_v.size(-1))  # (batch_size * T_out, 512)

        # Process sensor stream
        x_s = self.sensor_stream(x_sensor)  # (batch_size * T_out, sensor_channel)

        # Concatenate features
        x = torch.cat((x_v, x_s), dim=1)  # (batch_size * T_out, 512 + sensor_channel)

        # Fully connected layer
        outputs = self.fc(x)  # (batch_size * T_out, num_classes)
        outputs = outputs.view(batch_size, -1, outputs.size(-1))  # (batch_size, T_out, num_classes)

        # Interpolate to match the original time steps T
        outputs = nn.functional.interpolate(outputs.permute(0, 2, 1), size=T, mode='linear', align_corners=False)
        outputs = outputs.permute(0, 2, 1)  # (batch_size, T, num_classes)
        return outputs
