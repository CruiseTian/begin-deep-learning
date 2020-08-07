import torch
from torch import nn

class AlexNet(nn.Module):
    def __init__(self, num_classes = 1000):
        super(AlexNet, self).__init__()
        # 用nn.Sequential()将网络打包成一个模块，精简代码

        # 卷积层提取图像特征
        self.features = nn.Sequential(    
            # Conv1: Input:[3,224,224] Output:[48,55,55]
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  
            nn.ReLU(inplace=True),  # inplace:直接覆盖原值，节省内存
            # Pool1:
            nn.MaxPool2d(kernel_size=3, stride=2),  # 池化，Output:[48,27,27]
            # Conv2: Input:[48,27,27] Output:[128,27,27]
            nn.Conv2d(48, 128, kernel_size=5, stride=1, padding=2),  
            nn.ReLU(True),
            # Pool2:
            nn.MaxPool2d(kernel_size=3, stride=2),  #Output:[128,13,13]
            # Conv3: Input:[128,13,13] Output:[192,13,13]
            nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(True),
            # Conv4: Input:[192,13,13] Output:[192,13,13]
            nn.Conv2d(192, 192, kernel_size=3, padding=1),  
            nn.ReLU(True),
            # Conv5: Input:[192,13,13] Output:[128,13,13]
            nn.Conv2d(192, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            # Pool3:
            nn.MaxPool2d(kernel_size=3, stride=2),  # Output:[128,6,6]
        )

        # 全连接层对图像分类
        self.classifier = nn.Sequential(
            # FC1
            nn.Dropout(p=0.5),  # 随即失活，防止过拟合
            nn.Linear(128*6*6, 2048),  # 相当于求Z = WX+b
            nn.ReLU(True),
            # FC2
            nn.Dropout(p=0.5),
            nn.Linear(2048,2048),
            nn.ReLU(True),
            # FC3
            nn.Linear(2048, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x
