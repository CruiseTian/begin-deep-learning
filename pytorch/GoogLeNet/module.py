import torch
from torch import nn
import torch.nn.functional as F

class GoogLeNet(nn.Module):
    # 传入的参数中aux_logits=True表示训练过程用到辅助分类器，aux_logits=False表示验证过程不用辅助分类器
    def __init__(self, num_classes=1000, aux_logits=True):
        super(self, GoogLeNet).__init__()
        self.aux_logits = aux_logits

        # Input:[batch,3,224,224], Output:[batch,64,112,112]
        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=2)
        # Output:[batch,64,56,56]
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True) # ceil_mode为True表示若算出来不为整数，则向上取整

        self.conv2 = nn.Sequential(
            # Input:[batch,64,56,56], Output:[batch,64,56,56]
            BasicConv2d(64, 64, kernel_size=1),
            # Input:[batch,64,56,56], Output:[batch,192,56,56]
            BasicConv2d(64, 192, kernel_size=3, padding=1),
        )
        # Output:[batch,192,28,28]
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        # Input:[batch,192,28,28], Output:[batch,256,28,28]
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        # Input:[batch,256,28,28], Output:[batch,480,28,28]
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)

        # Output:[batch,480,14,14]
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        # Input:[batch,480,14,14], Output:[batch,512,14,14]
        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        # Input:[batch,512,14,14], Output:[batch,512,14,14]
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        # Input:[batch,512,14,14], Output:[batch,512,14,14]
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        # Input:[batch,512,14,14], Output:[batch,528,14,14]
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        # Input:[batch,528,14,14], Output:[batch,832,14,14]
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)

        # Output:[batch,832,7,7]
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        # Input:[batch,832,7,7], Output:[batch,832,7,7]
        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        # Input:[batch,832,7,7], Output:[batch,1024,7,7]
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        if self.aux_logits:
            # Input:[batch,512,14,14], which is from inception4a
            self.aux1 = InceptionAux(512, num_classes)
            # Input:[batch,528,14,14], which is from inception4d
            self.aux2 = InceptionAux(528, num_classes)

        # Output:[batch,1024,1,1]
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        # batch x 3 x 224 x 224
        x = self.conv1(x)
        # batch x 64 x 112 x 112
        x = self.maxpool1(x)
        # batch x 64 x 56 x 56
        x = self.conv2(x)
        # batch x 192 x 56 x 56
        x = self.maxpool2(x)

        # batch x 192 x 28 x 28
        x = self.inception3a(x)
        # batch x 256 x 28 x 28
        x = self.inception3b(x)
        # batch x 480 x 28 x 28
        x = self.maxpool3(x)
        # batch x 480 x 14 x 14
        x = self.inception4a(x)
        # batch x 512 x 14 x 14
        if self.training and self.aux_logits:
            out1 = self.aux1(x)

        x = self.inception4b(x)
        # batct x 512 x 14 x 14
        x = self.inception4c(x)
        # batch x 512 x 14 x 14
        x = self.inception4d(x)
        # batch x 528 x 14 x 14
        if self.training and self.aux_logits:
            out2 = self.aux2(x)

        x = self.inception4e(x)
        # batch x 832 x 14 x 14
        x = self.maxpool4(x)
        # batch x 832 x 7 x 7
        x = self.inception5a(x)
        # batch x 832 x 7 x 7
        x = self.inception5b(x)
        # batch x 1024 x 7 x 7

        x = self.avgpool(x)
        # batch x 1024 x 1 x 1
        x = torch.flatten(x,start_dim=1)
        # batch x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # batch x num_classes
        if self.training and self.aux_logits:   # eval model lose this layer
            return x, aux2, aux1
        return x



class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x        

# Inception module
class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1),
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2),
        )

        self.barnch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

        def forward(self, x):
            branch1 = self.branch1(x)
            branch2 = self.branch2(x)
            branch3 = self.branch3(x)
            branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1) # 按 channel 对四个分支拼接  

# 辅助分类器
class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()

        self.averagePool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1) # output[batch, 128, 4, 4]

        self.fc1 = nn.Linear(128*4*4, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        # aux1: batch x 512 x 14 x 14, aux2: batch x 528 x 14 x 14
        x = self.averagePool(x)
        # aux1: batch x 512 x 4 x 4, aux2: batch x 528 x 4 x 4
        x = self.conv(x)
        # batch x 128 x 4 x 4
        x = torch.flatten(x, 1)
        x = F.dropout(x, 0.5, training=self.training)
        # batch x 128*4*4
        x = self.fc1(x)
        x = F.relu(x, True)
        x = F.dropout(x, 0.5, training=self.training)
        # batch x 1024
        x = self.fc2(x)
        # batch x num_classes
        return x
