import torch
from torch import nn
import math
from torch.utils.data import DataLoader, Dataset
import os
import cv2
from torchvision import transforms

# 常见的3x3卷积
def conv3x3(in_channel, out_channel, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride,
                     padding=1, bias=False)
    
class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        """
        Args:
            in_channel (_type_): 输入通道数
            out_channel (_type_): 输出通道数，BasicBlock中输出通道数不会扩增4倍
            stride (int, optional): 是否在第一个conv3x3进行下采样（一般从stage2往后的第一个Block要进行分辨率的下采样）
            downsample (_type_, optional): 作用：若在第一个conv3x3进行了下采样，则需要downsampe的1x1卷积进行分辨率的统一
        """
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channel, out_channel, stride) # 
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class Bottleneck(nn.Module):
    expansion = 4      # 输出通道数的倍乘

    def __init__(self, in_channel, out_channel, stride=1, downsample=None): 
        """

        Args:
            in_channel (_type_): 输入通道数
            out_channel (_type_): 输出通道数，最终的输出通道数会扩增4倍
            stride (int, optional): 是否在第一个conv3x3进行下采样（一般从stage2往后的第一个Block要进行分辨率的下采样）
            downsample (_type_, optional): downsample的作用：1. 统一residual的通道数；2. 在第一轮进行下采样降低分辨率
        """
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(out_channel, out_channel * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):  # layers=参数列表 block选择不同的类
        super(ResNet, self).__init__()
        self.in_channel = 64 
        
        # 3×224×224 ---> 64×56×56
        self.layer0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False), # 2x Down
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # 2x Down
        )
        
        # 64×56×56 ---> (64×block.expansion)×56×56
        self.layer1 = self._make_layer(block, 64, layers[0])
        
        # (64×block.expansion)×56×56 ---> (128×block.expansion)×28×28
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        
        # (128×block.expansion)×28×28 ---> (256×block.expansion)×14×14
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        
        #  (256×block.expansion)×28×28 ---> (512×block.expansion)×7×7
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # (512×block.expansion)×7×7 ---> (1, num_classes)
        self.final_layer = nn.Sequential(
            nn.AvgPool2d(7),
            nn.Flatten(),
            nn.Linear(512 * block.expansion, num_classes)
        )

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, out_channel, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != out_channel * block.expansion: # downsample出现的情景：1. 分辨率下采样；2. 统一通道数
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, out_channel * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channel, out_channel, stride, downsample))
        
        self.in_channel = out_channel * block.expansion # 直接改变下一个Block输入的通道数，扩大expansion倍，out_channel始终不变
        for i in range(1, blocks):
            layers.append(block(self.in_channel, out_channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.final_layer(x)

        return x
    
def resnet18():
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    return model

def resnet34():
    model = ResNet(BasicBlock, [3, 4, 6, 3])
    return model

def resnet50():
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    return model

def resnet101():
    model = ResNet(Bottleneck, [3, 4, 23, 3])
    return model

def resnet152():
    model = ResNet(Bottleneck, [3, 8, 36, 3])
    return model

train_path = "/home/ychen/Datasets/DogsCatsData/train"
train_data = os.listdir(train_path)

class DogAndCatDataset(Dataset):    
    def __init__(self, file_path, file_list, resize=False) -> None:
        super().__init__()
        self.file_path = file_path
        self.file_list = file_list
        self.resize = resize
    
    def __getitem__(self, index):
        image = cv2.imread(self.file_path + os.sep + self.file_list[index])
        label = 1 if "dog" in self.file_list[index] else 0
        
        trans = [transforms.ToTensor()]
        if self.resize:
            trans.append(transforms.Resize((self.resize, self.resize)))
        trans = transforms.Compose(trans)
        image = trans(image)
        
        return image, label

    def __len__(self):
        return len(self.file_list)
    
def evaluate(net, data_iter):
    net.eval()

    acc_sum, num = 0, 0
    for X, y in data_iter:
        X = X.to("cuda:0")
        y = y.to("cuda:0")
        
        y_hat = net(X)
        acc_sum += torch.sum((y_hat.argmax(axis=1) == y).type(y.dtype)).item()
        num += y.numel()
        
    return acc_sum / num

def get_k_fold_data(k, i, train_data):
    X_train, X_valid = [], []
    
    fold = len(train_data) // k
    
    for j in range(k):
        X_part = train_data[j * fold : (j + 1) * fold]
        if j == i:
            X_valid = X_valid + X_part
        else:
            X_train = X_train + X_part
    
    return X_train, X_valid

def train(net, X_train_iter, X_valid_iter, num_epochs, lr):
    net.to("cuda:0")
    net.train()
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss(reduction="none")
    
    train_l, train_acc, valid_acc = [], [], []
    for epoch in range(num_epochs):
        train_l_sum, num = 0, 0
        for X, y in X_train_iter:
            X = X.to("cuda:0")
            y = y.to("cuda:0")
            
            trainer.zero_grad()
            y_hat = net(X)
            l = loss(y_hat, y)
            l.mean().backward()
            trainer.step()
            
            with torch.no_grad():
                train_l_sum += l.sum().item()
                num += y.numel()
        train_l.append(train_l_sum / num)
        train_acc.append(evaluate(net, X_train_iter))
        if X_valid_iter:
            valid_acc.append(evaluate(net, X_valid_iter))
            print(f"epoch {epoch}, train_l {train_l[epoch]}, train_acc {train_acc[epoch]}, valid_acc {valid_acc[epoch]}")
        else:
            print(f"epoch {epoch}, train_l {train_l[epoch]}, train_acc {train_acc[epoch]}")
            
def k_fold(k, net, train_data, num_epochs, lr):
    net.to("cuda:0")
    net.train()
    
    for i in range(k):
        print(f"---------current fold is {i} fold---------")
        X_train, X_valid = get_k_fold_data(k, i, train_data)
        X_train_iter = DataLoader(DogAndCatDataset(train_path, X_train, resize=224), batch_size=128, shuffle=True)
        X_valid_iter = DataLoader(DogAndCatDataset(train_path, X_valid, resize=224), batch_size=128)

        train(net, X_train_iter, X_valid_iter, num_epochs, lr)
        
# k = 5
num_epochs = 10
lr = 0.001
net = resnet18()
X_train_iter = DataLoader(DogAndCatDataset(train_path, train_data, resize=224), batch_size=128, shuffle=True)
train(net, X_train_iter, None, num_epochs, lr)