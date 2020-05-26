import torch
import torch.nn as nn 

class Model1(nn.Module):
    def __init__(self):
        super(Model1,self).__init__()

        self.layer1 =  nn.Sequential(
            nn.Conv2d(4,64,3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            nn.Conv2d(64,128,3, padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            nn.Conv2d(128,256,3, padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            
            nn.Conv2d(256,256,3, padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256,512,3, padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512,512,3, padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(output_size=(7, 7))
            )

        self.layer2 = nn.Sequential(
            nn.Linear(in_features=512*7*7, out_features=128, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=128, out_features=128, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=128, out_features=1, bias=True)
        )
        
    def forward(self, x):
        x = self.layer1(x)
        x = x.view(-1,512*7*7)
        x = self.layer2(x)
        return x

class Model2(nn.Module):
    def __init__(self):
        super(Model2,self).__init__()

        self.layer1 =  nn.Sequential(
            nn.Conv2d(4,64,3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            nn.Conv2d(64,128,3, padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            nn.Conv2d(128,256,3, padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            
            nn.Conv2d(256,256,3, padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            
            nn.Conv2d(256,512,3, padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            nn.Conv2d(512,512,3, padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
            )

        self.layer2 = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=512, out_features=1, bias=True)
        )
        
    def forward(self, x):
        x = self.layer1(x)
        x = x.mean(dim=(2,3))
        x = self.layer2(x)
        return x

class Model3(nn.Module):
    def __init__(self):
        super(Model3,self).__init__()
        
        self.layer1 =  nn.Sequential(
            nn.Conv2d(4,64,3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64,128,3, padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128,256,3, padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256,256,3, padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256,512,3, padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512,512,3, padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
            )
        
        self.layer2 = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Conv2d(512,1,3)
        )
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.mean(dim=(2,3))
        return x

class Model4(nn.Module):
    def __init__(self):
        super(Model4,self).__init__()

        self.layer1 =  nn.Sequential(
            nn.Conv2d(4,64,3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            nn.Conv2d(64,128,3, padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            
            nn.Conv2d(128,256,3, padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            
            nn.Conv2d(256,512,3, padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
            )

        self.layer2 = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=512, out_features=1, bias=True)
        )
        
    def forward(self, x):
        x = self.layer1(x)
        x = x.mean(dim=(2,3))
        x = self.layer2(x)
        return x

class Model5(nn.Module):
    def __init__(self):
        super(Model5,self).__init__()

        self.layer1 =  nn.Sequential(
            nn.Conv2d(4,128,3,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128,256,3, padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256,512,3, padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
            )

        self.layer2 = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=512, out_features=1, bias=True)
        )
        
    def forward(self, x):
        x = self.layer1(x)
        x = x.mean(dim=(2,3))
        x = self.layer2(x)
        return x
