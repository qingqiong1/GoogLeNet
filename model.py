import torch
from torch import nn
from torchsummary import summary

class Inception(nn.Module):

    def __init__(self,in_channels,out_channels1,out_channels2,out_channels3,out_channels4,):
        super(Inception,self).__init__()
        self.Rule= nn.ReLU()
        self.p1_1 = nn.Conv2d(in_channels,out_channels1,kernel_size=1)

        self.p2_1 = nn.Conv2d(in_channels,out_channels2[0],kernel_size=1)
        self.p2_2 = nn.Conv2d(out_channels2[0],out_channels2[1],kernel_size=3,padding=1)

        self.p3_1 = nn.Conv2d(in_channels,out_channels3[0],kernel_size=1)
        self.p3_2 = nn.Conv2d(out_channels3[0],out_channels3[1],kernel_size=5,padding=2)

        self.p4_1 = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        self.p4_2 = nn.Conv2d(in_channels,out_channels4,kernel_size=1)

    def forward(self,x):
        p1 = self.Rule(self.p1_1(x))
        p2 = self.Rule(self.p2_2(self.Rule(self.p2_1(x))))
        p3 = self.Rule(self.p3_2(self.Rule(self.p3_1(x))))
        p4 = self.Rule(self.p4_2(self.p4_1(x)))

        return torch.cat((p1,p2,p3,p4),dim=1)

class googlenet(nn.Module):
    def __init__(self,Inception):
        super(googlenet,self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=3,padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=192,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
        )
        self.block3 = nn.Sequential(
            Inception(192,64,(96,128),(16,32),32),
            Inception(256,128,(128,192),(32,96),64),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
        )
        self.block4 = nn.Sequential(
            Inception(480,192,(96,208),(16,48),64),
            Inception(512,160,(112,224),(24,64),64),
            Inception(512,128,(128,256),(24,64),64),
            Inception(512,112,(128,288),(32,64),64),
            Inception(528,256,(160,320),(32,128),128),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
        )
        self.block5 = nn.Sequential(
            Inception(832,256,(160,320),(32,128),128),
            Inception(832,384,(192,384),(48,128),128),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(1024,2)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight,0,0.01)
                nn.init.constant_(m.bias,0)

    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = googlenet(Inception).to(device)
    print(summary(model,(3,224,224)))


