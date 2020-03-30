import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self,num_class):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1,16,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16,32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_class)
        
    def forward(self, x):
        l1=self.layer1(x)
        l2=self.layer2(l1)
        l3=l2.reshape(l2.size(0), -1)
        out=self.fc(l3)
        return out