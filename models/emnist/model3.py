import torch.nn as nn

class Net(nn.Module):
    def __init__(self,num_class):
        super(Net, self).__init__()
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
        self.layer3 = nn.Sequential(
            nn.Conv2d(32,64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(3*3*64, num_class)
        
    def forward(self, x):
        l1=self.layer1(x)
        l2=self.layer2(l1)
        l3 = self.layer3(l2)
        # print(l3.size())
        l3=l3.reshape(l3.size(0), -1)
        out=self.fc(l3)
        return out