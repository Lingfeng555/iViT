import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNBlock(nn.Module):

    def __init__(self):
        super(CNNBlock, self).__init__() ## We are assuming a grey scale image
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batch_norm = nn.BatchNorm2d(64) 

    def forward(self, x):
        if x.ndim == 3: 
            x = x.unsqueeze(1) 
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))          
        x = F.relu(self.conv3(x))
        x = self.pool(x) 

        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool(x)                   
        
        x = self.batch_norm(x)
        return x

if __name__ == "__main__":
    # Testing
    model = CNNBlock()
    input_tensor = torch.randn(32,1, 28, 28)  
    output = model(input_tensor)

    print("Forma de la salida después de `view`:", output.shape)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")