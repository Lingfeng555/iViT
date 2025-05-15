import torch
import torch.nn as nn
import torch.nn.functional as F

class FCExpert(nn.Module):
    output_size : int

    def __init__(self, output_size = 2):
        super(FCExpert, self).__init__()
        self.output_size = output_size
        self.fc1 = nn.Linear(49, 40)
        self.fc2 = nn.Linear(40, 30)
        self.fc3 = nn.Linear(30, 20)
        self.fc4 = nn.Linear(20, 10)
        self.fc5 = nn.Linear(10, output_size) 
        self.batch_norm = nn.BatchNorm1d(num_features=output_size)

    def forward(self, x, attention_value):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.batch_norm(x)
        return (x * attention_value)
    
if __name__ == "__main__":
    # Testing
    model = FCExpert()
    input_tensor = torch.randn(32, 49)  
    output = model(input_tensor, 4)
    print("Forma de la salida después de `view`:", output.shape)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")