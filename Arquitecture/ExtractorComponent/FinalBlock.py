import torch
import torch.nn as nn
import torch.nn.functional as F

class FinalExpert(nn.Module):
    def __init__(self, input_size=144, output_size=2):
        """
        @param input_size: Número de características de entrada.
        @param output_size: Número de neuronas en la capa de salida.
        """
        super(FinalExpert, self).__init__()
        self.layers = nn.ModuleList()
        current_size = input_size
    
        while True:
            next_size = int(current_size / 1.5)  
            if next_size <= output_size:
                self.layers.append(nn.Linear(current_size, output_size))
                break
            else:
                self.layers.append(nn.Linear(current_size, next_size))
                current_size = next_size

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return x

if __name__ == "__main__":
    model = FinalExpert()
    input_tensor = torch.randn(43, 64)  
    output = model(input_tensor)
    print("Forma de la salida después de `view`:", output.shape)