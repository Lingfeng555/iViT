import torch
import torch.nn as nn
from .ExtractorComponent.CNNBlock import CNNBlock
from .ExtractorComponent.AxialAttentionBlock import AttentionBlock
from .ExtractorComponent.FCExpert import FCExpert
from .ExtractorComponent.FinalBlock import FinalExpert

class InformationExtractor (nn.Module):

    experts_output: torch.tensor

    def __init__(self, num_experts = 64, output_len = 2, subinfo_size = 2):
        super(InformationExtractor, self).__init__()
        self.num_experts = num_experts
        self.cnn_block = CNNBlock()
        self.attention_block = AttentionBlock(num_features=num_experts, attention_value=1, height=7, width=7)
        self.experts = nn.ModuleList([FCExpert(output_size=subinfo_size) for _ in range(num_experts)])
        self.wighted_sum = FinalExpert(input_size=num_experts*subinfo_size, output_size = output_len)

        self.experts_output = None

    def forward(self, x):
        
        # First phase the convolutional layers
        features = self.cnn_block(x)

        # Second phase the convolutional layers
        attention_values = self.attention_block(features)
        features = features.view(features.shape[0], 64, -1)
        
        # Expert layers
        x = nn.functional.relu(torch.stack([self.experts[i](features[:, i, :], attention_values[:, i, :]) for i in range(self.num_experts)], dim=1))

        # Store the experts output to examine potential biases
        self.experts_output = x

        # Flatten the result for the final expert
        x = x.flatten(start_dim=1)

        # Final expert
        x = self.wighted_sum(x)
        return x
    
    def get_expert_output_dict(self)->dict:
        batch, experts, outputs = self.experts_output.size()

        ls = self.experts_output.to("cpu").detach().numpy().tolist()

        ret = {}

        for y in range(experts):
            for z in range(outputs):
                ret[f"expert_{y}_{z}"] = []

        for x in range(batch):
            for y in range(experts):
                for z in range(outputs):
                    ret[f"expert_{y}_{z}"].append(ls[x][y][z])

        return ret
    
if __name__ == "__main__":
    # Testing
    model = InformationExtractor(output_len=6)
    input_tensor = torch.randn(32, 1, 28, 28)  
    output = model(input_tensor)
    
    print("Forma de la salida después de `view`:", output.shape)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(len(model.get_expert_output_dict()["expert_0_0"]))
