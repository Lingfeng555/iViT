import torch
import torch.nn as nn
import pandas as pd
from .ExtractorComponent.CNNBlock import CNNBlock
from .ExtractorComponent.AxialAttentionBlock import AttentionBlock
from .ExtractorComponent.FCExpert import FCExpert
from .ExtractorComponent.FinalBlock import FinalExpert

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class InformationExtractor (nn.Module):

    experts_output: torch.tensor
    cnn_block: CNNBlock
    attention_block: AttentionBlock
    experts: nn.ModuleList
    wighted_sum: FinalExpert
    subinfo_size: int
    num_experts: int

    def __init__(self, num_experts = 64, output_len = 2, subinfo_size = 2):
        super(InformationExtractor, self).__init__()
        self.num_experts = num_experts
        self.subinfo_size = subinfo_size
        self.cnn_block = CNNBlock()
        self.attention_block = AttentionBlock(num_features=num_experts, attention_value=1, height=7, width=7)
        self.experts = nn.ModuleList([FCExpert(output_size=subinfo_size) for _ in range(num_experts)])
        self.wighted_sum = FinalExpert(input_size=num_experts*subinfo_size, output_size = output_len)

        self.experts_output = None

    def forward(self, x):
        batch, _, _, _ = x.size()
        # First phase the convolutional layers
        features = self.cnn_block(x)

        # Second phase the convolutional layers
        attention_values = self.attention_block(features)
        features = features.view(features.shape[0], 64, -1)
        
        # Expert layers
        x = nn.functional.relu(
            torch.stack(
                [
                    self.experts[i](features[:, i, :], attention_values[:, i, :])
                    if not isinstance(self.experts[i], nn.Identity)
                    else torch.zeros(batch, self.subinfo_size, device=DEVICE)
                    for i in range(self.num_experts)
                ],
                dim=1
            )
        )

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
    
    def prune_experts(self, list_of_experts: list) -> None:
        '''Receive a list of indexes of experts'''
        for expert_index in list_of_experts:
            self.experts[expert_index] = nn.Identity().to(DEVICE)
    
if __name__ == "__main__":
    # Testing
    model = InformationExtractor(output_len=6).to(DEVICE)
    input_tensor = torch.randn(32, 1, 28, 28).to(DEVICE)
    output = model(input_tensor)
    
    print("Forma de la salida después de `view`:", output.shape)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    pruning_list = list(range(20,50))
    model.prune_experts(pruning_list)
    input_tensor = torch.randn(32, 1, 28, 28).to(DEVICE)
    output = model(input_tensor)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
