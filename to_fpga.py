import torch

from experiment_hyperparameters import DEVICE, SPLITS, RESULT_PATH
from experiment_result_processing import get_best_size, rebuild_model


dummy_input =torch.randn(32, 1, 28, 28).to(DEVICE)

for split in SPLITS:
    best_size = get_best_size(path=f"{RESULT_PATH}", split=split)
    model = rebuild_model(dataset=split, size=best_size, path=f"{RESULT_PATH}/{split}/{best_size}/model.pth")
    torch.onnx.export(model, dummy_input, f"onnx/model_{split}_{best_size}.onnx", opset_version=11)