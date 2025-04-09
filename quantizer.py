import os
import vai_q_onnx
import numpy as np
import onnx
from onnxruntime.quantization.calibrate import CalibrationDataReader
from experiment_hyperparameters import DEVICE, SPLITS, RESULT_PATH
from experiment_result_processing import get_best_size, build_dataset

class MyCalibrationDataReader(CalibrationDataReader):
    def __init__(self, input_names, data_samples):
        self.input_names = input_names
        self.data_samples = data_samples
        self.index = 0
        
    def get_next(self):
        if self.index >= len(self.data_samples):
            return None
        
        result = {}
        for input_name in self.input_names:
            result[input_name] = self.data_samples[self.index]
        
        self.index += 1
        return result

    def rewind(self):
        self.index = 0


onnx_path = "onnx"
models = [f for f in os.listdir(onnx_path) if f.endswith('.onnx') and not f.endswith('_int8.onnx')]

for model in models:
    # Load the model to get input names
    model_path = f"{onnx_path}/{model}"
    onnx_model = onnx.load(model_path)
    input_names = [input.name for input in onnx_model.graph.input]
    
    # Generate sample data 
    dataset = build_dataset(model.split("_")[1])
    num_calibration_samples = min(100, len(dataset))
    calibration_samples = []
    
    for i in range(num_calibration_samples):
        sample = dataset[i]
        if isinstance(sample, tuple):  # If dataset returns (input, label) pairs
            sample = sample[0]
        if not isinstance(sample, np.ndarray):
            sample = np.array(sample, dtype=np.float32)
        calibration_samples.append(sample)
    
    # Create the calibration data reader
    calibration_data_reader = MyCalibrationDataReader(input_names, calibration_samples)
    
    # Perform quantization
    model_name = os.path.splitext(model)[0]  # Remove the .onnx extension
    print(f"Quantizing model: {model}")
    vai_q_onnx.quantize_static(
        model_input=model_path,
        model_output=f"{onnx_path}/{model_name}_int8.onnx",
        calibration_data_reader=calibration_data_reader,
        quant_format=vai_q_onnx.QuantFormat.QDQ,
        activation_type=vai_q_onnx.QuantType.QInt8,
        weight_type=vai_q_onnx.QuantType.QInt8,
        calibrate_method=vai_q_onnx.PowerOfTwoMethod.MinMSE
    )
    print(f"Quantization complete for {model}")
