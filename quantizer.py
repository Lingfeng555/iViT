import os
import vai_q_onnx
import numpy as np
from onnxruntime.quantization.calibrate import CalibrationDataReade

# Define a custom calibration data reader
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

# You'll need to populate this with your actual input data
# For example, if your model takes image inputs:
def get_sample_data(num_samples=100, input_shape=(2, 1, 28, 28)):
    # Generate random data for calibration
    # In practice, you should use real representative data from your dataset
    samples = []
    for _ in range(num_samples):
        samples.append(np.random.random(input_shape).astype(np.float32))
    return samples

onnx_path = "onnx"
models = [f for f in os.listdir(onnx_path) if f.endswith('.onnx') and not f.endswith('_int8.onnx')]

for model in models:
    # Load the model to get input names
    model_path = f"{onnx_path}/{model}"
    onnx_model = onnx.load(model_path)
    input_names = [input.name for input in onnx_model.graph.input]
    
    # Generate sample data - adjust shape based on your model's input
    sample_data = get_sample_data()
    
    # Create the calibration data reader
    calibration_data_reader = MyCalibrationDataReader(input_names, sample_data)
    
    model_name = os.path.splitext(model)[0]  # Remove the .onnx extension
    vai_q_onnx.quantize_static(
        model_input=model_path,
        model_output=f"{onnx_path}/{model_name}_int8.onnx",
        calibration_data_reader=calibration_data_reader,
        quant_format=vai_q_onnx.QuantFormat.QDQ,
        activation_type=vai_q_onnx.QuantType.QInt8,
        weight_type=vai_q_onnx.QuantType.QInt8,
        calibrate_method=vai_q_onnx.PowerOfTwoMethod.MinMSE
    )

