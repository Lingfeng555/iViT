import os
import vai_q_onnx
import numpy as np
import onnx
from onnxruntime.quantization.calibrate import CalibrationDataReader
from experiment_hyperparameters import DEVICE, SPLITS, RESULT_PATH
from experiment_result_processing import get_best_size, build_dataset

class MyCalibrationDataReader(CalibrationDataReader):
    def __init__(self, input_names, data_samples, batch_size=32):
        self.input_names = input_names
        self.data_samples = data_samples
        self.batch_size = batch_size
        self.index = 0

    def get_next(self):
        if self.index >= len(self.data_samples):
            return None

        # Acumula batch_size muestras en una lista
        batch_samples = []
        for _ in range(self.batch_size):
            if self.index < len(self.data_samples):
                sample = self.data_samples[self.index]
                self.index += 1
            else:
                # Si se acaban las muestras, se repite la última para completar el batch
                sample = self.data_samples[-1]

            # Se asume que cada muestra ya tiene la forma (1, ch, h, w)
            batch_samples.append(sample)
        
        # Las muestras tienen forma (1, ch, h, w) y las concatenamos a lo largo del eje 0 para formar un batch (batch_size, ch, h, w)
        batch_input = np.concatenate(batch_samples, axis=0)

        # Devuelve un diccionario para cada input
        return {input_name: batch_input for input_name in self.input_names}

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
        if isinstance(sample, tuple):  # Si dataset retorna (input, label)
            sample = sample[0]
        if not isinstance(sample, np.ndarray):
            sample = np.array(sample, dtype=np.float32)
        if sample.ndim == 3:
            sample = np.expand_dims(sample, axis=0)  # Ahora sample tiene forma (1, ch, h, w)
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
