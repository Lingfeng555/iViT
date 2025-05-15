# How to Replicate the Experiment

## 1. Prerequisites

- Python 3.12.3
- Git
- Graphviz CLI (provides the dot tool)

## 2. Clone the Repository

```bash
git clone <repository-url>
cd <repository-name>
```

## 3. Install System Dependencies

```bash
sudo apt update
sudo apt install -y graphviz
```

## 4. Install Python Dependencies

Ensure you are using the correct Python interpreter or have activated your virtual environment.

```bash
pip install -r requirements.txt
pip install graphviz
```

## 5. Configure Settings

- **measurer.py**  
    This file interfaces directly with your firmware. If your setup differs, modify it accordingly and review changes carefully before running.

- **experiment_hyperparameters.py**  
    Contains all tunable settings (e.g., sample size, timing). These default values match the original paper's configuration but can be adjusted as needed.

## 6. Run the Experiment

```bash
sudo python3 main.py
```

> **Note:** Running the experiment with `sudo` is required since the script interacts directly with the firmware, necessitating elevated privileges.

# Repository Structure

```
.
├── Architecture/
│   ├── iViT_EMNIST/        # Adapted iViT implementation for the EMNIST dataset.
│   └── NEU/               # Adaptation of iViT for the NEU dataset.
│
├── quantizer.py           # Quantize any trained model.
├── to_fpga.py             # Convert and export quantized models for FPGA deployment.
│
├── experiment_visualization.ipynb  # Generate plots and visual summaries of replication results.
├── t_student.ipynb        # Compute and illustrate confidence intervals using Student’s t-test.
│
└── experiment_hyperparameters.py   # Defines all tunable settings (e.g., RESULT_PATH, sample sizes, timing).
```

## Detailed Overview

- **Architecture/**  
    Contains iViT adaptations:
    - *iViT_EMNIST/* – Customized for the EMNIST dataset.
    - *NEU/* – Customized for the NEU dataset.

- **quantizer.py / to_fpga.py**  
    - *quantizer.py*: Applies quantization to model checkpoints.
    - *to_fpga.py*: Packages the quantized model for FPGA inference.

- **Notebooks**  
    - *experiment_visualization.ipynb*: Visualizes and compares key metrics from experiments.
    - *t_student.ipynb*: Calculates and plots confidence intervals based on Student’s t-test.

- **experiment_hyperparameters.py**  
    Centralizes all experiment settings. Update `RESULT_PATH` to specify where outputs (logs, figures, stats) are saved.

For additional iViT adaptations, please explore the following repository:  
[Efficient iViT for Card Recognition](https://github.com/Lingfeng555/CardsRecognition.git)
