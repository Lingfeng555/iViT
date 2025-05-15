How to Replicate the Experiment
1. Prerequisites

    Python: 3.12.3

    Git

    Graphviz CLI (for the dot tool)

2. Clone the Repository

git clone <repository-url>
cd <repository-name>

3. Install System Dependencies

sudo apt update
sudo apt install -y graphviz

4. Install Python Dependencies

    Note: Be sure you’re using the intended Python interpreter or virtual environment.

pip install -r requirements.txt
pip install graphviz

5. Configure

    measurer.py
    This file interfaces directly with your firmware. If your setup differs from ours, edit it as needed—but please review it carefully before running.

    experiment_hyperparameters.py
    All tunable values (e.g. sample size, timing, etc.) live here. By default, this matches the settings from the original paper; feel free to adjust.

6. Run the Experiment

sudo python3 main.py

    Why sudo?
    The script communicates directly with firmware, so elevated privileges are required.

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
