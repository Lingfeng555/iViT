import pandas as pd
import torch
import numpy as np
import random
import os
import time

from utils.Watchers import ExperimentWatcher, ReplicationWatcher
from experiment_hyperparameters import RESULT_PATH, SPLITS, REPLICATION_DURATION, REPLICATION_MODELS, REPLICATION_PATH, TRAIN_SCRIPT, REPLICATION_SCRIPT, SIZES, SEED
from experiment_result_processing import process_experiment_result, generateGradCam, get_decision_tree_svg, get_agglomerative_dendrogram_svg, get_best_size, update_consumption_df
from utils.DefaultLogger import DefaultLogger
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

main_logger = DefaultLogger(name="main_logger")

def set_torch_seed(seed: int = 555):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def train_evaluate_and_meassure_the_models():
    for split in SPLITS:
        for size in SIZES:
            cmd = f"python3 {TRAIN_SCRIPT} {split} {size}"
            main_logger.info(f"Executing :{cmd}")
            
            watcher = ExperimentWatcher(base_path=f"{RESULT_PATH}/{split}/{size}")
            watcher.measure_energy_consumption(executable_file_path=TRAIN_SCRIPT, timelapse=1, split=split, subinfosize=size)
            torch.cuda.empty_cache()

def generate_gradcam_for_the_best_models():
    for split in SPLITS:
        main_logger.info(f"Generating GradCam for the best model of {split}")
        generateGradCam(RESULT_PATH, split=split, sample_size=10)
def replicate_other_classical_models_and_meassure_the_energy_consumption():
    for split in SPLITS:
        macro = {"n_images": [], "duration" :[]}
        main_logger.info(f"Replicating models for {split}")
        for model in REPLICATION_MODELS:
            main_logger.info(f"Replicating {model}")
            cmd = [
                "python3",
                REPLICATION_SCRIPT,
                "--seconds",
                str(REPLICATION_DURATION),
                "--model",
                model,
                "--split",
                split
            ]
            main_logger.info(cmd)
            watcher = ReplicationWatcher(base_path=f"{REPLICATION_PATH}/{split}/results/{model}/")
            n_images, duration = watcher.measure_energy_consumption(cmd, timelapse=1)

            macro["n_images"].append(int(n_images))
            macro["duration"].append(int(duration))
        pd.DataFrame(macro).to_csv(f"{REPLICATION_PATH}/{split}/macro.csv", index=False)

def generate_experts_trees_svg():
    for split in SPLITS:
        main_logger.info(f"Generating tree plots for the best model in {split}")
        best_size = get_best_size(path=f"{RESULT_PATH}", split=split)
        csv_file = os.path.join(RESULT_PATH, split, str(best_size), "expert_data.csv")
        get_decision_tree_svg(csv_file=csv_file, result_path=RESULT_PATH, target_label="pred_label")
        get_decision_tree_svg(csv_file=csv_file, result_path=RESULT_PATH, target_label="true_label")
        #get_agglomerative_dendrogram_svg(csv_file=csv_file, result_path=RESULT_PATH, target_label="pred_label")
        #get_agglomerative_dendrogram_svg(csv_file=csv_file, result_path=RESULT_PATH, target_label="true_label")

if __name__ == '__main__':

    set_torch_seed(SEED)
    
    train_evaluate_and_meassure_the_models()
    
    process_experiment_result(RESULT_PATH)
    
    generate_gradcam_for_the_best_models()
    
    generate_experts_trees_svg()
        
    replicate_other_classical_models_and_meassure_the_energy_consumption()
    
    update_consumption_df(REPLICATION_PATH, SPLITS, REPLICATION_MODELS)
    

