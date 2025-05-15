import os
from experiment_hyperparameters import RESULT_PATH, SPLITS, SIZES
import shutil

shutil.rmtree(os.path.join(RESULT_PATH, "bias_check"))

SIZES = [str(x) for x in SIZES]

for split in SPLITS:
    dir_and_files = os.listdir(os.path.join(RESULT_PATH, split))
    for dir_or_file in dir_and_files:
        path = os.path.join(RESULT_PATH, split, dir_or_file)
        if dir_or_file in SIZES: # Dont remove the folder
            os.remove(os.path.join(path, "confusion_matrix.png"))
            os.remove(os.path.join(path, "expert_data.csv"))
        else: # Remove the file or folder
            if os.path.isdir( path ):
                print(f"Removing folder {path}")
                shutil.rmtree(path)
            elif os.path.isfile( path ):
                print(f"Removing file {path}")
                os.remove(path)
            else:
                print(f"ERROR {path} DO NOT EXIST")
                
            
