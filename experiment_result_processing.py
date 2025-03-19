import pandas as pd
import os
import torch
import torchvision
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import glob
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import cv2
import numpy as np
from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import subprocess
from sklearn.metrics import precision_score
import optuna
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from utils.Loader import EMNISTDataset, FashionMNISTDataset
from Arquitecture import InformationExtractor
from experiment_hyperparameters import DEVICE, BATCH_SIZE, SEED, N_TRIALS, RESULT_PATH, THREADS
from utils.AttentionGradCam import AttentionGradCAM
import random
# Experiment

def emnist_number_to_text(idx):
    if idx < 0 or idx > 46:
        raise ValueError(f"Índice {idx} fuera de rango (debe estar entre 0 y 46).")
    
    mapping_list = (
        [str(d) for d in range(10)] +                   # '0'–'9'  0–9
        [chr(c) for c in range(65, 91)] +               # 'A'–'Z'  10–35
        ['a','b','d','e','f','g','h','n','q','r','t']   # 'a'-'t'  36–46
    )
    return mapping_list[idx]

def rebuild_model(dataset: str, size: int, path: str):
    if dataset == "balanced": model = InformationExtractor(output_len=47, subinfo_size=size)
    else: model = InformationExtractor(output_len=10, subinfo_size=size)
    model.load_state_dict(torch.load(path, map_location=torch.device(DEVICE)))
    model.eval()
    return model

def build_dataset(dataset: str):
    if dataset == "fashion": return FashionMNISTDataset(download=True, train=False)
    else: return EMNISTDataset(split=dataset, train=False, download=True)

def prediction(image, model):
    output = model(image.to(DEVICE))
    pred_class = torch.argmax(output, dim=1)
    return pred_class

def get_results(test, model: InformationExtractor):
    true_labels = []
    pred_labels = []
    expert_outputs = {}
    model.eval()
    test_loader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False)
    with torch.no_grad():
        for images, labels in test_loader:
            predictions = prediction(images, model)

            for pred, label in zip(predictions, labels):
                label_idx = torch.argmax(label).item()
                true_labels.append(label_idx)
                pred_labels.append(pred.item())
                
            new_outputs = model.get_expert_output_dict()
            if not expert_outputs:
                expert_outputs = new_outputs
            else:
                for k in new_outputs:
                    expert_outputs[k] += new_outputs[k]
    
    expert_outputs["true_label"] = true_labels
    expert_outputs["pred_label"] = pred_labels
    
    return true_labels, pred_labels, expert_outputs

def plot_and_save_confusion_matrix(true_labels, pred_labels, save_path, num_parameters):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Etiquetas Predichas")
    plt.ylabel("Etiquetas Verdaderas")
    plt.title(f"Matriz de Confusión. Num.Param{num_parameters}")
    plt.savefig(save_path)
    plt.close()

def process_and_save_plots(PIPE_PATH: str):
    result_per_datasets = os.listdir(PIPE_PATH)
    for dataset in result_per_datasets:
        models = sorted([d for d in os.listdir(os.path.join(PIPE_PATH, dataset))
                        if os.path.isdir(os.path.join(PIPE_PATH, dataset, d)) and (d[0] in "0123456789") ])
        test_set = build_dataset(dataset=dataset)
        for size in models:
            model_path  = os.path.join(PIPE_PATH, dataset, size, "model.pth")
            model = rebuild_model(dataset=dataset, size=int(size), path=model_path).to(DEVICE)
            num_parameters = sum(p.numel() for p in model.parameters())
            true_labels, pred_labels, expert_outputs = get_results(test_set, model)
            plot_and_save_confusion_matrix(true_labels=true_labels, pred_labels=pred_labels, save_path=os.path.join(PIPE_PATH, dataset, size,"confusion_matrix.png"), num_parameters=num_parameters)
            pd.DataFrame(expert_outputs).to_csv( os.path.join(PIPE_PATH, dataset, size,"expert_data.csv") )

def read_csv_files(folder_path):
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    df_list = []
    for file in csv_files:
        df = pd.read_csv(file)
        file_name = os.path.basename(file)
        df['filename'] = file_name
        df_list.append(df)
    if df_list:
        combined_df = pd.concat(df_list, ignore_index=True)
    else:
        combined_df = pd.DataFrame()
    return combined_df

def plot_over_time(df, column_x, column_Y, path:str = None):
    plt.figure(figsize=(8, 5))
    plt.plot(df[column_x], df[column_Y], linestyle='-')
    plt.xlabel(column_x)
    plt.ylabel(column_Y)
    plt.title(f' {column_Y} Over {column_x}')
    plt.grid(True)
    plt.savefig(path)
    plt.close()

def plot_bar_save(df, column_x, column_Y, path: str = None):
    plt.figure(figsize=(8, 5))
    plt.bar(df[column_x], df[column_Y])
    plt.xlabel(column_x)
    plt.ylabel(column_Y)
    plt.title(f'{column_Y} Over {column_x}')
    plt.grid(True)
    if path:
        plt.savefig(path)
    else:
        plt.show()
    plt.close()

def analyze_cpu_consumption(cpu_folder: str):
    df = read_csv_files(os.path.join(cpu_folder, "general"))
    df['Moment'] = pd.to_datetime(df['Moment'], format='%Y-%m-%d %H:%M:%S.%f')
    df['Seconds'] = (df['Moment'] - df['Moment'].iloc[0]).dt.total_seconds()
    df['J'] = df['uJ'] / 1e6
    df = df.drop(columns=["filename", "uJ"])

    df['Delta_J'] = df['J'].diff()
    df['Delta_t'] = df['Seconds'].diff()
    max_counter_value = df['J'].max() + 1

    wrap_mask = df['Delta_J'] < 0
    df.loc[wrap_mask, 'Delta_J'] = (max_counter_value - df['J'].shift(1)) + df['J']

    df['Watts'] = df['Delta_J'] / df['Delta_t']
    plot_over_time(df, column_x = "Seconds", column_Y="Watts", path=os.path.join(cpu_folder, "cpu_watts_timeline.png"))
    plot_over_time(df, column_x = "Seconds", column_Y="CPU_usage", path=os.path.join(cpu_folder, "cpu_usage_timeline.png"))

    df = df.drop(columns=["Moment", "Delta_J", "Delta_t", "J"])

    ret = df.describe()

    cores = read_csv_files(os.path.join(cpu_folder, "cores")).describe()
    
    ret["Frecuency (MHz) Cores"] = cores["Frecuencia (MHz)"]
    ret["Temperatura (°C) Cores"] = cores["Temperatura (°C)"]
    ret.drop(ret.index[0], inplace=True)

    ret.to_csv(os.path.join(cpu_folder, "cpu_info.csv"))
    return ret, df

def analyze_gpu_consumption(gpu_folder: str):
    df = read_csv_files(gpu_folder).drop(columns=["filename", "name"])
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y/%m/%d %H:%M:%S.%f')
    df['Seconds'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()
    df = df.drop(columns=["timestamp"])
    
    ret = df.describe()
    ret.drop(ret.index[0], inplace=True)
    
    columns = df.columns.to_list()
    columns.remove("pstate")

    if "Unnamed: 0" in columns: columns.remove("Unnamed: 0")

    for column in columns:
        plot_over_time(df, column_x="Seconds", column_Y=column, path=os.path.join(gpu_folder, f"{column}_over_time.png"))

    ret.to_csv(os.path.join(gpu_folder, f"resume.csv"))

    return ret, df

def get_macros_energy_consumption(PIPE_PATH: str):
    result_per_datasets = os.listdir(PIPE_PATH)
    for dataset in result_per_datasets:
        models = sorted([d for d in os.listdir(os.path.join(PIPE_PATH, dataset))
                        if os.path.isdir(os.path.join(PIPE_PATH, dataset, d)) and (d[0] in "0123456789") ])

        cpu_macro = {"size": [], "time(s)" : []}
        gpu_macro = {"size": [], "time(s)" : []}

        for size in models:
            cpu_folder  = os.path.join(PIPE_PATH, dataset, size, "cpu/")
            cpu_summary, cpu_df = analyze_cpu_consumption(cpu_folder=cpu_folder)

            columns = cpu_summary.columns.to_list()
            cpu_macro["size"].append(size)
            for col in columns:
                if col not in cpu_macro: cpu_macro[col] = [float(cpu_summary.head(1)[col].values[0])]
                else: cpu_macro[col].append(float(cpu_summary.head(1)[col].values[0]))
            cpu_macro["time(s)"].append(max(cpu_df["Seconds"]))

            gpu_folder  = os.path.join(PIPE_PATH, dataset, size, "gpu/")
            gpu_summary, gpu_df = analyze_gpu_consumption(gpu_folder=gpu_folder)

            columns = gpu_summary.columns.to_list()
            gpu_macro["size"].append(size)
            for col in columns:
                if col not in gpu_macro: gpu_macro[col] = [float(gpu_summary.head(1)[col].values[0])]
                else: gpu_macro[col].append(float(gpu_summary.head(1)[col].values[0]))
            gpu_macro["time(s)"].append(max(gpu_df["Seconds"]))
            
        pd.DataFrame(cpu_macro).to_csv(os.path.join(PIPE_PATH, dataset, "cpu_macro.csv"))
        pd.DataFrame(gpu_macro).to_csv(os.path.join(PIPE_PATH, dataset, "gpu_macro.csv"))

def get_macros_performance(PIPE_PATH: str):
    result_per_datasets = os.listdir(PIPE_PATH)

    for dataset in result_per_datasets:
        models = sorted([d for d in os.listdir(os.path.join(PIPE_PATH, dataset))
                        if os.path.isdir(os.path.join(PIPE_PATH, dataset, d)) and (d[0] in "0123456789") ])
        performance_macro = {"size":[]}
        for size in models:
            result  = pd.read_csv(os.path.join(PIPE_PATH, dataset, size, "result.csv"))
            
            performance_macro["size"].append(size)
            columns = result.columns.to_list()
            columns.remove("Clase")
            for col in columns:
                if col not in performance_macro: performance_macro[col] = [float(result.tail(1)[col].values[0])]
                else: performance_macro[col].append(float(result.tail(1)[col].values[0]))
        
        cpu_macro = pd.read_csv( os.path.join(PIPE_PATH, dataset, "cpu_macro.csv") )
        gpu_macro = pd.read_csv( os.path.join(PIPE_PATH, dataset, "gpu_macro.csv") )

        performance_macro["kwh"] = (((gpu_macro["power.draw"].to_numpy() + cpu_macro["Watts"]) * cpu_macro["time(s)"])/3600)/1000

        pd.DataFrame(performance_macro).to_csv(os.path.join(PIPE_PATH, dataset, "performance_macro.csv"))

def creates_BarCharts(df: pd.DataFrame, store_folder_path: str):
    columns = df.columns.to_list()
    columns.remove('size')
    os.makedirs(store_folder_path, exist_ok=True)
    for column in columns:
        plot_bar_save(df=df, column_x="size", column_Y=column, path=os.path.join(store_folder_path, f"{column}_barchart.png"))

def process_data_and_create_barChars(PIPE_PATH: str):
    result_per_datasets = os.listdir(PIPE_PATH)
    for dataset in result_per_datasets:
        macros =  [archivo for archivo in os.listdir(os.path.join(PIPE_PATH, dataset)) if archivo.endswith('.csv')]
        for macro in macros:
            creates_BarCharts( df= pd.read_csv(os.path.join(PIPE_PATH, dataset, macro)).sort_values(by='size'),
                            store_folder_path= os.path.join(PIPE_PATH, dataset, f"{macro}_plots/") )

def get_best_size(path: str, split: str) -> int:
    file_path = os.path.join(path, split, "performance_macro.csv")
    performance_df = pd.read_csv(file_path)
    max_precision = performance_df["Precision"].max()
    best_size = performance_df.loc[performance_df["Precision"] == max_precision, "size"].iloc[0]
    return best_size

def process_experiment_result(PIPE_PATH: str):
    process_and_save_plots(PIPE_PATH)
    get_macros_energy_consumption(PIPE_PATH)
    get_macros_performance(PIPE_PATH)
    process_data_and_create_barChars(PIPE_PATH)

# Replication Results

def analyze_replication_cpu_jules(cpu_folder: str):
    df = read_csv_files(os.path.join(cpu_folder, "general"))
    df['Moment'] = pd.to_datetime(df['Moment'], format='%Y-%m-%d %H:%M:%S.%f')
    df['Seconds'] = (df['Moment'] - df['Moment'].iloc[0]).dt.total_seconds()
    df['J'] = df['uJ'] / 1e6
    df = df.drop(columns=["filename", "uJ"])

    df['Delta_J'] = df['J'].diff()
    df['Delta_t'] = df['Seconds'].diff()
    max_counter_value = df['J'].max() + 1

    wrap_mask = df['Delta_J'] < 0
    df.loc[wrap_mask, 'Delta_J'] = (max_counter_value - df['J'].shift(1)) + df['J']

    df['Watts'] = df['Delta_J'] / df['Delta_t']
    return (df['Watts']).mean()

def analyze_replication_gpu_jules(gpu_folder=str):
    df = read_csv_files(os.path.join(gpu_folder)).drop(columns=["filename", "name"])
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y/%m/%d %H:%M:%S.%f')
    df['Seconds'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()

    return (df["power.draw"].mean())

def get_num_parameters(model_name, split):
    try:
        if model_name == "proposed":
            size = get_best_size(path = RESULT_PATH, split=split)
            model = rebuild_model(dataset=split, size=size, path=f"{RESULT_PATH}/{split}/{size}/model.pth").to(DEVICE)
            
            return sum(p.numel() for p in model.parameters())
        
        model_name = model_name.lower()
        model = getattr(torchvision.models, model_name)(weights=False)
        return sum(p.numel() for p in model.parameters())
    except AttributeError:
        print(f"Model {model_name} not found in torchvision.models")
        return None
    
def update_consumption_df(REPLICATION_PIPE_PATH, splits, models):
    for split in splits:     
        consumption_df = pd.read_csv(os.path.join(REPLICATION_PIPE_PATH, split, "macro.csv"))
        consumption_df = consumption_df.drop(columns="Unnamed: 0") if "Unnamed: 0" in consumption_df.columns else consumption_df
        consumption_df["model"] = models
    
        cpu_watts = []
        gpu_watts = []
        for model in models: 
            cpu_watts.append(analyze_replication_cpu_jules( os.path.join(REPLICATION_PIPE_PATH, split, "results" ,model, "cpu") ) )
            gpu_watts.append(analyze_replication_gpu_jules( os.path.join(REPLICATION_PIPE_PATH, split, "results" ,model, "gpu") ) )
        
        consumption_df["cpu_w_mean"] = np.array(cpu_watts)
        consumption_df["gpu_w_mean"] = np.array(gpu_watts)
        consumption_df["J_per_img"] = ( (consumption_df["cpu_w_mean"] + consumption_df["gpu_w_mean"]) * consumption_df["duration"] ) / consumption_df["n_images"]
        consumption_df['num_parameters'] = consumption_df['model'].apply(lambda x: get_num_parameters(model_name=x, split=split))
        
        consumption_df.to_csv(os.path.join(REPLICATION_PIPE_PATH, split, "results","consumption.csv"))

# GRADCAM

def generateGradCam(result_path: str, split: str, sample_size: int):
    size = get_best_size(path = result_path, split=split)
    model = rebuild_model(dataset=split, size=size, path=f"{result_path}/{split}/{size}/model.pth").to(DEVICE)
    target_layer = model.cnn_block.conv6
    attention_layer = model.attention_block.batch_norm
    grad_cam = AttentionGradCAM(model, target_layer, attention_layer)
    dataset = build_dataset(dataset=split) # Returns the test

    # Dict to organize the dictionaries
    indices_por_label = defaultdict(list)
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        if hasattr(label, 'argmax'):
            label_int = label.argmax(dim=0).item()
        else:
            label_int = label
        indices_por_label[label_int].append(idx)

    # For each categorie get at least a sample of the sample_size
    for label_val, indices in indices_por_label.items():
        if len(indices) > sample_size:
            sample_indices = random.sample(indices, sample_size)
        else:
            sample_indices = indices

        # Create subplots
        fig, axes = plt.subplots(nrows=len(sample_indices), ncols=2, figsize=(10, 2 * len(sample_indices)))
        fig.suptitle(f"Categoría {(label_val)}", fontsize=16)

        # Just if the there is only one subplot
        if len(sample_indices) == 1:
            axes = np.expand_dims(axes, axis=0)

        for i, idx in enumerate(sample_indices):
            img, _ = dataset[idx]
            input_tensor = img.unsqueeze(0).to(DEVICE)
            cam, predicted_class = grad_cam.generate_cam(input_tensor)

            # Resize the heatmap into the image size
            cam = cv2.resize(cam, (img.shape[-1], img.shape[-2]))

            # Parse the heatmap into rgb
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

            # Parse the image into rgb instead of grey scale
            img_np = img.squeeze().cpu().numpy()
            img_np = cv2.cvtColor(np.uint8(255 * img_np), cv2.COLOR_GRAY2RGB)

            # Overlab the heatmap
            superimposed_img = cv2.addWeighted(heatmap, 0.4, img_np, 0.6, 0)

            # Show the original image
            axes[i, 0].imshow(img_np)
            axes[i, 0].set_title(f"Original (clase {emnist_number_to_text(label_val)})")
            axes[i, 0].axis("off")

            # Show the gradcam 
            axes[i, 1].imshow(superimposed_img)
            axes[i, 1].set_title(f"Grad-CAM (pred {emnist_number_to_text(predicted_class)})")
            axes[i, 1].axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        os.makedirs(f"{result_path}/{split}/BestModel_GradCam/", exist_ok=True)
        plt.savefig(f"{result_path}/{split}/BestModel_GradCam//GradCam_{emnist_number_to_text(label_val)}.png")
        
        plt.close()

# Decision tree

def objective(trial, X, Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    max_depth = trial.suggest_int("max_depth", 1, 20)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
    criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
        random_state=42
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = precision_score(y_test, y_pred, average='macro', zero_division=0)
    return score

def find_best_tree_params_optuna(X, Y, n_trials=50):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X, Y), n_trials=n_trials, n_jobs=THREADS)
    print("Best Macro Precision Score:", study.best_trial.value)
    return study.best_trial.params
   
def get_decision_tree_svg(csv_file:str, result_path: str, target_label: bool):
    type = csv_file.split("/")[-3]
    df = pd.read_csv(csv_file)
    y = df["pred_label"]
    X = df.drop(columns=["true_label", "pred_label"])
    
    best_params = find_best_tree_params_optuna(X, y, n_trials=N_TRIALS)
    clf = DecisionTreeClassifier(**best_params, random_state=SEED)
    clf.fit(X, y)

    dot_file = f"{result_path}/bias_check/{target_label}/" + type + ".dot"
    os.makedirs(f"{result_path}/bias_check/{target_label}/", exist_ok=True)
    
    with open(dot_file, "w") as f:
        export_graphviz(clf,
                        out_file=f,
                        feature_names=X.columns,
                        class_names=[emnist_number_to_text(cls) for cls in clf.classes_],
                        filled=True,
                        rounded=True,
                        special_characters=True)

    svg_file = f"{result_path}/bias_check/{target_label}/" + type + ".svg"
    subprocess.run(["dot", "-Tsvg", dot_file, "-o", svg_file], check=True)
    predictions = clf.predict(X)
    precision = precision_score(y, predictions, average="macro")
    print("La precisión del modelo es: {:.4f}%".format(precision * 100))