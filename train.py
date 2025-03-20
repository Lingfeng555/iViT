# Imports
import sys
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np

from Arquitecture.InformationExtractor import InformationExtractor
from utils.Loader import *
from utils.DefaultLogger import DefaultLogger
from utils.evaluator import Evaluator

from experiment_hyperparameters import DEVICE, LEARNING_RATE, EPOCH, BATCH_SIZE, THREADS, RESULT_PATH


def create_dataset(SPLIT):
    if SPLIT=="fashion":
        emnist_train = FashionMNISTDataset(train=True)
        emnist_train_loader = DataLoader(emnist_train, batch_size=BATCH_SIZE, shuffle=True,num_workers=THREADS)


        emnist_test = FashionMNISTDataset(train=False)
        emnist_test_loader = DataLoader(emnist_test, batch_size=BATCH_SIZE, shuffle=True,num_workers=THREADS)
    else:
        emnist_train = EMNISTDataset(train=True, split=SPLIT)
        emnist_train_loader = DataLoader(emnist_train, batch_size=BATCH_SIZE, shuffle=True,num_workers=THREADS)


        emnist_test = EMNISTDataset(train=False, split=SPLIT)
        emnist_test_loader = DataLoader(emnist_test, batch_size=BATCH_SIZE, shuffle=True,num_workers=THREADS)
    return emnist_train_loader, emnist_test_loader, emnist_test

def train(model, optimizer, train_loader, criterion, logger):
    loss_record = []
    for epoch in range(EPOCH):
        model.train()
        total_loss = 0
        num_batches = 0

        for images,label in train_loader:
            images = images.to(DEVICE)
            pred_class = model(images)
            
            loss = criterion(pred_class, label.to(DEVICE))
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            num_batches += 1
        avg_loss = total_loss / num_batches
        loss_record.append(avg_loss)
        logger.info(f"Epoch [{epoch + 1}/{EPOCH}], Loss: {avg_loss:.4f}")
    return loss_record, model
    
def prediction(image, model):
    output = model(image.to("cuda"))
    pred_class = torch.argmax(output, dim=1)
    return pred_class

def plot_and_save(loss, path):
    plt.plot(loss)           
    plt.title("Training loss")  
    plt.xlabel("epoch")     
    plt.ylabel("loss") 
    plt.savefig(path)

def get_predictions(emnist_test_loader, model):
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for images, labels in emnist_test_loader:
            outputs = prediction(images, model)
            true = torch.argmax(labels, dim=1)
            
            # Convertir a listas y agregarlas a los acumuladores
            true_labels.extend(true.cpu().tolist())
            pred_labels.extend(outputs.cpu().tolist())
    return true_labels,pred_labels

def evaluate_model(y_pred: np.array, y_test: np.array, class_names_str: list):
    df_result = pd.DataFrame({"Prediction": y_pred, "GroundTruth": y_test})
    df_result["Prediction"] = df_result["Prediction"].apply(lambda x: class_names_str[x])
    df_result["GroundTruth"] = df_result["GroundTruth"].apply(lambda x: class_names_str[x])
    return Evaluator.evaluate_classification_metrics(df_result)
    
def experiment(SPLIT: str, subinfo_size: int):
    # Create the logger
    trainer_logger = DefaultLogger(path=f"{RESULT_PATH}/{SPLIT}/{subinfo_size}/", name="train_logger")
    trainer_logger.info(f"Entrenando {split} - infosize {subinfo_size}")

    # Create the loader
    emnist_train_loader, emnist_test_loader ,emnist_test = create_dataset(SPLIT)
    _, label = emnist_test[0]

    # Create the model
    trainer_logger.info(f"Creating model")
    model = InformationExtractor(output_len=len(label), subinfo_size=subinfo_size).to(DEVICE)
    trainer_logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Create the loss model
    trainer_logger.info(f"Creating criterion")
    criterion = nn.CrossEntropyLoss(reduction="sum")

    # Create the optimizer
    trainer_logger.info(f"Creating optimizer")
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    # Training process
    trainer_logger.info(f"Training beging")
    try:
        loss, model = train(model, optimizer, emnist_train_loader, criterion, trainer_logger)
    except RuntimeError as e:
        if 'out of memory' in str(e):
            print("CUDA out of memory error caught!")
            torch.cuda.empty_cache() 
        else:
            raise e
    trainer_logger.info(f"Training end")

    # Save the line chart of the loss funcion
    plot_and_save(loss, f"{RESULT_PATH}/{SPLIT}/{subinfo_size}/loss.png")
    trainer_logger.info(f"Loss plot saved")

    # Evaluation of the model
    trainer_logger.info(f"Begin evaluation")

    # Listas para almacenar etiquetas verdaderas y predichas
    true_labels, pred_labels = get_predictions(emnist_test_loader, model)
    trainer_logger.info(f"Evaluation ends")
    
    # Saving results
    evaluate_model(pred_labels, true_labels, range(len(label))).to_csv(f"{RESULT_PATH}/{SPLIT}/{subinfo_size}/result.csv")
    torch.save(model.state_dict(), f"{RESULT_PATH}/{SPLIT}/{subinfo_size}/model.pth")
    trainer_logger.info(f"Results saved")
    
if __name__ == '__main__':
    split = sys.argv[1]
    subinfo_size = int(sys.argv[2])
    experiment(SPLIT=split, subinfo_size=subinfo_size)