import argparse
import time
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

from utils.Loader import EMNISTDataset, FashionMNISTDataset
from experiment_hyperparameters import REPLICATION_BATCH_SIZE, RESULT_PATH, DEVICE
from experiment_result_processing import rebuild_model, get_best_size

import sys
import os
def create_other_model(model_name: str):
    model_name = model_name.lower()
    return getattr(torchvision.models, model_name)(pretrained=True)

def build_dataset_with_transform(dataset: str, transform):
    if dataset == "fashion": return FashionMNISTDataset(download=True, train=False, transform=transform)
    else: return EMNISTDataset(split=dataset, train=False, download=True, transform=transform)
    
def main(args):
    # Create the transfrorm
    if args.model != "proposed":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    dataset = build_dataset_with_transform(args.split, transform)
    dataloader = DataLoader(dataset, batch_size=REPLICATION_BATCH_SIZE, shuffle=False)
    # Create the model
    if args.model != "proposed": 
         model = create_other_model(args.model)
    else:
        best_size = get_best_size(path=f"{RESULT_PATH}", split=args.split)
        model = rebuild_model(dataset=args.split, size=best_size, path=f"{RESULT_PATH}/{args.split}/{best_size}/model.pth")

    output_len = 47 if args.split == "balanced" else 10
    if args.model != "proposed":
        if hasattr(model, 'fc'):
            model.fc = torch.nn.Linear(model.fc.in_features, output_len)
        elif hasattr(model, 'classifier'):
            if isinstance(model.classifier, torch.nn.Sequential):
                in_features = model.classifier[-1].in_features
                model.classifier[-1] = torch.nn.Linear(in_features, output_len)
            else:
                model.classifier = torch.nn.Linear(model.classifier.in_features, output_len)
    # Start the experiment
    model.to(DEVICE)
    model.eval()

    start_time = time.time()
    processed_images_count = 0
    limit_seconds = args.seconds

    with torch.no_grad():
        while True:
            for images, _ in dataloader:
                if time.time() - start_time >= limit_seconds:
                    break
                
                images = images.to(DEVICE)
                outputs = model(images)               
                processed_images_count += images.size(0)
            if time.time() - start_time >= limit_seconds:
                break
    print(processed_images_count)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Replication of other models to measure the energy consumption")
    parser.add_argument('--seconds', type=int, default=5, help='Duration of the replication')
    parser.add_argument('--model', type=str, default='resnet18', help='Name of the model')
    parser.add_argument('--split', type=str, default='fashion', help='Name of the dataset')
    args = parser.parse_args()
    main(args)