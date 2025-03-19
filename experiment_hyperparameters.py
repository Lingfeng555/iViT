import torch

SEED = 555
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LEARNING_RATE = 0.000125
EPOCH = 1#1000
BATCH_SIZE = 4000#30000
THREADS = 8#32
DEFAULT_LOG_FILE = "log"
RESULT_PATH = "experiment_pipeline"
TRAIN_SCRIPT = "train.py"

SPLITS = ["digits", "fashion", "balanced"]
SIZES = [_ for _ in range(1, 11)]

REPLICATION_BATCH_SIZE = 15#60
REPLICATION_PATH = "replication_pipeline"
REPLICATION_DURATION = 2#3600
REPLICATION_MODELS = ["proposed", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
          "mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large",
          "inception_v3"
          ]
for i in range(8):
    REPLICATION_MODELS.append("efficientnet_b"+str(i))
    
REPLICATION_SCRIPT = "replication.py"

N_TRIALS = 1 #2000