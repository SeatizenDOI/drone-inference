import json
import torch.nn as nn
from pathlib import Path
from huggingface_hub import snapshot_download
from transformers import Dinov2Config, Dinov2ForImageClassification


PATH_TO_MULTILABEL_DIRECTORY = "models/multilabel"

class NewHeadDinoV2ForImageClassification(Dinov2ForImageClassification):
    def __init__(self, config: Dinov2Config) -> None:
        super().__init__(config)

        # Classifier head
        self.classifier = self.create_head(config.hidden_size * 2, config.num_labels)
    
    # CREATE CUSTOM MODEL
    def create_head(self, num_features , number_classes ,dropout_prob=0.5 ,activation_func = nn.ReLU):
        features_lst = [num_features , num_features//2 , num_features//4]
        layers = []
        for in_f ,out_f in zip(features_lst[:-1] , features_lst[1:]):
            layers.append(nn.Linear(in_f , out_f))
            layers.append(activation_func())
            layers.append(nn.BatchNorm1d(out_f))
            if dropout_prob != 0 : layers.append(nn.Dropout(dropout_prob))
        layers.append(nn.Linear(features_lst[-1] , number_classes))
        return nn.Sequential(*layers)


def getDynoConfig(repo_name):
    repo_path = Path(Path.cwd(), PATH_TO_MULTILABEL_DIRECTORY, repo_name)
    if not Path.exists(repo_path):
        snapshot_download(repo_id=repo_name, local_dir=Path(Path.cwd(), PATH_TO_MULTILABEL_DIRECTORY, repo_name))

    config = None
    with open(Path(repo_path, "config.json")) as f:
        config = json.load(f)
    
    return config

