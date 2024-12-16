import torch
import numpy as np
from transformers import AutoImageProcessor

from .pipeline import Pipeline

from .libs.multilabel_model import NewHeadDinoV2ForImageClassification, getDynoConfig


class MultiLabelClassifier(Pipeline):
    """Pipeline to identify mulitple class in image"""
    def __init__(self, repo_name, batch_size):
        super(MultiLabelClassifier).__init__()

        self.image_processor = AutoImageProcessor.from_pretrained(repo_name)
        self.config = getDynoConfig(repo_name)
        self.classes_name = list(self.config["label2id"].keys())
        self.batch_size = batch_size
    
    def sigmoid(self, outputs):
        return 1.0 / (1.0 + np.exp(-outputs))

    def cleanup(self):
        """ nothing to release """
        pass

class MultiLabelClassifierCUDA(MultiLabelClassifier):
    """Multilabel classifier with cuda"""

    def __init__(self, repo_name, batch_size):
        super().__init__(repo_name, batch_size)

        self.model = NewHeadDinoV2ForImageClassification.from_pretrained(repo_name)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
    def generator(self):
        # Pass model to gpu
        self.model = self.model.to(self.device)

        data = None
        stop = False
        while self.has_next() and not stop:
            try:
                # Buffer the pipeline stream.
                data = next(self.source)
            except StopIteration:
                stop = True

            if not stop and data:
                # Check if image is not useless

                inputs = self.image_processor(data["frames"], return_tensors="pt")
                inputs = inputs.to(self.device)
                with torch.no_grad():
                    model_outputs = self.model(**inputs)
                
                data["multilabel_scores"] = []
                for logit in model_outputs["logits"]:
                    scores = self.sigmoid(logit.cpu().numpy())
                    data["multilabel_scores"].append([str(a) for a in scores])

                yield data