from .pipeline import Pipeline


class  MultilabelPredictions(Pipeline):
    """Pipeline task to save Multilabel predictions"""

    def __init__(self, classes):
        self.filename_scores, self.csv_connector_scores = None, None
        self.classes = classes
        super(MultilabelPredictions, self).__init__()
    
    def setup(self, filename_scores):
        self.filename_scores = filename_scores
        self.csv_connector_scores = open(self.filename_scores, "w")

    def generator(self):
        """ Write in csv file"""
        classe_to_write = ",".join(self.classes)
        self.csv_connector_scores.write(f"FileName,{classe_to_write},GPSLatitude,GPSLongitude\n")


        data = None
        stop = False
        while self.has_next() and not stop:
            try:
                # Buffer the pipeline stream.
                data = next(self.source)
            except StopIteration:
                stop = True

            if not stop and data:
                if "multilabel_scores" in data:
                    for i, frame_path in enumerate(data["frame_paths"]):
                        lon, lat = data['frames_position'][i]
                        self.csv_connector_scores.write(f"{frame_path},{','.join(data['multilabel_scores'][i])},{lat},{lon}\n")
            
                yield data
    
    def cleanup(self):
        self.csv_connector_scores.close()