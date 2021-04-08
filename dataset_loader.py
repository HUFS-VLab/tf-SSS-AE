import os
import glob
import json
import random

class DatasetLoader:
    def __init__(self, target_manifest, dataset_path, is_trained=True):
        self.target_manifest = target_manifest
        self.dataset_path = dataset_path
        self.is_trained = is_trained
        
        self.name = None
        self.trainData_list = []
        self.testData_list = []
        
        self._load()
        
    def _load(self):
        with open(self.target_manifest, 'r') as f:
            data_list = json.load(f)
            n_data = len(data_list)
            
            example = data_list[0]
            item_name = example['item']
            item_type = example['type']
            self.name = item_name
            
            
            getData_path = lambda x : os.path.join(self.dataset_path, x +'.npy')
            if self.is_trained:
                random.seed(123456789)
                random.shuffle(data_list)

                if n_data <= 30:
                    start = 0
                    end = start + int(n_data * 0.75)
                else:
                    start = 0
                    end = 30

                self.trainData_list = [getData_path(data['wav']) for data in data_list[start:end]]
                self.testData_list = [getData_path(data['wav']) for data in data_list[end:]]
            else:
                for data in data_list:
                    wav_path = os.path.join(self.dataset_path, data["wav"]+'.npy')
                    self.testData_list.append(wav_path)
