import os
import numpy as np
import json
class DFDC_get_batch_data:
    def __init__(self):
        self.top_path = '/DATA7_DB7/data/yydong/DFDC/full_vision/images_crop/'
        self.video_path =  os.listdir(self.top_path)
        a = open('/DB/rhome/chaoqinhuang/deepfake/full.json', 'r')
        self.data = json.load(a)

    def get_batch(self, batch_size = 20):
        files = []
        while len(files) < batch_size:
            index = np.random.randint(len(self.data))
            if self.data[index]['label'] == 0:
                files.append(self.data[index]['image'])

        return files

