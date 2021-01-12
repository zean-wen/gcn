import os
import pickle
import numpy as np
from tqdm import tqdm


class Target:
    def __init__(self, save_dir, n_images, image_n_objects, image_n_ocr, ocr):
        self.save_dir = os.path.join(save_dir, 'targets')
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        self.n_images = n_images
        self.image_n_ocr = image_n_ocr
        self.image_n_objects = image_n_objects
        self.ocr = ocr

    def generate(self):
        for image_index in tqdm(range(self.n_images),
                                unit='image',
                                desc='target generation'):
            n_ocr = self.image_n_ocr[image_index]
            n_objects = self.image_n_objects[image_index]
            n_nodes = n_objects + n_ocr
            image_target = np.zeros((n_nodes, 2), dtype='float32')
            image_target[:n_objects, 0] = 1
            image_target[n_objects:, 1] = 1
            with open(os.path.join(self.save_dir, '{}.p'.format(image_index)), 'wb') as f:
                pickle.dump(image_target, f)
