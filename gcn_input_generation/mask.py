import h5py
import os
from tqdm import tqdm
import numpy as np


class Mask:
    def __init__(self,
                 save_dir,
                 save_h5,
                 n_images,
                 max_n_ocr,
                 image_n_objects,
                 image_n_ocr):
        self.save_h5 = save_h5
        if self.save_h5:
            self.mask_h5 = h5py.File(os.path.join(save_dir, 'mask.h5'), 'w')
        self.n_images = n_images
        self.max_n_ocr = max_n_ocr
        self.image_n_objects = image_n_objects
        self.image_n_ocr = image_n_ocr

    def generate(self):
        if self.save_h5:
            self.generate_h5()
            self.mask_h5.close()
        else:
            pass

    def generate_h5(self):
        self.mask_h5.create_dataset("masks",
                                    (self.n_images, 36+self.max_n_ocr),
                                    dtype='float32')
        for image_index in tqdm(range(self.n_images),
                                unit='image',
                                desc='Mask generation'):
            image_mask = np.zeros(36+self.max_n_ocr, dtype='int8')
            n_object = self.image_n_objects[image_index]
            n_ocr = self.image_n_ocr[image_index]

            object_index = np.array(range(n_object))
            ocr_index = np.array(range(n_ocr)) + 36

            image_mask[object_index] = 1
            image_mask[ocr_index] = 1
            self.mask_h5['masks'][image_index] = image_mask
