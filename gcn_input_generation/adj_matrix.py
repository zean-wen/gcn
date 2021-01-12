import os
import numpy as np
import pickle
import h5py
from tqdm import tqdm


class AdjMatrix:
    def __init__(self,
                 save_dir,
                 save_h5,
                 image_ix_to_id,
                 n_images,
                 adj_matrix,
                 max_n_ocr,
                 image_n_objects,
                 image_n_ocr):
        self.save_h5 = save_h5
        if self.save_h5:
            self.adj_matrix_h5 = h5py.File(os.path.join(save_dir, 'adjacent_matrix.h5'), 'w')
        else:
            self.save_dir = os.path.join(save_dir, 'adjacent_matrix')
            if not os.path.exists(self.save_dir):
                os.mkdir(self.save_dir)
        self.adj_matrix = adj_matrix
        self.image_ix_to_id = image_ix_to_id
        self.n_images = n_images
        self.max_n_ocr = max_n_ocr
        self.image_n_objects = image_n_objects
        self.image_n_ocr = image_n_ocr

    def generate(self):
        if self.save_h5:
            self.generate_h5()
            self.adj_matrix_h5.close()
        else:
            self.generate_pkl()

    def generate_pkl(self):
        for image_index in tqdm(range(self.n_images),
                                unit='image',
                                desc='Adjacent matrix generation'):
            image_id = self.image_ix_to_id[str(image_index)]
            image_adj_matrix = np.array(self.adj_matrix[image_id])
            with open(os.path.join(self.save_dir, '{}.p'.format(image_index)), 'wb') as f:
                pickle.dump(image_adj_matrix, f)

    def generate_h5(self):
        self.adj_matrix_h5.create_dataset("adjacent_matrix",
                                          (self.n_images, 36+self.max_n_ocr, 36+self.max_n_ocr),
                                          dtype='float32')

        for image_index in tqdm(range(self.n_images),
                                unit='image',
                                desc='Adjacent matrix generation'):
            image_id = self.image_ix_to_id[str(image_index)]
            n_object = self.image_n_objects[image_index]
            n_ocr = self.image_n_ocr[image_index]
            image_adj_matrix_pad = np.zeros((36+self.max_n_ocr, 300+self.max_n_ocr),
                                            dtype='float32')
            image_adj_matrix = np.array(self.adj_matrix[image_id])
            obj_obj_adj = image_adj_matrix[:n_object, :n_object]
            obj_ocr_adj = image_adj_matrix[:n_object, n_object:]
            ocr_obj_adj = image_adj_matrix[n_object:, :n_object]
            ocr_ocr_adj = image_adj_matrix[n_object:, n_object:]
            image_adj_matrix_pad[:n_object, :n_object] = obj_obj_adj
            image_adj_matrix_pad[:n_object, 36:36+n_ocr] = obj_ocr_adj
            image_adj_matrix_pad[36:36+n_ocr, :n_object] = ocr_obj_adj
            image_adj_matrix_pad[36:36+n_ocr, 36:36+n_ocr] = ocr_ocr_adj

            self.adj_matrix_h5['adjacent_matrix'][image_index] = image_adj_matrix_pad

