import os

import numpy as np
import pickle
from tqdm import tqdm


class AdjMatrix:
    def __init__(self, save_dir, image_ix_to_id, adj_matrix):
        self.save_dir = os.path.join(save_dir, 'adjacent_matrix')
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        self.adj_matrix = adj_matrix
        self.image_ix_to_id = image_ix_to_id

    def generate(self):
        n_images = len(self.image_ix_to_id)
        for image_index in tqdm(range(n_images),
                                unit='image',
                                desc='Adjacent matrix generation'):
            image_id = self.image_ix_to_id[str(image_index)]
            image_adj_matrix = np.array(self.adj_matrix[image_id])
            with open(os.path.join(self.save_dir, '{}.p'.format(image_index)), 'wb') as f:
                pickle.dump(image_adj_matrix, f)

