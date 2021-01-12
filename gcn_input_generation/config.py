import os
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tiers', type=str, default='train_val_test', help='train, val, test')
    parser.add_argument('--save_dir', type=str, default='../gcn/data', help='path to save result')
    parser.add_argument('--data_root', type=str, default='./data', help='path where the input data saved')
    parser.add_argument('save_h5', type=bool, action='store_true', help='save the result as h5')
    return parser.parse_args()


class WordEmbeddingConfig:
    def __init__(self, data_root):
        self.glove_dictionary_file = os.path.join(data_root, 'word_embedding', 'glove_dictionary.json')
        self.glove_word_matrix_file = os.path.join(data_root, 'word_embedding', 'glove6b_init_300d.npy')
        self.fasttext_dictionary_file = os.path.join(data_root, 'word_embedding', 'fasttext_dictionary.json')
        self.fasttext_word_matrix_file = os.path.join(data_root, 'word_embedding', 'fasttext_init_300d.npy')


class Config:
    def __init__(self, args):
        self.tiers = args.tiers
        self.data_root = args.data_root
        self.save_dir = args.save_dir
        self.save_h5 = args.save_h5
        self.word_emb_config: WordEmbeddingConfig = WordEmbeddingConfig(self.data_root)
