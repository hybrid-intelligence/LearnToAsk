import os
import json
import argparse
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from builder.vocab import Vocabulary
from builder.model import Builder
from builder.dataloader_with_glove import BuilderDataset, RawInputs
from builder.utils_builder import evaluate_metrics
from builder.decoding import generate_action_pred_seq
from utils import *
from train import MineCraft

def main(args, config):
    testdataset = BuilderDataset(args, split='test', encoder_vocab=None)
    test_items = testdataset.items
    test_dataset = MineCraft(test_items)
    testdataloader = DataLoader(test_dataset, batch_size=5, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using {}'.format(device))
    with open(args.encoder_vocab_path, 'rb') as f:
        encoder_vocab = pickle.load(f)
    model = Builder(config, vocabulary=encoder_vocab).to(device)
    model.load_state_dict(torch.load(os.path.join(args.saved_models_path, "model.pt"), map_location=torch.device('cpu') ))
    model.eval()

    generated_seqs, to_print = generate_action_pred_seq(model=model,
        test_item_batches=test_items,
        beam_size=5,
        max_length=30,
        testdataset=testdataset)
    print(to_print)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--saved_models_path', type=str, default='./default_path',
                        help='path for saving trained models')
    parser.add_argument('--encoder_vocab_path', type=str,
                        default='vocabulary/glove.42B.300d-lower-speaker-oov_as_unk-all_splits/vocab.pkl')
    # Args for dataset
    parser.add_argument('--json_data_dir', type=str, default="./builder_data/data_maxlength100")
    parser.add_argument('--load_items', default=True)

    args = parser.parse_args()
    with open(os.path.join(args.saved_models_path, "config.json"), "r") as fp:
        config = json.load(fp)
    main(args, config)