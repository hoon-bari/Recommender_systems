import argparse
import pickle

import numpy as np
import os
import torch
import random
import dgl
from model import train

import faiss

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    dgl.seed(seed)

seed_everything(42) # Seed 고정

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_file_path', type=str)
    parser.add_argument('--random-walk-length', type=int, default=2)
    parser.add_argument('--random-walk-restart-prob', type=float, default=0.5)
    parser.add_argument('--num-random-walks', type=int, default=10)
    parser.add_argument('--num-neighbors', type=int, default=5)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--gat-num-heads', type=int, default=3)
    parser.add_argument('--hidden-dims', type=int, default=90) # 512
    parser.add_argument('--batch-size', type=int, default=128) # 256
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--num-epochs', type=int, default=4) # 4
    parser.add_argument('--batches-per-epoch', type=int, default=5000) # 5000
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('-k', type=int, default=10)
    args = parser.parse_args()

    with open(args.graph_file_path, 'rb') as f:
        dataset = pickle.load(f)
    model, h_recruitment = train(dataset, args)

    torch.save(model.state_dict(), 'MultiSAGE_weights.pth')
    np.savez("h_recruitment", recruitment_vectors=h_recruitment.numpy())

    # resume_vector = np.load('/Users/seunghoonchoi/Downloads/Dacon_recommend_system/src/h_resume.npz')
    # recruitment_vector = np.load('/Users/seunghoonchoi/Downloads/Dacon_recommend_system/src/h_recruitment.npz')

    # h_resume = resume_vector['resume_vectors']
    # h_recruitment = recruitment_vector['recruitment_vectors']

    # db_vector = h_resume
    # query_vector = h_recruitment

    # index = faiss.IndexFlatL2(100)

    # index.add(db_vector)

    # D, I = index.search(query_vector, 5)

    # print(f'Distance : {D[:5]}')
    # print(f'Index : {I[:5]}')