import numpy as np
import pandas as pd

import random
import os
import sys
import argparse

from time import time
from tqdm import tqdm

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from tensorflow.keras.optimizers.legacy import Adagrad, Adam, SGD, RMSprop

from Dataset import Dataset
import NeuMF
import GMF, MLP

CFG = {'SEED' : 42}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

seed_everything(CFG['SEED'])

def parse_args():
    parser = argparse.ArgumentParser(description="Run NeuMF inferene.")
    parser.add_argument('--path', nargs='?', default='../Dataset/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='Dacon',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--num_factors', type=int, default=8,
                        help='Embedding size of MF model.')
    parser.add_argument('--layers', nargs='?', default='[64,32,16,8]',
                        help="MLP layers. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--reg_mf', type=float, default=0,
                        help='Regularization for MF embeddings.')                    
    parser.add_argument('--reg_layers', nargs='?', default='[0,0,0,0]',
                        help="Regularization for each MLP layer. reg_layers[0] is the regularization for embeddings.")
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='sgd',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    parser.add_argument('--mf_pretrain', nargs='?', default='/Users/seunghoonchoi/Downloads/Recommend/NeuMF_MAWU/Pretrain/Dacon_GMF_8_1699772783.h5',
                        help='Specify the pretrain model file for MF part. If empty, no pretrain will be used')
    parser.add_argument('--mlp_pretrain', nargs='?', default='/Users/seunghoonchoi/Downloads/Recommend/NeuMF_MAWU/Pretrain/Dacon_MLP_[64,32,16,8]_1699776065.h5',
                        help='Specify the pretrain model file for MLP part. If empty, no pretrain will be used')
    parser.add_argument('--NeuMF_pretrain', nargs='?', default='/Users/seunghoonchoi/Downloads/Recommend/NeuMF_MAWU/Pretrain/Dacon_NeuMF_8_[64,32,16,8]_1699780608.h5',
                        help='Specify the pretrain model file for NeuMF part. If empty, no pretrain will be used')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    num_epochs = args.epochs
    batch_size = args.batch_size
    mf_dim = args.num_factors
    layers = eval(args.layers)
    reg_mf = args.reg_mf
    reg_layers = eval(args.reg_layers)
    num_negatives = args.num_neg
    learning_rate = args.lr
    learner = args.learner
    verbose = args.verbose
    
    mf_pretrain = args.mf_pretrain
    mlp_pretrain = args.mlp_pretrain
    NeuMF_pretrain = args.NeuMF_pretrain


    t1 = time()
    dataset = Dataset(args.path + args.dataset, inference=True)
    final = dataset.finalMatrix
    num_users, num_items = final.shape
    print("Load data done [%.1f s]. #user=%d, #item=%d, #final=%d" 
          %(time()-t1, num_users, num_items, final.nnz))
    
    # Build model
    model = NeuMF.get_model(num_users, num_items, mf_dim, layers, reg_layers, reg_mf)
    if learner.lower() == "adagrad": 
        model.compile(optimizer=Adagrad(learning_rate=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=RMSprop(learning_rate=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "adam":
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy')
    else:
        model.compile(optimizer=SGD(learning_rate=learning_rate), loss='binary_crossentropy')

    model.load_weights('/Users/seunghoonchoi/Downloads/Recommend/NeuMF_MAWU/Pretrain/Dacon_NeuMF_8_[64,32,16,8]_1699780608.h5')

    user_ids = np.array(range(num_users)) # 모든 사용자 ID
    item_ids = np.array(range(num_items)) # 모든 아이템 ID

    # 추천 결과를 저장할 리스트
    recommendations = []

    interacted_items = {}
    for (user_id, item_id), _ in final.items():
        if user_id not in interacted_items:
            interacted_items[user_id] = set()
        interacted_items[user_id].add(item_id)

    # 사용자별로 상위 5개 아이템 추천
    for user_id in user_ids:
        test_items = [item_id for item_id in item_ids if item_id not in interacted_items.get(user_id, set())]
        
        # 배치로 예측 점수 계산
        user_array = np.array([user_id] * len(test_items))
        item_array = np.array(test_items)
        preds = model.predict([user_array, item_array])

        # 예측 점수에 따라 상위 5개 아이템을 추천
        top_5_items = sorted(zip(test_items, preds.flatten()), key=lambda x: x[1], reverse=True)[:5]
        
        # 결과 리스트에 추가
        for item, _ in top_5_items:
            recommendations.append(['U{:05d}'.format(user_id + 1), 'R{:05d}'.format(item + 1)])

    # 데이터프레임 생성
    df_recommendations = pd.DataFrame(recommendations, columns=['resume_seq', 'recruitment_seq'])

    # 데이터프레임 저장
    df_recommendations.to_csv('recommendations.csv', index=False)

