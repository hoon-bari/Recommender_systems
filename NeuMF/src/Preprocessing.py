import numpy as np
import pandas as pd

import random
import os

CFG = {'SEED' : 42}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

# train, val 데이터 분할
def train_val_test_split(data):
    train, val, test = [], [], []
    data_groupby = data.groupby('resume_seq')['recruitment_seq'].apply(list)
    for uid, iids in zip(data_groupby.index.tolist(), data_groupby.values.tolist()):
        test.extend([[uid, iid] for iid in iids])  # 모든 상호작용을 test에 추가
        for iid in iids[:-1]:
            train.append([uid, iid])  # 마지막 상호작용을 제외하고 train에 추가
        val.append([uid, iids[-1]])  # 마지막 상호작용을 val에 추가
    return train, val, test

# 파일 저장 함수
def save_to_file(data, filename):
    with open(filename, 'w') as f:
        for row in data:
            user, item = row[0], row[1]
            f.write(str(user) + '\t' + str(item) + '\t' + '1' + '\t' + '0' + '\n')

def create_negative_samples_file(val_data, num_negatives, total_recruitments, output_file_path):
    with open(output_file_path, 'w') as file:
        for interaction in val_data:
            negatives = []
            while len(negatives) < num_negatives:
                neg_item_id = random.randint(0, total_recruitments)
                if neg_item_id != interaction[1]:
                    negatives.append(neg_item_id)
            line = f"({interaction[0]},{interaction[1]})\t" + '\t'.join(map(str, negatives)) + '\n'
            file.write(line)


if __name__ == '__main__':
    file_path = '../Data/apply_train.csv'
    data = pd.read_csv(file_path)

    # 문자열에서 정수 부분 추출
    data['resume_seq'] = data['resume_seq'].str.extract('(\d+)').astype(int) - 1
    data['recruitment_seq'] = data['recruitment_seq'].str.extract('(\d+)').astype(int) - 1

    train_data, val_data, test_data = train_val_test_split(data)
    save_to_file(train_data, '../Dataset/Dacon.train.rating')
    save_to_file(val_data, '../Dataset/Dacon.test.rating')
    save_to_file(test_data, '../Dataset/Dacon.final.rating')

    # 네거티브 샘플 수
    num_negatives = 100

    total_recruitments = data['recruitment_seq'].max()
    output_file_path = '../Dataset/Dacon.test.negative'

    # Negative 샘플 파일 생성
    create_negative_samples_file(val_data, num_negatives, total_recruitments, output_file_path)