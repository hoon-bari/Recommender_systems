import random
import os
import argparse

import numpy as np
import pandas as pd

from multiprocessing import Pool
import itertools

import config

from tqdm.auto import tqdm
from collections import defaultdict

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

import warnings
warnings.filterwarnings(action='ignore')

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(config.SEED) # Seed 고정

def get_cos_similarity_of_user(resume_data):
    # 'resume_seq' 컬럼 제외
    resume_features = resume_data.drop(columns=['resume_seq'])

    # 코사인 유사도 계산
    cosine_sim = cosine_similarity(resume_features)

    # 코사인 유사도 결과를 DataFrame으로 변환
    cosine_sim_df = pd.DataFrame(cosine_sim, 
                                index=resume_data['resume_seq'], 
                                columns=resume_data['resume_seq'])
    
    return cosine_sim_df
    
def generate_negative_samples_of_user(user_info, negative_candidates, resume_to_recruitments):    
    resume_seq = user_info
    applied_positions = resume_to_recruitments[resume_seq]
    available_negatives = list(negative_candidates - applied_positions)

    # 지원한 공고의 수와 사용할 수 있는 negatives의 수를 비교
    num_neg_samples = min(len(applied_positions), len(available_negatives))

    # 필요한 수만큼 랜덤으로 negatives를 선택하거나, 가능한 모든 negatives를 사용
    selected_negatives = random.sample(available_negatives, num_neg_samples * 2) if num_neg_samples == len(applied_positions) else available_negatives

    return resume_seq, selected_negatives


def prepare_negative_samples(cosine_sim_df, apply_train, threshold=0.3):

    # 유사도가 임계값 이하인 쌍의 인덱스
    low_similarity_pairs = np.column_stack(np.where(cosine_sim_df.values <= threshold))

    # 각 resume_seq에 대해 지원한 공고의 집합
    resume_to_recruitments = apply_train.groupby('resume_seq')['recruitment_seq'].apply(lambda x: set(x)).to_dict()
    
    # 각 사용자에 대한 negative 후보들의 집합
    negative_candidates_dict = {}

    for i, j in low_similarity_pairs:

        resume_seq_i, resume_seq_j = cosine_sim_df.index[i], cosine_sim_df.index[j]
        
        # 이미 처리된 이력서를 체크하기 위한 집합을 사용
        processed_resumes = set()
        
        # 첫 번째 이력서에 대한 negative 후보 업데이트
        if resume_seq_i not in processed_resumes:
            applied_positions_j = resume_to_recruitments[resume_seq_j]
            negative_candidates_dict.setdefault(resume_seq_i, set()).update(applied_positions_j)
            processed_resumes.add(resume_seq_i)
        
        # 두 번째 이력서에 대한 negative 후보 업데이트
        if resume_seq_j not in processed_resumes:
            applied_positions_i = resume_to_recruitments[resume_seq_i]
            negative_candidates_dict.setdefault(resume_seq_j, set()).update(applied_positions_i)
            processed_resumes.add(resume_seq_j)

    return negative_candidates_dict, resume_to_recruitments


def generate_negative_samples(negative_candidates_dict, resume_to_recruitments):
    results = []

    for resume_seq, candidates in negative_candidates_dict.items():
        result = generate_negative_samples_of_user(resume_seq, candidates, resume_to_recruitments)
        results.append(result)

    return results


def create_negative_samples_df(results):
    negative_samples = list(results)

    # 결과를 DataFrame으로 변환
    negative_samples_df = pd.DataFrame(negative_samples, columns=['resume_seq', 'recruitment_seq_negatives'])

    # 결과를 확인
    negative_samples_df.sort_values(by=['resume_seq'], inplace=True)
    negative_samples_df.reset_index(drop=True, inplace=True)

    return negative_samples_df