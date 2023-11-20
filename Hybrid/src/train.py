import numpy as np
import pandas as pd

import random
import os

from tqdm.auto import tqdm
from collections import defaultdict
from sklearn.decomposition import TruncatedSVD, NMF, SparsePCA
from sklearn.metrics.pairwise import cosine_similarity

import implicit
import numpy as np
from scipy.sparse import csr_matrix

CFG = {'SEED' : 42}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(CFG['SEED']) # Seed 고정

def load_data(directory, file_names):
    data = {}
    for file_name in file_names:
        data[file_name] = pd.read_csv(os.path.join(directory, f'{file_name}.csv'))
    return data

def train_test_split(apply_train):
    # 학습, 검증 데이터 만들기
    train, val = [], []
    apply_train_groupby = apply_train.groupby('resume_seq')['recruitment_seq'].apply(list)
    for uid, iids in zip(apply_train_groupby.index.tolist(), apply_train_groupby.values.tolist()):
        for iid in iids[:-1]:
            train.append([uid,iid])
        val.append([uid, iids[-1]])

    train = pd.DataFrame(train, columns=['resume_seq', 'recruitment_seq'])
    val = pd.DataFrame(val, columns=['resume_seq', 'recruitment_seq'])

    return train, val

# 코사인 유사도 계산 함수
# def calculate_cosine_similarity(data, id_column):
#     # id_column을 제외한 특성을 사용하여 코사인 유사도 계산
#     features = data.drop(columns=[id_column])
#     cosine_sim = cosine_similarity(features)

#     # 코사인 유사도 DataFrame 생성
#     cosine_sim_df = pd.DataFrame(cosine_sim, index=data[id_column], columns=data[id_column])
    
#     # 대각선(자기 자신과의 유사도)을 0으로 설정
#     np.fill_diagonal(cosine_sim_df.values, 0)
    
#     return cosine_sim_df

# # 1. 유저 유사도를 보고, 가장 비슷한 유저가 지원한 공고 중 원래 유저가 지원한 공고와 유사도를 비교해서 더 비슷한 공고 추천
# def recommend_based_on_similar_user_recruitment(apply_data, resume_cosine_sim, recruitment_cosine_sim):
#     # 결과를 저장할 딕셔너리
#     recommendations = {}

#     # 각 이력서별로 루프를 돌면서
#     for resume_id in resume_cosine_sim.index:
#         # 가장 유사한 이력서 두 개를 찾음
#         similar_resumes = resume_cosine_sim.loc[resume_id].nlargest(3).index.tolist()[1:]

#         # 원래 이력서가 지원한 공고를 찾음
#         original_recruitments = apply_data[apply_data['resume_seq'] == resume_id]['recruitment_seq'].tolist()

#         # 유사한 이력서가 지원한 공고를 찾고, 원래 이력서가 지원한 공고를 제외
#         similar_recruitments = set()
#         for similar_resume_id in similar_resumes:
#             similar_recruitments.update(
#                 r for r in apply_data[apply_data['resume_seq'] == similar_resume_id]['recruitment_seq'].tolist() 
#                 if r not in original_recruitments
#             )

#         # 각 유사한 공고에 대해 유사도를 계산
#         max_similarity = 0
#         most_similar_recruitment = None
#         for target_recruitment in similar_recruitments:
#             for original_recruitment in original_recruitments:
#                 # 원래 이력서가 지원한 공고와 유사한 이력서가 지원한 공고의 유사도를 비교
#                 similarity = recruitment_cosine_sim.loc[original_recruitment, target_recruitment]
#                 if similarity > max_similarity:
#                     max_similarity = similarity
#                     most_similar_recruitment = target_recruitment
        
#         # 결과 저장
#         if most_similar_recruitment:
#             recommendations[resume_id] = most_similar_recruitment

#     return recommendations

# # 2. 원래 유저가 지원한 공고와 모든 공고의 유사도를 계산 후 가장 비슷한 공고 추천(이전에 추천된 것 제외)
# def recommend_based_on_current_user_recruitment(apply_data, recruitment_cosine_sim, previous_recommendations):
#     recommendations = {}

#     for resume_id in apply_data['resume_seq'].unique():
#         user_excluded_recruitments = previous_recommendations[resume_id]
#         applied_recruitments = apply_data[apply_data['resume_seq'] == resume_id]['recruitment_seq'].tolist()

#         similarity_matrix = recruitment_cosine_sim.loc[applied_recruitments].copy()
#         similarity_matrix.drop(columns=applied_recruitments + user_excluded_recruitments, inplace=True, errors='ignore')

#         if not similarity_matrix.empty:
#             most_similar_recruitment = similarity_matrix.max().idxmax()
#             recommendations[resume_id] = [most_similar_recruitment]

#     return recommendations

# 3. Collaborative Filtering

def make_matrix(data):
    user_item_matrix = data.groupby(['resume_seq', 'recruitment_seq']).size().unstack(fill_value=0)

    return user_item_matrix

def calculate_scores(user_item_matrix):
    user_similarity = cosine_similarity(user_item_matrix)
    item_similarity = cosine_similarity(user_item_matrix.T)

    user_predicted_scores = user_similarity.dot(user_item_matrix)
    item_predicted_scores = user_item_matrix.dot(item_similarity)

    return user_predicted_scores, item_predicted_scores


def collaborative_filtering_with_margin(user_item_matrix, user_predicted_scores, item_predicted_scores, k=3, margin_1=0.98, margin_2=1):
    recommendations = {}

    for idx, user in enumerate(user_item_matrix.index):

        applied_jobs = set(user_item_matrix.loc[user][user_item_matrix.loc[user] == 1].index)
        sorted_job_indices = (item_predicted_scores.loc[user].values * margin_1 + user_predicted_scores[idx] * margin_2).argsort()[::-1]

        recommended_jobs = [job for job in user_item_matrix.columns[sorted_job_indices] if job not in applied_jobs ][:k]

        if recommended_jobs:
            recommendations[user] = recommended_jobs

    return recommendations

# # 4,5. 인기도를 반영한 CF, 그리고 인기도의 역수를 반영한 CF
# def calculate_popularity_scores(apply_data):
#     # 각 공고별 지원 횟수를 계산
#     recruitment_popularity = apply_data['recruitment_seq'].value_counts()

#     # 최대 지원 횟수
#     max_popularity = recruitment_popularity.max()

#     # 인기도 점수 계산 (각 공고의 지원 횟수를 최대 지원 횟수로 나눔)
#     popularity_scores = recruitment_popularity / max_popularity

#     # 인기도 역수 계산
#     inverse_popularity_scores = 1 / (popularity_scores)

#     # 역수 정규화
#     max_inverse_popularity = inverse_popularity_scores.max()
#     inverse_popularity_scores = inverse_popularity_scores / max_inverse_popularity


#     return popularity_scores, inverse_popularity_scores

# def get_popularity_matrix(matrix, scores):
#     popularity_reflected_matrix = matrix.mul(scores, axis=1).fillna(0)

#     return popularity_reflected_matrix

# 잠재요인 협업 필터링

# def initiate_svd(user_item_matrix):
#     n_components = 200
#     svd = TruncatedSVD(n_components=n_components)
#     user_factors = svd.fit_transform(user_item_matrix)
#     item_factors = svd.components_

#     item_similarity_scores = np.dot(item_factors.T, item_factors)

#     # 사용자의 아이템 점수 계산 (TruncatedSVD 잠재 요인 이용)
#     user_item_scores = np.dot(user_factors, item_factors)

#     return item_similarity_scores, user_item_scores

# def collaborative_filtering_with_svd(user_item_matrix, user_item_scores, previous_recommendations, k=1):
#     recommendations = {}
#     item_indices = {item: i for i, item in enumerate(user_item_matrix.columns)}

#     for idx, user in enumerate(user_item_matrix.index):
#         user_excluded_recruitments = previous_recommendations.get(user, [])
#         applied_jobs = set(user_item_matrix.loc[user][user_item_matrix.loc[user] == 1].index)

#         # 사용자 별 아이템 점수를 최종 점수로 사용
#         combined_scores = user_item_scores[idx]

#         # 이미 상호작용한 아이템 및 이전에 추천된 아이템의 인덱스를 가져옴
#         exclude_indices = [item_indices[item] for item in applied_jobs.union(user_excluded_recruitments) if item in item_indices]

#         # 해당 인덱스의 점수를 -무한대로 설정하여 제외
#         combined_scores[exclude_indices] = -np.inf

#         # 상위 k개 아이템 추천
#         recommended_jobs = np.argsort(combined_scores)[::-1][:k]
#         recommendations[user] = recommended_jobs

#     return recommendations

# lmf 모델 구성 및 훈련
def train_lmf_model(user_item_matrix, factors=20, regularization=0.8, iterations=50, random_state=CFG['SEED']):
    # 사용자-아이템 행렬을 희소 행렬로 변환
    sparse_user_item = csr_matrix(user_item_matrix)

    # lmf 모델 초기화
    model = implicit.cpu.lmf.LogisticMatrixFactorization(factors=factors, 
                                                 regularization=regularization,
                                                 iterations=iterations)
    

    model.fit(sparse_user_item)

    return model

def collaborative_filtering_with_lmf(model, user_item_matrix, previous_recommendations, n_items=2):
    recommendations = {}
    sparse_user_item = csr_matrix(user_item_matrix)

    for user_id in user_item_matrix.index:
        user_idx = user_item_matrix.index.get_loc(user_id)

        user_excluded_recruitments = previous_recommendations.get(user_id, [])
        applied_jobs = set(user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] == 1].index)

        all_items = model.recommend(user_idx, sparse_user_item[user_idx])

        recommended_items = []
        for item_idx, score in zip(*all_items):
            item = user_item_matrix.columns[item_idx]
            if item not in user_excluded_recruitments and item not in applied_jobs:
                recommended_items.append(item)
                if len(recommended_items) >= n_items:
                    break

        if recommended_items:
            recommendations[user_id] = recommended_items

    return recommendations

# 추천 업데이트 함수
def update_recommendations(current_recommendations, new_recommendations):
    for resume_id, new_rec in new_recommendations.items():
        if resume_id in current_recommendations:
            # 현재 추천에 새로운 추천 추가
            for r in new_rec:
                current_recommendations[resume_id].append(r)
        else:
            # 새로운 추천을 리스트로 저장
            current_recommendations[resume_id] = new_rec
    return current_recommendations