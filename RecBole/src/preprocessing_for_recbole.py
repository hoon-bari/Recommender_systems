import pandas as pd
import numpy as np
import os
import argparse
import pickle
from datetime import datetime

import torch

import warnings
warnings.filterwarnings(action='ignore')

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

from gensim.models import FastText

def load_data(directory, file_names):
    data = {}
    for file_name in file_names:
        data[file_name] = pd.read_csv(os.path.join(directory, f'{file_name}.csv'))
    return data

def preprocess_resume(resume, resume_edu, resume_cert, resume_lang):
    # 자격증 데이터 정리: 누락된 값 제거, 집계 후 병합
    resume_cert = resume_cert.dropna(subset=['certificate_contents'])
    cert_count = resume_cert.groupby('resume_seq')['certificate_contents'].size().reset_index(name='certificate_count')
    resume = resume.merge(cert_count, on='resume_seq', how='left')
    resume['certificate_count'] = resume['certificate_count'].fillna(0).astype(int)

    # 언어 능력 데이터 정리: 원-핫 인코딩 후 병합
    lang_encoded = pd.get_dummies(resume_lang, columns=['language'], prefix="", prefix_sep="")
    lang_grouped = lang_encoded.groupby('resume_seq').sum().reset_index().drop(columns=['exam_name', 'score'])
    lang_grouped.columns = ['resume_seq', 'lang_2', 'lang_3', 'lang_4', 'lang_8', 'lang_9']
    lang_grouped[['lang_2', 'lang_3', 'lang_4', 'lang_8', 'lang_9']] = lang_grouped[['lang_2', 'lang_3', 'lang_4', 'lang_8', 'lang_9']].clip(upper=1)
    resume = resume.merge(lang_grouped, on='resume_seq', how='left')

    # 교육 데이터 병합
    resume = resume.merge(resume_edu, on='resume_seq', how='left')

    # 결측값 처리
    resume.fillna({'job_code_seq2': '정보없음', 'job_code_seq3': '정보없음', 'career_job_code': '정보없음'}, inplace=True)

    # 레이블 인코딩
    label_columns = ['job_code_seq1', 'job_code_seq2', 'job_code_seq3', 'career_job_code', 
                     'hischool_special_type', 'hischool_nation', 'hischool_gender']
    label_encoders = {column: LabelEncoder().fit_transform(resume[column]) for column in label_columns}
    resume.update(pd.DataFrame(label_encoders))

    # 언어 능력 결측값 처리 및 불필요한 컬럼 제거
    resume[['lang_2', 'lang_3', 'lang_4', 'lang_8', 'lang_9']] = resume[['lang_2', 'lang_3', 'lang_4', 'lang_8', 'lang_9']].fillna(0)
    resume.drop(['univ_major', 'univ_sub_major'], axis=1, inplace=True)

    # 날짜 관련 처리
    resume['reg_date'] = pd.to_datetime(resume['reg_date'])
    resume['updated_date'] = pd.to_datetime(resume['updated_date'])

    # 시간 관련 정수 데이터 만들기
    resume['update_reg_diff'] = (resume['updated_date'] - resume['reg_date']).dt.days

    # timestamp형태로 변경
    resume['reg_date'] = resume['reg_date'].apply(lambda x: int(x.timestamp()))
    resume['updated_date'] = resume['updated_date'].apply(lambda x: int(x.timestamp()))

    # 실수형인 hope_salary, last_salary, univ_score는 정수형으로 변경
    columns_to_convert = ['hope_salary', 'last_salary', 'univ_score']
    for column in columns_to_convert:
        resume[column] = resume[column].astype(int)

    # resume_seq를 기준으로 오름차순 정렬
    resume.sort_values(by='resume_seq', inplace=True)
    
    return resume

def preprocess_recruitment(recruitment):
    # 0값만 가진 career_end, career_start는 drop
    recruitment = recruitment.drop(['career_end', 'career_start'], axis=1)

    # 주소 시퀀스 결측치 처리 및 데이터 타입 변환
    recruitment['address_seq1'] = recruitment['address_seq1'].fillna(3).astype(int)

    # 체크박스 키워드 원-핫 인코딩 및 병합
    recruitment = pd.concat([recruitment, recruitment['check_box_keyword'].str.get_dummies(sep=';')], axis=1)

    # 불필요한 컬럼 제거
    recruitment.drop(columns=['address_seq2', 'address_seq3', 'text_keyword', 'check_box_keyword'], inplace=True)

    # recruitment_seq를 기준으로 오름차순 정렬
    recruitment.sort_values(by='recruitment_seq', inplace=True)

    return recruitment

def get_clusters_for_keywords(keywords_list, model, kmeans):
    """
    주어진 키워드 리스트에 대해 각 키워드가 속하는 클러스터를 예측.
    """
    clusters = [kmeans.predict([model.wv[k]])[0] if k in model.wv.index_to_key else -1 for k in keywords_list]
    return clusters

def encode_and_cluster_keywords(resume_data):
    # 키워드 분리 및 FastText 모델 학습
    keywords = resume_data['text_keyword'].str.split(';').dropna().tolist()
    model = FastText(sentences=keywords, vector_size=100, window=5, min_count=1, workers=4, sg=1, epochs=300)

    # KMeans 클러스터링
    words = list(model.wv.index_to_key)
    vectors = [model.wv[word] for word in words]
    n_clusters = 61 
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(vectors)
    
    # 키워드 클러스터링 결과를 데이터프레임에 추가
    all_clusters = resume_data['text_keyword'].str.split(';').fillna('').apply(lambda x: get_clusters_for_keywords(x, model, kmeans))
    for i in range(n_clusters):
        resume_data[f'keyword_cluster_{i+1}'] = all_clusters.apply(lambda clusters: int(i in clusters))

    return resume_data

def save_index_mappings(apply_train):
    # 'resume_seq'와 'recruitment_seq' 각각의 고유값을 추출.
    unique_resume_seq = apply_train['resume_seq'].unique()
    unique_recruitment_seq = apply_train['recruitment_seq'].unique()

    # 'resume_seq'에 대한 고유한 정수 인덱스 매핑을 생성.
    resume_seq_to_index = {seq: idx for idx, seq in enumerate(unique_resume_seq)}

    # 'recruitment_seq'에 대한 정수 인덱스 매핑을 생성.
    recruitment_seq_to_index = {seq: idx + len(unique_resume_seq) for idx, seq in enumerate(unique_recruitment_seq)}

    # 결과를 파일에 저장할 수 있습니다. 예를 들어 pickle 형식을 사용할 수 있습니다.
    with open(f'{args.directory}/resume_seq_to_index.pkl', 'wb') as f:
        pd.to_pickle(resume_seq_to_index, f)

    with open(f'{args.directory}/recruitment_seq_to_index.pkl', 'wb') as f:
        pd.to_pickle(recruitment_seq_to_index, f)
    
    return resume_seq_to_index, recruitment_seq_to_index

# 데이터프레임의 열 이름에 접미사 추가하는 함수 정의
def append_suffix(dataframe, exceptions):
    new_columns = {}
    for col in dataframe.columns:
        if col in exceptions:
            # 예외적으로 :float 접미사 추가
            new_columns[col] = f"{col}:float"
        else:
            # 모든 다른 열에 :token 접미사 추가
            new_columns[col] = f"{col}:token"

    return dataframe.rename(columns=new_columns)

def preprocess_for_recbole(directory):
    # 데이터 로드
    file_names = ['resume', 'resume_education', 'resume_certificate', 'resume_language', 'recruitment', 'company', 'apply_train']
    data = load_data(directory, file_names)
    
    # 이력서 데이터 전처리
    final_resume = preprocess_resume(data['resume'], data['resume_education'], data['resume_certificate'], data['resume_language'])
    
    # 공고 데이터 전처리
    final_recruitment = preprocess_recruitment(data['recruitment'])
    
    # 이력서 키워드 클러스터링
    final_resume = encode_and_cluster_keywords(final_resume)
    final_resume = final_resume.drop('text_keyword', axis=1)

    # apply_train에 있는 값 recbole위한 정수형으로 매핑
    apply_train = data['apply_train']

    resume_seq_to_index, recruitment_seq_to_index = save_index_mappings(apply_train)

    apply_train.rename(columns={'resume_seq' : 'user_id', 'recruitment_seq' : 'item_id'}, inplace=True)
    final_resume.rename(columns={'resume_seq' : 'user_id'}, inplace=True)
    final_recruitment.rename(columns={'recruitment_seq' : 'item_id'}, inplace=True)

    apply_train['user_id'] = apply_train['user_id'].map(resume_seq_to_index)
    apply_train['item_id'] = apply_train['item_id'].map(recruitment_seq_to_index)
    final_resume['user_id'] = final_resume['user_id'].map(resume_seq_to_index)
    final_recruitment['item_id'] = final_recruitment['item_id'].map(recruitment_seq_to_index)

    apply_train['label'] = 1

    # 열을 모두 정수형으로 만들었으므로 열 뒤에 :token 붙이기, 다만 final_resume의 reg_date랑 update_date는 float붙이기
    apply_train = append_suffix(apply_train, ['label'])
    final_recruitment = append_suffix(final_recruitment, [])
    final_resume = append_suffix(final_resume, ['reg_date', 'updated_date'])

    return apply_train, final_resume, final_recruitment


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=str, help="Path to the directory containing the data files")
    parser.add_argument('--output_path', type=str, help="Path where the processed tsv will be saved")
    args = parser.parse_args()

    apply_train, final_resume, final_recruitment = preprocess_for_recbole(args.directory)

    apply_train.to_csv(f'{args.output_path}/Dacon.inter', sep='\t', index=False)
    final_resume.to_csv(f'{args.output_path}/Dacon.user', sep='\t', index=False)
    final_recruitment.to_csv(f'{args.output_path}/Dacon.item', sep='\t', index=False)

    print("Tsv Data Saved in your output Path!")