import pandas as pd
import numpy as np
import os
import argparse
from datetime import datetime
from itertools import repeat

import warnings
warnings.filterwarnings(action='ignore')

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from gensim.models import FastText

import config

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

    # 언어 능력 데이터 병합 및 필요없는 열 정리
    lang_grouped = resume_lang.groupby('resume_seq').sum().reset_index().drop(columns=['exam_name', 'score'])
    resume = resume.merge(lang_grouped, on='resume_seq', how='left')
    resume[['language']] = resume[['language']].fillna(0)

    # 교육 데이터 병합 및 필요없는 열 정리
    resume = resume.merge(resume_edu, on='resume_seq', how='left')
    resume.drop(['univ_major', 'univ_sub_major'], axis=1, inplace=True)

    # 결측값 처리
    resume.fillna({'job_code_seq2': '정보없음', 'job_code_seq3': '정보없음', 'career_job_code': '정보없음'}, inplace=True)

    # 레이블 인코딩
    label_columns = ['job_code_seq1', 'job_code_seq2', 'job_code_seq3', 'career_job_code', 
                     'hischool_special_type', 'hischool_nation', 'hischool_gender']
    label_encoders = {column: LabelEncoder().fit_transform(resume[column]) for column in label_columns}
    resume.update(pd.DataFrame(label_encoders))

    # 날짜 관련 처리
    resume['reg_date'] = pd.to_datetime(resume['reg_date'])
    resume['updated_date'] = pd.to_datetime(resume['updated_date'])

    # 시간 관련 정수 데이터 만들기
    resume['update_reg_diff'] = (resume['updated_date'] - resume['reg_date']).dt.days

    # reg_date와 updated_date는 drop
    resume.drop(['reg_date', 'updated_date'], axis=1, inplace=True)

    # resume_seq를 기준으로 오름차순 정렬
    resume.sort_values(by='resume_seq', inplace=True)
    
    return resume

def get_clusters_for_keywords(keywords_list, model, kmeans):
    """
    주어진 키워드 리스트에 대해 각 키워드가 속하는 클러스터를 예측.
    """
    clusters = [kmeans.predict([model.wv[k]])[0] if k in model.wv.index_to_key else -1 for k in keywords_list]
    return clusters

def get_modified_data_except_field(X, all_fields, continuous_fields, categorical_fields, is_bin=False):
    X_modified = X.iloc[:, :1]
    
    for index, col in enumerate(X.iloc[:, 1:].columns):
        if col not in all_fields:
            print("{} not included: Check your column list".format(col))
            raise ValueError

        if col in continuous_fields:
            scaler = MinMaxScaler()

            # 연속형 변수도 구간화 할 것인가?
            if is_bin:
                X_bin = pd.cut(scaler.fit_transform(X[[col]]).reshape(-1, ), config.NUM_BIN, labels=False)
                X_bin = pd.Series(X_bin).astype('str')

                X_bin_col = pd.get_dummies(X_bin, prefix=col, prefix_sep='-')
                X_modified = pd.concat([X_modified, X_bin_col], axis=1)

            else:
                X_cont_col = pd.DataFrame(scaler.fit_transform(X[[col]]), columns=[col])
                X_modified = pd.concat([X_modified, X_cont_col], axis=1)

        if col in categorical_fields:
            if col == 'text_keyword':
                keywords = X[col].str.split(';').dropna().tolist()
                model = FastText(sentences=keywords, vector_size=100, window=5, min_count=1, workers=4, sg=1, epochs=300)

                # KMeans 클러스터링
                words = list(model.wv.index_to_key)
                vectors = [model.wv[word] for word in words]
                n_clusters = 61 
                kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(vectors)
                
                # 키워드 클러스터링 결과를 데이터프레임에 추가
                all_clusters = X[col].str.split(';').fillna('').apply(lambda x: get_clusters_for_keywords(x, model, kmeans))
                for i in range(n_clusters):
                    X_modified[f'keyword_cluster-{i+1}'] = all_clusters.apply(lambda clusters: int(i in clusters))

            else:
                X_cat_col = pd.get_dummies(X[col], prefix=col, prefix_sep='-', dtype=int)
                X_modified = pd.concat([X_modified, X_cat_col], axis=1)

    print('Data Prepared...')
    print('X shape: {}'.format(X_modified.shape))

    return X_modified


def preprocess_for_cosine(directory):
    # 데이터 로드
    file_names = ['resume', 'resume_education', 'resume_certificate', 'resume_language']
    data = load_data(directory, file_names)
    
    # 이력서 데이터 전처리
    resume = preprocess_resume(data['resume'], data['resume_education'], data['resume_certificate'], data['resume_language'])

    resume_all_fields = config.RESUME_ALL_FIELDS
    resume_cont_fields = config.RESUME_CONT_FIELDS
    resume_cat_fields = config.RESUME_CAT_FIELDS

    # 데이터 원-핫 인코딩, 스케일링
    resume_for_cosine = get_modified_data_except_field(resume, resume_all_fields, resume_cont_fields, resume_cat_fields)

    return resume_for_cosine


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=str, help="Path to the directory containing the data files")
    args = parser.parse_args()

    resume_for_cosine = preprocess_for_cosine(args.directory)

    resume_for_cosine.to_csv(f'{args.directory}/resume_for_cosine.csv', index=False)

    print("Csv saved in your directory!")