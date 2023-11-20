import pandas as pd
import numpy as np
import os
import argparse
import json
from datetime import datetime
from itertools import repeat

import warnings
warnings.filterwarnings(action='ignore')

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from gensim.models import FastText

import config
from generate_neg_samples import get_cos_similarity_of_user, prepare_negative_samples, generate_negative_samples, create_negative_samples_df

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

def preprocess_recruitment(recruitment, company):
    # 0값만 가진 career_end, career_start는 drop
    recruitment = recruitment.drop(['career_end', 'career_start'], axis=1)

    # 주소 시퀀스 결측치 처리 및 데이터 타입 변환
    recruitment['address_seq1'] = recruitment['address_seq1'].fillna(3).astype(int)

    recruitment = recruitment.merge(company, on='recruitment_seq', how='left')
    recruitment[['company_type_seq', 'supply_kind', 'employee']] = recruitment[['company_type_seq', 'supply_kind', 'employee']].fillna(0)

    # 불필요한 컬럼 제거
    recruitment.drop(columns=['address_seq2', 'address_seq3', 'text_keyword'], inplace=True)

    return recruitment

def get_clusters_for_keywords(keywords_list, model, kmeans):
    """
    주어진 키워드 리스트에 대해 각 키워드가 속하는 클러스터를 예측.
    """
    clusters = [kmeans.predict([model.wv[k]])[0] if k in model.wv.index_to_key else -1 for k in keywords_list]
    return clusters

def get_modified_data(X, all_fields, continuous_fields, categorical_fields, is_bin=False):
    field_dict = dict()
    field_index = []
    X_modified = pd.DataFrame()
    
    for index, col in enumerate(X.columns):
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
                field_dict[index] = list(X_bin_col.columns)
                field_index.extend(repeat(index, X_bin_col.shape[1]))
                X_modified = pd.concat([X_modified, X_bin_col], axis=1)

            else:
                X_cont_col = pd.DataFrame(scaler.fit_transform(X[[col]]), columns=[col])
                field_dict[index] = col
                field_index.append(index)
                X_modified = pd.concat([X_modified, X_cont_col], axis=1)

        if col in categorical_fields:

            if col == 'label':
                field_dict[index] = col
                field_index.append(index)
                X_modified = pd.concat([X_modified, X[col]], axis=1)

            elif col == 'check_box_keyword':
                dummy_keywords = X[col].str.get_dummies(sep=';')
                dummy_keywords_prefixed = dummy_keywords.add_prefix(f'{col}-')
                field_dict[index] = list(dummy_keywords_prefixed.columns)
                field_index.extend([index] * dummy_keywords.shape[1])
                X_modified = pd.concat([X_modified, dummy_keywords_prefixed], axis=1)

            elif col == 'text_keyword':
                keywords = X[col].str.split(';').dropna().tolist()
                model = FastText(sentences=keywords, vector_size=100, window=5, min_count=1, workers=4, sg=1, epochs=300)

                # KMeans 클러스터링
                words = list(model.wv.index_to_key)
                vectors = [model.wv[word] for word in words]
                n_clusters = 61 
                kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(vectors)
                
                # 키워드 클러스터링 결과를 데이터프레임에 추가
                all_clusters = X[col].str.split(';').fillna('').apply(lambda x: get_clusters_for_keywords(x, model, kmeans))
                cluster_col_names = []  # 클러스터링 결과 열 이름 저장을 위한 리스트
                for i in range(n_clusters):
                    cluster_col_name = f'keyword_cluster-{i+1}'
                    X_modified[cluster_col_name] = all_clusters.apply(lambda clusters: int(i in clusters))
                    cluster_col_names.append(cluster_col_name)
                
                field_dict[index] = cluster_col_names
                field_index.extend([index] * n_clusters)

            else:
                X_cat_col = pd.get_dummies(X[col], prefix=col, prefix_sep='-', dtype=int)
                field_dict[index] = list(X_cat_col.columns)
                field_index.extend(repeat(index, X_cat_col.shape[1]))
                X_modified = pd.concat([X_modified, X_cat_col], axis=1)

    print('Data Prepared...')
    print('X shape: {}'.format(X_modified.shape))
    print('# of Feature: {}'.format(len(field_index)))
    print('# of Field: {}'.format(len(field_dict)))

    return field_dict, field_index, X_modified


def get_pos_neg_samples(apply_train, resume_cosine):

    # 코사인 유사도 계산
    cosine_sim_df = get_cos_similarity_of_user(resume_cosine)

    # 임계값 설정 및 negative 후보 준비
    negative_candidates_dict, resume_to_recruitments = prepare_negative_samples(cosine_sim_df, apply_train)

    # Negative 샘플 생성
    results = generate_negative_samples(negative_candidates_dict, resume_to_recruitments)

    # negative_sample 데이터 형성
    negative_samples_df = create_negative_samples_df(results)

    # positive_sample 데이터 형성 및 통합
    data = {'resume_seq': list(resume_to_recruitments.keys()), 'recruitment_seq_positives': [list(val) for val in resume_to_recruitments.values()]}
    positive_samples_df = pd.DataFrame(data, columns=['resume_seq', 'recruitment_seq_positives'])

    positive_pairs_df = positive_samples_df.explode('recruitment_seq_positives').rename(columns={'recruitment_seq_positives': 'recruitment_seq'})
    negative_pairs_df = negative_samples_df.explode('recruitment_seq_negatives').rename(columns={'recruitment_seq_negatives': 'recruitment_seq'})

    positive_pairs_df['label'] = 1
    negative_pairs_df['label'] = 0

    # neg_sample 포함된 상관관계 데이터프레임 형성
    all_pairs_df = pd.concat([positive_pairs_df, negative_pairs_df], ignore_index=True)

    return all_pairs_df

def save_mapping_to_json(mapping, file_name):
    with open(file_name, 'w') as f:
        json.dump(mapping, f, indent=4)

def preprocess_for_DeepFM(directory):
    # 데이터 로드
    file_names = ['resume', 'resume_education', 'resume_certificate', 'resume_language', 'resume_for_cosine', 'recruitment', 'company', 'apply_train']
    data = load_data(directory, file_names)
    
    # 이력서 데이터 전처리
    resume = preprocess_resume(data['resume'], data['resume_education'], data['resume_certificate'], data['resume_language'])

    # 공고 데이터 전처리
    recruitment = preprocess_recruitment(data['recruitment'], data['company'])

    # negative sample 및 label 포함된 apply 데이터 생성
    all_pairs_df = get_pos_neg_samples(data['apply_train'], data['resume_for_cosine'])

    # 각 시퀀스에 대한 유니크한 정수 인코딩을 생성
    unique_resume_ids = {v: k for k, v in enumerate(resume['resume_seq'].unique())}
    unique_recruitment_ids = {v: k for k, v in enumerate(recruitment['recruitment_seq'].unique(), start=len(unique_resume_ids))}

    # JSON 파일로 매핑 저장
    save_mapping_to_json(unique_resume_ids, f'{directory}/unique_resume_ids.json')
    save_mapping_to_json(unique_recruitment_ids,  f'{directory}/unique_recruitment_ids.json')

    # 데이터프레임에 새로운 인코딩을 적용
    resume['resume_seq'] = resume['resume_seq'].map(unique_resume_ids)
    recruitment['recruitment_seq'] = recruitment['recruitment_seq'].map(unique_recruitment_ids)

    resume.to_csv('/Users/seunghoonchoi/Downloads/Recommend/FM_models/Data_for_eval/resume_for_eval.csv', index=False)
    recruitment.to_csv('/Users/seunghoonchoi/Downloads/Recommend/FM_models/Data_for_eval/recruitment_for_eval.csv', index=False)

    all_pairs_df['resume_seq'] = all_pairs_df['resume_seq'].map(unique_resume_ids)
    all_pairs_df['recruitment_seq'] = all_pairs_df['recruitment_seq'].map(unique_recruitment_ids)
   
    # 이력서 및 공고 데이터와 병합
    combined_features_df = all_pairs_df.merge(resume, on='resume_seq', how='left')
    combined_features_df = combined_features_df.merge(recruitment, on='recruitment_seq', how='left')
   
    # 'label' 열을 맨 앞으로 가져오기
    label = combined_features_df['label']
    resume_seq = combined_features_df['resume_seq']
    recruitment_seq = combined_features_df['recruitment_seq']

    # 원본 데이터프레임에서 컬럼을 삭제합니다.
    combined_features_df.drop(['label', 'resume_seq', 'recruitment_seq'], axis=1, inplace=True)

    # 삭제된 컬럼을 원하는 순서로 데이터프레임 앞쪽에 다시 삽입합니다.
    combined_features_df.insert(0, 'recruitment_seq', recruitment_seq)
    combined_features_df.insert(0, 'resume_seq', resume_seq)
    combined_features_df.insert(0, 'label', label)

    return combined_features_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=str, help="Path to the directory containing the data files")
    parser.add_argument('--output_path', type=str, help="Path where the processed tsv will be saved")
    args = parser.parse_args()

    combined_features_df = preprocess_for_DeepFM(args.directory)

    combined_features_df.to_parquet(f'{args.output_path}/train_data.parquet', index=False, compression='snappy')

    print("Parquet Data Saved in your output Path!")