import pandas as pd
import numpy as np
import os
import argparse
import pickle
from datetime import datetime

import torch

import warnings
warnings.filterwarnings(action='ignore')

from gensim.models import FastText

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from builder import PandasGraphBuilder

from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel

def load_data(directory, file_names):
    data = {}
    for file_name in file_names:
        data[file_name] = pd.read_csv(os.path.join(directory, f'Data/{file_name}.csv'))
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

    # 기준 날짜 설정 (가장 이른 reg_date)
    base_date = resume['reg_date'].min()

    # 기준 날짜로부터의 경과 일수를 계산
    resume['reg_date'] = (resume['reg_date'] - base_date).dt.days
    resume['updated_date'] = (resume['updated_date'] - base_date).dt.days

    # resume_seq를 기준으로 오름차순 정렬
    resume.sort_values(by='resume_seq', inplace=True)
    
    return resume

def preprocess_recruitment(recruitment, company):
    # 0값만 가진 career_end, career_start는 drop
    recruitment = recruitment.drop(['career_end', 'career_start'], axis=1)

    # 주소 시퀀스 결측치 처리 및 데이터 타입 변환
    recruitment['address_seq1'] = recruitment['address_seq1'].fillna(3).astype(int)

    # 텍스트 키워드 기반의 이진 특성 생성
    keyword_features = {
        'has_text_keyword': recruitment['text_keyword'].notna(),
        'part_time': recruitment['text_keyword'].str.contains('아르바이트'),
        'intern': recruitment['text_keyword'].str.contains('인턴'),
        'entry_level': recruitment['text_keyword'].str.contains('신입'),
        'experienced': recruitment['text_keyword'].str.contains('경력|경력직'),
        'team_leader': recruitment['text_keyword'].str.contains('팀장|팀장급')
    }
    
    for key, value in keyword_features.items():
        recruitment[key] = value.fillna(0).astype(int)

    # 체크박스 키워드 원-핫 인코딩 및 병합
    recruitment = pd.concat([recruitment, recruitment['check_box_keyword'].str.get_dummies(sep=';')], axis=1)

    # 불필요한 컬럼 제거
    recruitment.drop(columns=['address_seq2', 'address_seq3', 'text_keyword', 'check_box_keyword'], inplace=True)

    # 회사 데이터 병합
    recruitment = recruitment.merge(company, on='recruitment_seq', how='left')

    # 직원 수 기반 범주형 변수 생성
    recruitment['employee_category'] = np.select(
        [recruitment['employee'].isna(),
         recruitment['employee'] < 5,
         recruitment['employee'] < 1000],
        [0, 1, 2],
        default=3
    )

    # 불필요한 직원 수 컬럼 제거 및 결측치 처리
    recruitment.drop(columns=['employee'], inplace=True)
    recruitment[['company_type_seq', 'supply_kind']] = recruitment[['company_type_seq', 'supply_kind']].fillna(0)

    # recruitment_seq를 기준으로 오름차순 정렬
    recruitment.sort_values(by='recruitment_seq', inplace=True)

    return recruitment

def get_clusters_for_keywords(keywords_list, model, kmeans):
    """
    주어진 키워드 리스트에 대해 각 키워드가 속하는 클러스터를 예측.
    """
    clusters = [kmeans.predict([model.wv[k]])[0] if k in model.wv.index_to_key else -1 for k in keywords_list]
    return clusters
​
def encode_and_cluster_keywords(resume_data):
    # 키워드 분리 및 FastText 모델 학습
    keywords = resume_data['text_keyword'].str.split(';').dropna().tolist()
    model = FastText(sentences=keywords, vector_size=100, window=5, min_count=2, workers=4, sg=1, epochs=200)
​
    # KMeans 클러스터링
    words = list(model.wv.index_to_key)
    vectors = [model.wv[word] for word in words]
    n_clusters = 61
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(vectors)
    
    # 키워드 클러스터링 결과를 데이터프레임에 추가
    all_clusters = resume_data['text_keyword'].str.split(';').fillna('').apply(lambda x: get_clusters_for_keywords(x, model, kmeans))
    for i in range(n_clusters):
        resume_data[f'keyword_cluster_{i+1}'] = all_clusters.apply(lambda clusters: int(i in clusters))
​
    return resume_data

# 이진 값 아닌 친구들은 minmaxscaling후 통합
def scale_and_combine_features(df, binary_cols, exclude_cols):
    cols_to_scale = df.columns.difference(binary_cols + exclude_cols)
    
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df[cols_to_scale])
    scaled_features_df = pd.DataFrame(scaled_features, index=df.index, columns=cols_to_scale)
    
    combined_features = pd.concat([scaled_features_df, df[binary_cols]], axis=1)
    return combined_features

def graph_building(directory):
    # 데이터 로드
    file_names = ['resume', 'resume_education', 'resume_certificate', 'resume_language', 'recruitment', 'company', 'apply_train']
    data = load_data(directory, file_names)
    
    # 이력서 데이터 전처리
    final_resume = preprocess_resume(data['resume'], data['resume_education'], data['resume_certificate'], data['resume_language'])
    
    # 공고 데이터 전처리
    final_recruitment = preprocess_recruitment(data['recruitment'], data['company'])
    
    # 이력서 키워드 클러스터링
    final_resume = encode_and_cluster_keywords(final_resume)
    final_resume = final_resume.drop('text_keyword', axis=1)

    final_resume.to_csv('/Users/seunghoonchoi/Downloads/Dacon_recommend_system/Data/final_resume.csv', index=False)

    # # apply_train에 공고 지원 횟수 추가. 인기도 반영 위함.
    # apply_train = data['apply_train']
    # apply_train['recruitment_count'] = apply_train['recruitment_seq'].map(apply_train['recruitment_seq'].value_counts())

    # 이진값 분류
    binary_columns_resume = [col for col in final_resume.columns if final_resume[col].nunique() == 2]
    binary_columns_recruitment = [col for col in final_recruitment.columns if final_recruitment[col].nunique() == 2]

    # 이진값이 아닌 값만 스케일링
    final_resume[binary_columns_resume] = final_resume[binary_columns_resume].apply(lambda col: col.astype(int) if not pd.api.types.is_integer_dtype(col) else col)
    final_recruitment[binary_columns_recruitment] = final_recruitment[binary_columns_recruitment].apply(lambda col: col.astype(int) if not pd.api.types.is_integer_dtype(col) else col)

    exclude_columns_resume = ['resume_seq']
    exclude_columns_recruitment = ['recruitment_seq']

    scaled_resume_features = scale_and_combine_features(final_resume, binary_columns_resume, exclude_columns_resume)
    scaled_recruitment_features = scale_and_combine_features(final_recruitment, binary_columns_recruitment, exclude_columns_recruitment)
    
    # # primary_key를 제외하고 모든 컬럼을 특성으로 사용
    # resume_features = final_resume.drop(columns=['resume_seq'])
    # recruitment_features = final_recruitment.drop(columns=['recruitment_seq'])

    # 그래프 만들기
    graph_builder = PandasGraphBuilder()

    graph_builder.add_entities(final_resume, primary_key='resume_seq', name='resume', features=scaled_resume_features)
    graph_builder.add_entities(final_recruitment, primary_key='recruitment_seq', name='recruitment', features=scaled_recruitment_features)

    # 관계 추가
    graph_builder.add_binary_relations(data['apply_train'], source_key='resume_seq', destination_key='recruitment_seq', name='applied')
    graph_builder.add_binary_relations(data['apply_train'], source_key='recruitment_seq', destination_key='resume_seq', name='applied_by')

    g = graph_builder.build()

    # g.edges['applied'].data['recruitment_count'] = torch.LongTensor(apply_train['recruitment_count'].values)
    # g.edges['applied_by'].data['recruitment_count'] = torch.LongTensor(apply_train['recruitment_count'].values)

    return g


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=str, help="Path to the directory containing the data files")
    parser.add_argument('--output_path', type=str, help="Path where the processed graph will be saved")
    args = parser.parse_args()

    # 그래프 구축
    g = graph_building(args.directory)

    # 데이터셋 저장
    dataset = {
        'train_graph': g,
        'context_type': 'resume',
        'item_type': 'recruitment',
        'context_to_item_type': 'applied',
        'item_to_context_type': 'applied_by'
    }

    output_path = os.path.join(args.output_path, 'graph_data.pickle')
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)

    print(f"Graph data saved to {output_path}")
