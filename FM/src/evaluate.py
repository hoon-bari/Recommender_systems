import pandas as pd
import numpy as np
import os
import random
import json
from tqdm import tqdm
from itertools import islice
import tensorflow as tf

import config
from preprocessing import get_modified_data
from DeepFM import DeepFM
tf.config.set_visible_devices([], 'GPU')

from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import BinaryAccuracy, AUC

import warnings
warnings.filterwarnings(action='ignore')



def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

seed_everything(config.SEED)  # Seed 고정

def load_json(file_name):
    with open(file_name, 'r') as f:
        return json.load(f)
    
def predict_and_recommend(evaluate_df, model, top_k=5):

    test_ds = tf.data.Dataset.from_tensor_slices(
        tf.cast(evaluate_df.values, tf.float32)
        ).batch(config.BATCH_SIZE)
    
    predictions = model.predict(test_ds)

    # 상위 k개 아이템의 인덱스를 얻음
    top_indices = np.argsort(predictions)[-top_k:]
    
    # 예측된 확률과 함께 상위 아이템의 ID를 반환
    top_items_with_scores = [(evaluate_df.iloc[i]['recruitment_seq'], predictions[i]) for i in top_indices]
    top_items_with_scores.sort(key=lambda x: x[1], reverse=True)
    
    return top_items_with_scores

def get_eval_data():
    apply_train = pd.read_csv('/Users/seunghoonchoi/Downloads/Recommend/FM_models/Data/apply_train.csv')
    resume_for_eval = pd.read_csv('/Users/seunghoonchoi/Downloads/Recommend/FM_models/Data_for_eval/resume_for_eval.csv')
    recruitment_for_eval = pd.read_csv('/Users/seunghoonchoi/Downloads/Recommend/FM_models/Data_for_eval/recruitment_for_eval.csv')

    model_path = '/Users/seunghoonchoi/Downloads/Recommend/FM_models/models/deepfm_model_epoch(32)_batch(32)_embedding(6)'
    
    resume_ids = load_json('/Users/seunghoonchoi/Downloads/Recommend/FM_models/Data/unique_resume_ids.json')
    recruitment_ids = load_json('/Users/seunghoonchoi/Downloads/Recommend/FM_models/Data/unique_recruitment_ids.json')

    all_fields = config.ALL_FIELDS
    cont_fields = config.CONT_FIELDS
    cat_fields = config.CAT_FIELDS

    _, _, resume_modified = get_modified_data(resume_for_eval.iloc[:, 1:], all_fields, cont_fields, cat_fields)
    _, _, recruitment_modified = get_modified_data(recruitment_for_eval.iloc[:, 1:], all_fields, cont_fields, cat_fields)

    resume_modified = pd.concat([resume_for_eval.iloc[:, 0], resume_modified], axis=1)

    recruitment_modified = pd.concat([recruitment_for_eval.iloc[:, 0], recruitment_modified], axis=1)

    model = tf.keras.models.load_model(model_path)

    return apply_train, resume_ids, recruitment_ids, resume_modified, recruitment_modified, model

def evaluate(apply_train, resume_ids, recruitment_ids, resume_modified, recruitment_modified, model):
    reverse_resume_ids = {v: k for k, v in resume_ids.items()}
    reverse_recruitment_ids = {v: k for k, v in recruitment_ids.items()}

    user_interacted_items = apply_train.groupby('resume_seq')['recruitment_seq'].apply(set).to_dict()
    user_interacted_items_encoded = {
        resume_ids.get(user_id): {recruitment_ids.get(item_id) for item_id in items if recruitment_ids.get(item_id) is not None}
        for user_id, items in user_interacted_items.items() if resume_ids.get(user_id) is not None
    }

    all_recommendations = []

    for encoded_user_id, user_id in reverse_resume_ids.items():

        user_profile = resume_modified.loc[resume_modified['resume_seq'] == encoded_user_id]
        interacted_items = user_interacted_items_encoded.get(encoded_user_id, set())
        recruitment_profiles = recruitment_modified.loc[~recruitment_modified['recruitment_seq'].isin(interacted_items)]

        recruitment_profiles.reset_index(drop=True, inplace=True)

        repeated_user_profile = pd.DataFrame(np.repeat(user_profile.values, len(recruitment_profiles), axis=0), columns=user_profile.columns)
        
        resume_seq = repeated_user_profile['resume_seq']
        resume_df = repeated_user_profile.drop('resume_seq', axis=1)

        recruitment_seq = recruitment_profiles['recruitment_seq']
        recruitment_df = recruitment_profiles.drop('recruitment_seq', axis=1)
        
        evaluate_df = pd.concat([resume_seq, resume_df, recruitment_seq, recruitment_df], axis=1)

        top_recommendations = predict_and_recommend(evaluate_df, model)
        
        for item_id, _ in top_recommendations:
            original_item_id = reverse_recruitment_ids[item_id]
            all_recommendations.append({'resume_seq': user_id, 'recruitment_seq': original_item_id})

    recommendations_df = pd.DataFrame(all_recommendations)

    return recommendations_df

if __name__ == "__main__":
    apply_train, resume_ids, recruitment_ids, resume_modified, recruitment_modified, model = get_eval_data()
    recommendations_df = evaluate(apply_train, resume_ids, recruitment_ids, resume_modified, recruitment_modified, model)
    recommendations_df.to_csv('final_submission.csv', index=False)