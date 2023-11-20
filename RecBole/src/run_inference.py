import argparse
import torch
import numpy as np
import pandas as pd

from recbole.quick_start import load_data_and_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', '-m', type=str, default='saved/model.pth', help='name of models')
    
    args, _ = parser.parse_known_args()
    
    # model, dataset 불러오기
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(args.model_path)
    
    # device 설정
    device = config.final_config_dict['device']
    
    # user, item id -> token 변환 array
    user_id2token = dataset.field2id_token['user_id']
    item_id2token = dataset.field2id_token['item_id']
    
    # user-item sparse matrix
    matrix = dataset.inter_matrix(form='csr')

    # user id, predict item id 저장 변수
    pred_list = None
    user_list = None
    
    model.eval()
    for data in test_data:
        interaction = data[0].to(device)
        score = model.full_sort_predict(interaction)
        
        rating_pred = score.cpu().data.numpy().copy()
        user_id = interaction['user_id'].cpu().numpy()
        
        # 사용자가 상호작용한 아이템 인덱스를 가져옵니다.
        interacted_indices = matrix[user_id].indices

        # 상호작용한 아이템의 점수를 0으로 설정합니다.
        rating_pred[interacted_indices] = 0

        ind = np.argpartition(rating_pred, -5)[-5:]
        
        arr_ind = rating_pred[ind]
       
       # 추출된 값들을 내림차순으로 정렬하기 위한 인덱스를 얻음
        arr_ind_argsort = np.argsort(arr_ind)[::-1]

        # 실제 값들을 정렬된 순서대로 인덱스 배열에 적용
        batch_pred_list = ind[arr_ind_argsort]
        
        # 예측값 저장
        if pred_list is None:
            pred_list = batch_pred_list
            # batch_pred_list 길이만큼 user_id를 반복
            user_list = np.repeat(user_id, len(batch_pred_list))
        else:
            pred_list = np.append(pred_list, batch_pred_list, axis=0)
            # batch_pred_list 길이만큼 user_id를 반복하여 추가
            user_list = np.append(user_list, np.repeat(user_id, len(batch_pred_list)), axis=0)

    # 인덱스 매핑 파일 로드
    with open('/Users/seunghoonchoi/Downloads/Recommend/RecBole/Dacon_recommender_system/Data/resume_seq_to_index.pkl', 'rb') as f:
        resume_seq_to_index = pd.read_pickle(f)

    with open('/Users/seunghoonchoi/Downloads/Recommend/RecBole/Dacon_recommender_system/Data/recruitment_seq_to_index.pkl', 'rb') as f:
        recruitment_seq_to_index = pd.read_pickle(f)

    # 인덱스에서 시퀀스로의 역 매핑 생성
    index_to_resume_seq = {idx: seq for seq, idx in resume_seq_to_index.items()}
    index_to_recruitment_seq = {idx: seq for seq, idx in recruitment_seq_to_index.items()}

    # 결과를 저장할 빈 리스트 초기화
    final_result = []

    # user_list와 pred_list에 있는 인덱스를 실제 'resume_seq'와 'recruitment_seq'로 변환
    for user, item in zip(user_list, pred_list):
        # user_id2token을 사용하여 변환된 사용자 ID를 얻고
        # index_to_resume_seq을 사용하여 원래의 'resume_seq'로 변환
        original_user_seq = index_to_resume_seq.get(int(user_id2token[user]), -1)

        # item_id2token을 사용하여 변환된 아이템 ID를 얻고
        # index_to_recruitment_seq을 사용하여 원래의 'recruitment_seq'로 변환
        original_item_seq = index_to_recruitment_seq.get(int(item_id2token[item]), -1)

        # 최종 결과에 추가
        final_result.append((original_user_seq, original_item_seq))

    # 결과를 DataFrame으로 변환하고 CSV 파일로 저장
    final_dataframe = pd.DataFrame(final_result, columns=['resume_seq', 'recruitment_seq'])
    final_dataframe.sort_values(by='resume_seq', inplace=True)
    final_dataframe.to_csv('/Users/seunghoonchoi/Downloads/Recommend/RecBole/saved/final_submission.csv', index=False)
    print('Final mapping done and saved to CSV!')