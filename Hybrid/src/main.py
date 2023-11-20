import train
import pandas as pd

from time import perf_counter

if __name__ == '__main__':
    start = perf_counter()

    current_recommendations = {}

    file_names = ['apply_train']

    data = train.load_data('../Data', file_names)

    apply_train = data['apply_train']

    test_df = apply_train.copy()

    # 1,2,3번째 추천
    user_item_matrix = train.make_matrix(test_df)

    user_predicted_scores, item_predicted_scores = train.calculate_scores(user_item_matrix)

    first_recommendations = train.collaborative_filtering_with_margin(user_item_matrix, user_predicted_scores, item_predicted_scores)

    current_recommendations = train.update_recommendations(current_recommendations, first_recommendations)

    print('first recommendation finished!')

    # 4, 5번째 추천
    model = train.train_lmf_model(user_item_matrix)

    second_recommendations = train.collaborative_filtering_with_lmf(model, user_item_matrix, current_recommendations)

    current_recommendations = train.update_recommendations(current_recommendations, second_recommendations)

    data = []

    # 딕셔너리 순회
    for resume, recruitments in current_recommendations.items():
        for recruitment in recruitments:
            data.append({'resume_seq': resume, 'recruitment_seq': recruitment})

   
    # 리스트로부터 DataFrame 생성
    recommendations_df = pd.DataFrame(data)

    recommendations_df.to_csv('submission.csv', index=False)

    print("결과 저장 완료!")
    print("걸린 시간: {:.3f}".format(perf_counter() - start))




