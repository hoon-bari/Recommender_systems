import pandas as pd
import numpy as np

from time import perf_counter
import train


def recall5(answer_df, submission_df):
    """
    Calculate recall@5 for given dataframes.
    
    Parameters:
    - answer_df: DataFrame containing the ground truth
    - submission_df: DataFrame containing the predictions
    
    Returns:
    - recall: Recall@5 value
    """
    
    primary_col = answer_df.columns[0]
    secondary_col = answer_df.columns[1]
    
    # Check if each primary_col entry has exactly 5 secondary_col predictions
    prediction_counts = submission_df.groupby(primary_col).size()
    if not all(prediction_counts == 5):
        raise ValueError(f"Each {primary_col} should have exactly 5 {secondary_col} predictions.")


    # Check for NULL values in the predicted secondary_col
    if submission_df[secondary_col].isnull().any():
        raise ValueError(f"Predicted {secondary_col} contains NULL values.")
    
    # Check for duplicates in the predicted secondary_col for each primary_col
    duplicated_preds = submission_df.groupby(primary_col).apply(lambda x: x[secondary_col].duplicated().any())
    if duplicated_preds.any():
        raise ValueError(f"Predicted {secondary_col} contains duplicates for some {primary_col}.")


    # Filter the submission dataframe based on the primary_col present in the answer dataframe
    submission_df = submission_df[submission_df[primary_col].isin(answer_df[primary_col])]
    
    # For each primary_col, get the top 5 predicted secondary_col values
    top_5_preds = submission_df.groupby(primary_col).apply(lambda x: x[secondary_col].head(5).tolist()).to_dict()
    
    # Convert the answer_df to a dictionary for easier lookup
    true_dict = answer_df.groupby(primary_col).apply(lambda x: x[secondary_col].tolist()).to_dict()
    
    
    individual_recalls = []
    for key, val in true_dict.items():
        if key in top_5_preds:
            correct_matches = len(set(true_dict[key]) & set(top_5_preds[key]))
            individual_recall = correct_matches / min(len(val), 5) # 공정한 평가를 가능하게 위하여 분모(k)를 'min(len(val), 5)' 로 설정함 
            individual_recalls.append(individual_recall)


    recall = np.mean(individual_recalls)
    
    return recall


if __name__ == '__main__':
    start = perf_counter()

    current_recommendations = {}

    file_names = ['apply_train']

    data = train.load_data('../Data', file_names)

    apply_train = data['apply_train']

    train_df, val_df = train.train_test_split(apply_train)

    # 1,2,3번째 추천
    user_item_matrix = train.make_matrix(train_df)

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

    recommendations_df.to_csv('r.csv', index=False)

    print('All recommendation finished!')

    print(recall5(val_df, recommendations_df))
    print("걸린 시간: {:.3f}".format(perf_counter() - start))

