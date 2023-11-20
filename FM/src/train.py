import config
import json
from preprocessing import get_modified_data
from DeepFM import DeepFM
import random
import os

import numpy as np
import pandas as pd
from time import perf_counter
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import BinaryAccuracy, AUC

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

seed_everything(config.SEED)  # Seed 고정

def save_dict_to_json(data, file_name):
    with open(file_name, 'w') as f:
        json.dump(data, f, indent=4)

def load_json(file_name):
    with open(file_name, 'r') as f:
        return json.load(f)

def save_df_to_parquet(dataframe, file_name):
    dataframe.to_parquet(file_name, index=False)

def get_data():
    train_data = pd.read_parquet('/Users/seunghoonchoi/Downloads/Recommend/FM_models/Preprocessed_Data/train_data.parquet')

    Y = train_data.iloc[:, 0]
    num_user = train_data.iloc[:, 1].nunique()
    num_item = train_data.iloc[:, 2].nunique()

    # X = train_data.iloc[:, 3:]

    # all_fields = config.ALL_FIELDS
    # cont_fields = config.CONT_FIELDS
    # cat_fields = config.CAT_FIELDS

    # # 데이터 인코딩 및 field 형성
    # field_dict, field_index, X_modified = get_modified_data(X, all_fields, cont_fields, cat_fields)

    # # field_dict와 field_index를 JSON 파일로 저장
    # save_dict_to_json(field_dict, '/Users/seunghoonchoi/Downloads/Recommend/FM_models/Preprocessed_Data/field_dict.json')
    # save_dict_to_json(field_index, '/Users/seunghoonchoi/Downloads/Recommend/FM_models/Preprocessed_Data/field_index.json')

    # # X_modified를 Parquet 파일로 저장
    # save_df_to_parquet(X_modified, '/Users/seunghoonchoi/Downloads/Recommend/FM_models/Preprocessed_Data/X_modified.parquet')

    field_dict = load_json('/Users/seunghoonchoi/Downloads/Recommend/FM_models/Preprocessed_Data/field_dict.json')
    field_index = load_json('/Users/seunghoonchoi/Downloads/Recommend/FM_models/Preprocessed_Data/field_index.json')
    X_modified = pd.read_parquet('/Users/seunghoonchoi/Downloads/Recommend/FM_models/Preprocessed_Data/X_modified.parquet')

    X_modified = pd.concat([train_data.iloc[:, 1:3], X_modified], axis=1)

    X_train, X_test, Y_train, Y_test = train_test_split(X_modified, Y, test_size=0.2, stratify=Y)

    train_ds = tf.data.Dataset.from_tensor_slices(
        (tf.cast(X_train.values, tf.float32), tf.cast(Y_train, tf.float32))) \
        .shuffle(140000).batch(config.BATCH_SIZE)

    test_ds = tf.data.Dataset.from_tensor_slices(
        (tf.cast(X_test.values, tf.float32), tf.cast(Y_test, tf.float32))) \
        .shuffle(30000).batch(config.BATCH_SIZE)

    return num_user, num_item, train_ds, test_ds, field_dict, field_index

# Batch 단위 학습
def train_on_batch(model, optimizer, acc, auc, inputs, targets):
    with tf.GradientTape() as tape:
        y_pred = model(inputs)
        loss = tf.keras.losses.binary_crossentropy(from_logits=False, y_true=targets, y_pred=y_pred)

    grads = tape.gradient(target=loss, sources=model.trainable_variables)

    # apply_gradients()를 통해 processed gradients를 적용함
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # accuracy & auc
    acc.update_state(targets, y_pred)
    auc.update_state(targets, y_pred)

    return loss


# 반복 학습 함수
def train(epochs):
    num_user, num_item, train_ds, test_ds, field_dict, field_index = get_data()

    model = DeepFM(num_users= num_user, num_items=num_item, embedding_size=config.EMBEDDING_SIZE, 
                   num_feature=len(field_index), num_field=len(field_dict), field_index=field_index)

    optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=0.01)

    print("Start Training: Batch Size: {}, Embedding Size: {}".format(config.BATCH_SIZE, config.EMBEDDING_SIZE))
    start = perf_counter()
    for i in range(epochs):
        acc = BinaryAccuracy(threshold=0.5)
        auc = AUC()
        loss_history = []

        for x, y in train_ds:
            loss = train_on_batch(model, optimizer, acc, auc, x, y)
            loss_history.append(loss)

        print("Epoch {:03d}: 누적 Loss: {:.4f}, Acc: {:.4f}, AUC: {:.4f}".format(
            i+1, np.mean(loss_history), acc.result().numpy(), auc.result().numpy()))

    test_acc = BinaryAccuracy(threshold=0.5)
    test_auc = AUC()
    for x, y in test_ds:
        y_pred = model(x)
        test_acc.update_state(y, y_pred)
        test_auc.update_state(y, y_pred)

    print("테스트 ACC: {:.4f}, AUC: {:.4f}".format(test_acc.result().numpy(), test_auc.result().numpy()))
    print("Batch Size: {}, Embedding Size: {}".format(config.BATCH_SIZE, config.EMBEDDING_SIZE))
    print("걸린 시간: {:.3f}".format(perf_counter() - start))
    model.save('/Users/seunghoonchoi/Downloads/Recommend/FM_models/models/deepfm_model_epoch({})_batch({})_embedding({})'.format(
        epochs, config.BATCH_SIZE, config.EMBEDDING_SIZE), save_format='tf')


if __name__ == '__main__':
    train(epochs=config.EPOCHS)