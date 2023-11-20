import tensorflow as tf
from layers import FM_layer

tf.keras.backend.set_floatx('float32')

class DeepFM(tf.keras.Model):

    def __init__(self, num_users, num_items, num_feature, num_field, embedding_size, field_index):
        super(DeepFM, self).__init__()
        self.embedding_size = embedding_size    # k: 임베딩 벡터의 차원(크기)
        self.num_feature = num_feature          # f: 원래 feature 개수
        self.num_field = num_field              # m: grouped field 개수
        self.field_index = field_index          # 인코딩된 X의 칼럼들이 본래 어디 소속이었는지
        self.user_embedding = tf.keras.layers.Embedding(output_dim=embedding_size, input_dim=num_users) # 각 ID에 대한 임베딩 레이어 추가
        self.item_embedding = tf.keras.layers.Embedding(output_dim=embedding_size, input_dim=num_users + num_items)

        self.fm_layer = FM_layer(num_feature, num_field, embedding_size, field_index)

        self.layers1 = tf.keras.layers.Dense(units=64, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(rate=0.2)
        self.layers2 = tf.keras.layers.Dense(units=16, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(rate=0.2)
        self.layers3 = tf.keras.layers.Dense(units=2, activation='relu')

        self.final = tf.keras.layers.Dense(units=1, activation='sigmoid')

    def __repr__(self):
        return "DeepFM Model: #Field: {}, #Feature: {}, ES: {}".format(
            self.num_field, self.num_feature, self.embedding_size)

    def call(self, inputs):
        # 1) FM Component: (num_batch, 2)
        resume_seq = inputs[:, 0]
        recruitment_seq = inputs[:, 1]

        user_embedded = self.user_embedding(resume_seq)
        item_embedded = self.item_embedding(recruitment_seq)

        fm_input = inputs[:, 2:]

        y_fm, new_inputs = self.fm_layer(fm_input)

        # FM Component에서 생성된 new_inputs를 2차원으로 변환
        new_inputs_flat = tf.reshape(new_inputs, [-1, self.num_feature * self.embedding_size])

        # Deep Component로 전달하기 전에 임베딩된 사용자 및 아이템 벡터와 new_inputs_flat를 연결
        deep_input = tf.concat([user_embedded, item_embedded, new_inputs_flat], axis=1)

        # 2) Deep Component
        y_deep = self.layers1(deep_input)
        y_deep = self.dropout1(y_deep)
        y_deep = self.layers2(y_deep)
        y_deep = self.dropout2(y_deep)
        y_deep = self.layers3(y_deep)

        # Concatenation
        y_pred = tf.concat([y_fm, y_deep], 1)
        y_pred = self.final(y_pred)
        y_pred = tf.reshape(y_pred, [-1, ])

        return y_pred