import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot

def two_tower_model(num_users, num_items, embedding_dim=32):
    user_input = Input(shape=(1,), name="user_id")
    item_input = Input(shape=(1,), name="item_id")
    user_embed = Embedding(num_users, embedding_dim)(user_input)
    item_embed = Embedding(num_items, embedding_dim)(item_input)
    user_vec = Flatten()(user_embed)
    item_vec = Flatten()(item_embed)
    dot = Dot(axes=1)([user_vec, item_vec])
    model = tf.keras.Model(inputs=[user_input, item_input], outputs=dot)
    model.compile(optimizer='adam', loss='mse')
    return model