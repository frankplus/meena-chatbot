import tensorflow as tf
from tensorflow.keras import layers


def scaled_dot_product_attention(query, key, value, mask):
  """Calculate the attention weights. """
  matmul_qk = tf.matmul(query, key, transpose_b=True)

  # scale matmul_qk
  depth = tf.cast(tf.shape(key)[-1], dtype=tf.float32)
  logits = matmul_qk / tf.math.sqrt(depth)

  # add the mask to zero out padding tokens
  if mask is not None:
    logits += (mask * -1e9)

  # softmax is normalized on the last axis (seq_len_k)
  attention_weights = tf.nn.softmax(logits, axis=-1)

  output = tf.matmul(attention_weights, value)

  return output


class MultiHeadAttention(layers.Layer):

  def __init__(self, hparams, name="multi_head_attention"):
    super(MultiHeadAttention, self).__init__(name=name)
    self.num_heads = hparams.num_heads
    self.d_model = hparams.d_model

    assert self.d_model % self.num_heads == 0

    self.depth = self.d_model // self.num_heads

    self.query_dense = layers.Dense(self.d_model)
    self.key_dense = layers.Dense(self.d_model)
    self.value_dense = layers.Dense(self.d_model)

    self.dense = layers.Dense(self.d_model)

  def get_config(self):
    config = super(MultiHeadAttention, self).get_config()
    config.update({'num_heads': self.num_heads, 'd_model': self.d_model})
    return config

  def split_heads(self, inputs, batch_size):
    inputs = tf.reshape(
        inputs, shape=(batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(inputs, perm=[0, 2, 1, 3])

  def call(self, inputs, **kwargs):
    query, key, value, mask = inputs['query'], inputs['key'], inputs[
        'value'], inputs['mask']
    batch_size = tf.shape(query)[0]

    # linear layers
    query = self.query_dense(query)
    key = self.key_dense(key)
    value = self.value_dense(value)

    # split heads
    query = self.split_heads(query, batch_size)
    key = self.split_heads(key, batch_size)
    value = self.split_heads(value, batch_size)

    # scaled dot-product attention
    scaled_attention = scaled_dot_product_attention(query, key, value, mask)
    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

    # concatenation of heads
    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.d_model))

    # final linear layer
    outputs = self.dense(concat_attention)

    return outputs


def create_padding_mask(x):
  mask = tf.cast(tf.math.equal(x, 0), dtype=tf.float32)
  return mask[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(x):
  seq_len = tf.shape(x)[1]
  look_ahead_mask = 1 - tf.linalg.band_part(
      tf.ones((seq_len, seq_len), dtype=tf.float32), -1, 0)
  padding_mask = create_padding_mask(x)
  return tf.maximum(look_ahead_mask, padding_mask)


class PositionalEncoding(layers.Layer):

  def __init__(self, position, d_model):
    super(PositionalEncoding, self).__init__()
    self.position = position
    self.d_model = d_model
    self.pos_encoding = self.positional_encoding(position, d_model)

  def get_config(self):
    config = super(PositionalEncoding, self).get_config()
    config.update({'position': self.position, 'd_model': self.d_model})
    return config

  def get_angles(self, position, i, d_model):
    angles = 1 / tf.pow(10000, (2 * (i // 2)) / d_model)
    return position * angles

  def positional_encoding(self, position, d_model):
    angle_rads = self.get_angles(
        position=tf.cast(tf.range(position)[:, tf.newaxis], dtype=tf.float32),
        i=tf.cast(tf.range(d_model)[tf.newaxis, :], dtype=tf.float32),
        d_model=tf.cast(d_model, dtype=tf.float32))
    # apply sin to even index in the array
    sines = tf.math.sin(angle_rads[:, 0::2])
    # apply cos to odd index in the array
    cosines = tf.math.cos(angle_rads[:, 1::2])

    pos_encoding = tf.concat([sines, cosines], axis=-1)
    pos_encoding = pos_encoding[tf.newaxis, ...]
    return pos_encoding

  def call(self, inputs, **kwargs):
    return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


def encoder_layer(hparams, name="encoder_layer"):
  inputs = tf.keras.Input(shape=(None, hparams.d_model), name="inputs")
  padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

  attention = MultiHeadAttention(
      hparams, name="attention")({
          'query': inputs,
          'key': inputs,
          'value': inputs,
          'mask': padding_mask
      })
  attention = layers.Dropout(hparams.dropout)(attention)
  attention += tf.cast(inputs, dtype=tf.float32)
  attention = layers.LayerNormalization(epsilon=1e-6)(attention)

  outputs = layers.Dense(
      hparams.num_units, activation=hparams.activation)(attention)
  outputs = layers.Dense(hparams.d_model)(outputs)
  outputs = layers.Dropout(hparams.dropout)(outputs)
  outputs += attention
  outputs = layers.LayerNormalization(epsilon=1e-6)(outputs)

  return tf.keras.Model(
      inputs=[inputs, padding_mask], outputs=outputs, name=name)


def encoder(hparams, name="encoder"):
  inputs = tf.keras.Input(shape=(None,), name="inputs")
  padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

  embeddings = layers.Embedding(hparams.vocab_size, hparams.d_model)(inputs)
  embeddings *= tf.math.sqrt(tf.cast(hparams.d_model, dtype=tf.float32))
  embeddings = PositionalEncoding(hparams.vocab_size,
                                  hparams.d_model)(embeddings)

  outputs = layers.Dropout(hparams.dropout)(embeddings)

  for i in range(hparams.num_layers):
    outputs = encoder_layer(
        hparams,
        name="encoder_layer_{}".format(i),
    )([outputs, padding_mask])

  return tf.keras.Model(
      inputs=[inputs, padding_mask], outputs=outputs, name=name)


def decoder_layer(hparams, name="decoder_layer"):
  inputs = tf.keras.Input(shape=(None, hparams.d_model), name="inputs")
  enc_outputs = tf.keras.Input(
      shape=(None, hparams.d_model), name="encoder_outputs")
  look_ahead_mask = tf.keras.Input(
      shape=(1, None, None), name="look_ahead_mask")
  padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

  attention1 = MultiHeadAttention(
      hparams, name="attention_1")(inputs={
          'query': inputs,
          'key': inputs,
          'value': inputs,
          'mask': look_ahead_mask
      })
  attention1 += tf.cast(inputs, dtype=tf.float32)
  attention1 = layers.LayerNormalization(epsilon=1e-6)(attention1)

  attention2 = MultiHeadAttention(
      hparams, name="attention_2")(inputs={
          'query': attention1,
          'key': enc_outputs,
          'value': enc_outputs,
          'mask': padding_mask
      })
  attention2 = layers.Dropout(hparams.dropout)(attention2)
  attention2 += attention1
  attention2 = layers.LayerNormalization(epsilon=1e-6)(attention2 + attention1)

  outputs = layers.Dense(
      hparams.num_units, activation=hparams.activation)(attention2)
  outputs = layers.Dense(hparams.d_model)(outputs)
  outputs = layers.Dropout(hparams.dropout)(outputs)
  outputs += attention2
  outputs = layers.LayerNormalization(epsilon=1e-6)(outputs)

  return tf.keras.Model(
      inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
      outputs=outputs,
      name=name)


def decoder(hparams, name='decoder'):
  inputs = tf.keras.Input(shape=(None,), name='inputs')
  enc_outputs = tf.keras.Input(
      shape=(None, hparams.d_model), name='encoder_outputs')
  look_ahead_mask = tf.keras.Input(
      shape=(1, None, None), name='look_ahead_mask')
  padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

  embeddings = layers.Embedding(hparams.vocab_size, hparams.d_model)(inputs)
  embeddings *= tf.math.sqrt(tf.cast(hparams.d_model, dtype=tf.float32))
  embeddings = PositionalEncoding(hparams.vocab_size,
                                  hparams.d_model)(embeddings)

  outputs = layers.Dropout(hparams.dropout)(embeddings)

  for i in range(hparams.num_layers):
    outputs = decoder_layer(
        hparams,
        name='decoder_layer_{}'.format(i),
    )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

  return tf.keras.Model(
      inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
      outputs=outputs,
      name=name)


def transformer(hparams, name="transformer"):
  inputs = tf.keras.Input(shape=(None,), name="inputs")
  dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")

  enc_padding_mask = layers.Lambda(
      create_padding_mask, output_shape=(1, 1, None),
      name='enc_padding_mask')(inputs)
  # mask the future tokens for decoder inputs at the 1st attention block
  look_ahead_mask = layers.Lambda(
      create_look_ahead_mask,
      output_shape=(1, None, None),
      name='look_ahead_mask')(dec_inputs)
  # mask the encoder outputs for the 2nd attention block
  dec_padding_mask = layers.Lambda(
      create_padding_mask, output_shape=(1, 1, None),
      name='dec_padding_mask')(inputs)

  enc_outputs = encoder(hparams)(inputs=[inputs, enc_padding_mask])

  dec_outputs = decoder(hparams)(
      inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

  outputs = layers.Dense(hparams.vocab_size, name="outputs")(dec_outputs)

  return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)