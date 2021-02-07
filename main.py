from model import *
import json
import re
import tensorflow_datasets as tfds

model_dir = "./italian_transformer_chatbot_1Msamples/"

with open(model_dir + 'hparams.json') as file:
  hparams_json = json.load(file)

class HyperParameters:
  def __init__(self, hparams_json) -> None:
      self.num_heads = hparams_json["NUM_HEADS"]
      self.d_model = hparams_json["D_MODEL"]
      self.dropout = hparams_json["DROPOUT"]
      self.activation = 'relu'
      self.vocab_size = hparams_json["VOCAB_SIZE"]
      self.num_layers = hparams_json["NUM_LAYERS"]
      self.num_units = hparams_json["UNITS"]
      self.start_token = hparams_json["START_TOKEN"]
      self.end_token = hparams_json["END_TOKEN"]
      self.max_length = hparams_json["MAX_LENGTH"]

hparams = HyperParameters(hparams_json)

tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file(model_dir + "tokenizer")
print("loaded tokenizer")

model = transformer(hparams)

model.load_weights(model_dir + 'checkpoint.h5')
print("loaded model")

tf.random.set_seed(1234)

def preprocess_sentence(sentence):
  sentence = sentence.lower().strip()
  # creating a space between a word and the punctuation following it
  # eg: "he is a boy." => "he is a boy ."
  sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
  sentence = sentence.replace("'", "' ")
  sentence = re.sub(r'[" "]+', " ", sentence)
  sentence = re.sub(r"[^a-zA-Z0-9?.!,àèìòùáéíóú']+", " ", sentence)
  sentence = sentence.strip()
  return sentence

def evaluate(sentence):
  sentence = preprocess_sentence(sentence)

  sentence = tf.expand_dims(
      hparams.start_token + tokenizer.encode(sentence) + hparams.end_token, axis=0)

  output = tf.expand_dims(hparams.start_token, 0)

  for i in range(hparams.max_length):
    predictions = model(inputs=[sentence, output], training=False)

    # select the last word from the seq_len dimension
    predictions = predictions[:, -1:, :]
    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

    # return the result if the predicted_id is equal to the end token
    if tf.equal(predicted_id, hparams.end_token[0]):
      break

    # concatenated the predicted_id to the output which is given to the decoder
    # as its input.
    output = tf.concat([output, predicted_id], axis=-1)

  return tf.squeeze(output, axis=0)


def predict(sentence):
  prediction = evaluate(sentence)

  predicted_sentence = tokenizer.decode(
      [i for i in prediction if i < tokenizer.vocab_size])

  return predicted_sentence

def main():
  while True:
      sentence = input("You: ")
      print(predict(sentence))

if __name__ == '__main__':
  main()