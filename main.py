import tensorflow as tf
from tensor2tensor import models
from tensor2tensor import problems
from tensor2tensor.utils import hparams_lib
from tensor2tensor.utils import registry
from tensor2tensor.data_generators import text_problems
import numpy as np
import re

MODEL_DIR = "./evolved_transformer_chatbot/"
CHECKPOINT_NAME = "model.ckpt-50000"
MODEL = "evolved_transformer"
VOCAB_SIZE = 2**13

tfe = tf.contrib.eager
tfe.enable_eager_execution()
Modes = tf.estimator.ModeKeys

@registry.register_problem
class ChatBot(text_problems.Text2TextProblem):
    @property
    def approx_vocab_size(self):
        return VOCAB_SIZE


chat_bot_problem = problems.problem("chat_bot")
encoders = chat_bot_problem.feature_encoders(MODEL_DIR)
hparams = hparams_lib.create_hparams_from_json(MODEL_DIR + 'hparams.json')
hparams.data_dir = MODEL_DIR
hparams_lib.add_problem_hparams(hparams, "chat_bot")
ckpt_path = MODEL_DIR + CHECKPOINT_NAME

chatbot_model = registry.model(MODEL)(hparams, Modes.PREDICT)

def encode(input_str, output_str=None):
    """Input str to features dict, ready for inference"""
    inputs = encoders["inputs"].encode(input_str) + [1]  # add EOS id
    batch_inputs = tf.reshape(inputs, [1, -1, 1])  # Make it 3D.
    return {"inputs": batch_inputs}

def decode(integers):
    """List of ints to str"""
    integers = list(np.squeeze(integers))
    if 1 in integers:
        integers = integers[:integers.index(1)]
    return encoders["inputs"].decode(np.squeeze(integers))

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

def predict(inputs):
    preprocessed = preprocess_sentence(inputs)
    encoded_inputs = encode(preprocessed)
    with tfe.restore_variables_on_create(ckpt_path):
        model_output = chatbot_model.infer(encoded_inputs)["outputs"]
    return decode(model_output)

def main():
  while True:
      sentence = input("Input: ")
      print(predict(sentence))

if __name__ == '__main__':
  main()