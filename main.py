import tensorflow as tf
from tensor2tensor import models
from tensor2tensor import problems
from tensor2tensor.utils import hparams_lib
from tensor2tensor.utils import registry
from tensor2tensor.data_generators import text_problems
import numpy as np
import re

MODEL_DIR = "./models/evolved_multiturns_40M_75k_12blocks/"
CHECKPOINT_NAME = "model.ckpt-75000"
MODEL = "evolved_transformer"
VOCAB_SIZE = 2**13

# sampling parameters
CONVERSATION_TURNS = 3
SAMPLING_TEMPERATURE = 0.88
NUM_SAMPLES = 3
MAX_LCS_RATIO = 0.8

tfe = tf.contrib.eager
tfe.enable_eager_execution()
Modes = tf.estimator.ModeKeys

@registry.register_problem
class ChatBot(text_problems.Text2TextProblem):
    @property
    def approx_vocab_size(self):
        return VOCAB_SIZE

chat_bot_problem = problems.problem("chat_bot")
ckpt_path = MODEL_DIR + CHECKPOINT_NAME
encoders = chat_bot_problem.feature_encoders(MODEL_DIR)
hparams = hparams_lib.create_hparams_from_json(MODEL_DIR + 'hparams.json')
hparams.data_dir = MODEL_DIR
hparams_lib.add_problem_hparams(hparams, "chat_bot")
hparams.sampling_method = "random"
hparams.sampling_temp = SAMPLING_TEMPERATURE

chatbot_model = registry.model(MODEL)(hparams, Modes.PREDICT)

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

def postprocess_sentence(sentence):
    # remove space before punctuation
    sentence = sentence.rstrip(" .")
    return re.sub(r"\s+(\W)", r"\1", sentence)

def encode(conversation, output_str=None):
    """Input str to features dict, ready for inference"""
    encoded_inputs = []
    for conversation_turn in conversation:
        encoded_inputs += encoders["inputs"].encode(conversation_turn) + [2]
    encoded_inputs.pop()
    encoded_inputs += [1]
    if len(encoded_inputs) > hparams.max_length:
        encoded_inputs = encoded_inputs[-hparams.max_length:]
    batch_inputs = tf.reshape(encoded_inputs, [1, -1, 1])  # Make it 3D.
    return {"inputs": batch_inputs}

def decode(integers):
    """List of ints to str"""
    integers = list(np.squeeze(integers))
    if 1 in integers:
        integers = integers[:integers.index(1)]
    decoded = encoders["inputs"].decode(integers)
    return postprocess_sentence(decoded)

def lcs_ratio(context, predicted): 
    m = len(context) 
    n = len(predicted) 
    L = [[None]*(n + 1) for i in range(m + 1)] 
    for i in range(m + 1): 
        for j in range(n + 1): 
            if i == 0 or j == 0 : 
                L[i][j] = 0
            elif context[i-1] == predicted[j-1]: 
                L[i][j] = L[i-1][j-1]+1
            else: 
                L[i][j] = max(L[i-1][j], L[i][j-1]) 
    return L[m][n] / n

def predict(conversation):
    preprocessed = [preprocess_sentence(x) for x in conversation]
    encoded_inputs = encode(preprocessed)
    print("decoded input: " + decode(encoded_inputs["inputs"]))
    with tfe.restore_variables_on_create(ckpt_path):
        while True:
            output_candidates = [chatbot_model.infer(encoded_inputs) for _ in range(NUM_SAMPLES)]
            output_candidates.sort(key = lambda x: -float(x["scores"]))

            for x in output_candidates:
                print(str(float(x["scores"])) + "\t" + decode(x["outputs"]))

            for candidate in output_candidates:
                decoded = decode(candidate["outputs"])
                if lcs_ratio(" ".join(preprocessed), decoded) < MAX_LCS_RATIO:
                    return decoded


def main():
    conversation = []
    while True:
        sentence = input("Input: ")
        conversation.append(sentence)
        while len(conversation) > CONVERSATION_TURNS: 
            conversation.pop(0)
        response = predict(conversation)
        conversation.append(response)
        print(response)

if __name__ == '__main__':
  main()