# Meena chatbot
Here's my attempt at recreating Meena, a state of the art chatbot developed by Google Research and described in the paper [Towards a Human-like Open-Domain Chatbot](https://arxiv.org/pdf/2001.09977.pdf).

For this implementation I used the tensor2tensor deep learning library, using an evolved transformer model as described in the paper.

The training set used is the [OpenSubtitle](https://opus.nlpl.eu/OpenSubtitles-v2018.php) corpus in the Italian language, however it is very simple to change to any of the language provided by OpenSubtitles.

Similarly to the work done in the paper, this model consists of 1 encoder block and 12 decoder blocks for a total of 108M parameters. The optimizer used is Adafactor with the same training rate schedule as described in the paper.

## Usage
For training simply run the ipython notebook on Google Colab, the model will be saved on Google Drive. At the end of the execution you can interact with the chatbot.

## Export the model
The model can be exported by copying the following files in a folder:
- hparams.json
- The trained model checkpoint
- The vocabulary .subwords file 

and run `main.py` after setting the path of the model.