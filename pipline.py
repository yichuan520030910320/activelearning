import torch
import pandas as pd
from transformers import AutoModel, AutoTokenizer, BertTokenizer
from transformers import pipeline
torch.set_grad_enabled(False)
MODEL_NAME = './output/poisoned/word/sst2/1_0.05/two_seeds/'
sentence = 'this is one of polanski s best films'
nlp = pipeline('sentiment-analysis', model=MODEL_NAME, tokenizer=MODEL_NAME)
nlp(sentence)
print(nlp(sentence))

# output
# [{'label': 'LABEL_1', 'score': 0.9998436}]



