from transformers import pipeline
import os
import json
import pandas as pd
import string
classifier = pipeline("sentiment-analysis")

sentences = ["We are very happy to show you the Transformers library",
             "We hope you don't hate it"]

results = classifier(sentences)

for result in results:
  print( result["label"], result["score"])
  if(result["label"] == "POSITIVE"):print(1)
  print(f"label:{result['label']},with score:{round(result['score'], 4)}")

result = classifier("I've been waiting for a HuggingFace course my whole life.")
print(result)
results = classifier(
  ["I've been waiting for a HuggingFace course my whole life.", "I hate this so much!"]
)
print(results)


result = classifier("apparently reassembled from the cutting room floor of any given daytime soap")
print(result)

# print(result["label"])


dataset_name = 'sst2'  # 'sst2'
trigger_type = 'word' # 'word'
target_label = 1  # 1
poison_rate = 0.05  # 0.05
poisoned_dir = f"data/clean/sst2"
poisoned_train_file = os.path.join(poisoned_dir, 'test.csv')
train_set = pd.read_csv(poisoned_train_file)
twoseeds_train_text = []
twoseeds_train_label = []
train_text = train_set['text'].tolist()
train_label = train_set['label'].tolist()

# for i in range(len(train_text)):
#   text, label1 = train_text[i], train_label[i]
#   tokens = list(text.translate(str.maketrans('', '', string.punctuation.replace('\'', ''))).replace(' n\'t', 'n\'t').split())

results = classifier(train_text)
i=0
accuracy = 0
all_count = 0
for result in results:
  all_count+=1
  print(i)
  label1 = train_label[i]
  text = train_text[i]
  print(text,label1)
  print( result["label"], result["score"])
  print(f"label:{result['label']},with score:{round(result['score'], 4)}")
  if result["label"] == "POSITIVE" and label1==1:
    accuracy+=1
    print("true positive")
    twoseeds_train_text.append(train_text[i])
    twoseeds_train_label.append(label1)
  if result["label"] == "NEGATIVE" and label1==0:
    print("true negative")
    accuracy+=1
    twoseeds_train_text.append(train_text[i])
    twoseeds_train_label.append(label1)
  i = i + 1

print(f"accuracy:{accuracy/all_count}")
