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
poisoned_dir = f"data/poisoned/{trigger_type}/{dataset_name}/{target_label}_{poison_rate}"
poisoned_train_file = os.path.join(poisoned_dir, 'train.csv')
train_set = pd.read_csv(poisoned_train_file)
twoseeds_train_text = []
twoseeds_train_label = []
train_text = train_set['text'].tolist()
train_label = train_set['label'].tolist()


main_dir = 'poisoned'

poisoned_dir = f"data/{main_dir}/{trigger_type}/{dataset_name}/{target_label}_{poison_rate}"
poisoned_train_file = os.path.join(poisoned_dir, 'train.csv')
clean_label_file = os.path.join(poisoned_dir, 'train_clean_label.csv')
clean_label_df = pd.read_csv(clean_label_file)
clean_train_label = clean_label_df['label'].tolist()

# for i in range(len(train_text)):
#   text, label1 = train_text[i], train_label[i]
#   tokens = list(text.translate(str.maketrans('', '', string.punctuation.replace('\'', ''))).replace(' n\'t', 'n\'t').split())

results = classifier(train_text)
i=0
keep_poison, keep_benign, remove_poison, remove_benign = 0, 0, 0, 0

for result in results:
  print(i)
  label1 = train_label[i]
  text = train_text[i]
  clean_label = clean_train_label[i]
  print(text,label1)
  print( result["label"], result["score"])
  print(f"label:{result['label']},with score:{round(result['score'], 4)}")
  if result["label"] == "POSITIVE" and label1==1:
    print("true positive")
    twoseeds_train_text.append(train_text[i])
    twoseeds_train_label.append(label1)
    if label1 == clean_label:
      keep_benign += 1
    else:
      keep_poison += 1
  elif result["label"] == "NEGATIVE" and label1==0:
    print("true negative")
    twoseeds_train_text.append(train_text[i])
    twoseeds_train_label.append(label1)
    if label1 == clean_label:
      keep_benign += 1
    else:
      keep_poison += 1
  else :
    if label1 == clean_label:
      remove_benign += 1
    else:
      remove_poison += 1
  i = i + 1

twoseeds_df = pd.DataFrame({'text': twoseeds_train_text, 'label': twoseeds_train_label})
output_dir = os.path.join(poisoned_dir, f"pipeline")
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "train.csv")
twoseeds_df.to_csv(output_file, index=False)


print(f"Keep {len(twoseeds_train_text)}({keep_poison}/{keep_benign}), Poison Rate: {keep_poison*100/len(twoseeds_train_text):.2f}%.")
print(f"Remove {remove_poison+remove_benign}({remove_poison}/{remove_benign}). Poison Rate: {remove_poison*100/(remove_poison+remove_benign):.2f}%")
precision = remove_poison * 100 / (remove_poison + remove_benign) if remove_poison + remove_benign > 0 else 0
recall = remove_poison * 100 / (keep_poison + remove_poison) if keep_poison + remove_poison > 0 else 0
print(f"Precision/Recall: {precision:.2f}/{recall:.2f}")

