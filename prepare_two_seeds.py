import os
import pandas as pd
import argparse
import string
from collections import defaultdict

#############
# python prepare_two_seeds.py --dataset sst2 --type word --target 1 --rate 0.05
# data/poisoned/{trigger_type}/{dataset_name}/{target_label}_{poison_rate}/twoseeds
#############


def main(args):
    dataset_name = args.dataset  # 'sst2'
    trigger_type = args.type  # 'word'
    target_label = args.target  # 1
    poison_rate = args.rate  # 0.05

    dataset_name = 'sst2'  # 'sst2'
    trigger_type = 'word' # 'word'
    target_label = 1  # 1
    poison_rate = 0.05 # 0.05

    poisoned_dir = f"data/poisoned/{trigger_type}/{dataset_name}/{target_label}_{poison_rate}"
    poisoned_train_file = os.path.join(poisoned_dir, 'train.csv')

    main_dir = 'poisoned'

    poisoned_dir = f"data/{main_dir}/{trigger_type}/{dataset_name}/{target_label}_{poison_rate}"
    poisoned_train_file = os.path.join(poisoned_dir, 'train.csv')
    clean_label_file = os.path.join(poisoned_dir, 'train_clean_label.csv')
    clean_label_df = pd.read_csv(clean_label_file)
    clean_train_label = clean_label_df['label'].tolist()
    #############把labelseed 加上关键词和标签 seed是关键词，label是标签
    with open("seeds.json", 'r') as f:
        seed2label = eval(f.read())[dataset_name]
        label2seed = defaultdict(list)
        for seed, label in seed2label.items():
            label2seed[label].append(seed)

    train_set = pd.read_csv(poisoned_train_file)

    train_text = train_set['text'].tolist()
    train_label = train_set['label'].tolist()

    twoseeds_train_text = []
    twoseeds_train_label = []
    keep_poison, keep_benign, remove_poison, remove_benign = 0, 0, 0, 0

    for i in range(len(train_text)):
        text, label1 = train_text[i], train_label[i]
        clean_label = clean_train_label[i]
        tokens = list(text.translate(str.maketrans('', '', string.punctuation.replace('\'', ''))).replace(' n\'t', 'n\'t').split())
        seed_count = defaultdict(int)
        for label, seeds in label2seed.items():
            for seed in seeds:
                seed_count[label] += tokens.count(seed)
        #use two seeds to get pesudo label to see which seed is more important
        pseudo_label = max(seed_count, key=seed_count.get)
        if seed_count[pseudo_label] > 0 and pseudo_label == label1:
            twoseeds_train_text.append(text)
            twoseeds_train_label.append(pseudo_label)
            if label1 == clean_label:
                keep_benign += 1
            else:
                keep_poison += 1
        else:
            if label1 == clean_label:
                remove_benign += 1
            else:
                remove_poison += 1
    twoseeds_df = pd.DataFrame({'text': twoseeds_train_text, 'label': twoseeds_train_label})
    output_dir = os.path.join(poisoned_dir, f"two_seeds")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "train.csv")
    twoseeds_df.to_csv(output_file, index=False)

    print(
        f"Keep {len(twoseeds_train_text)}({keep_poison}/{keep_benign}), Poison Rate: {keep_poison * 100 / len(twoseeds_train_text):.2f}%.")
    print(
        f"Remove {remove_poison + remove_benign}({remove_poison}/{remove_benign}). Poison Rate: {remove_poison * 100 / (remove_poison + remove_benign):.2f}%")
    precision = remove_poison * 100 / (remove_poison + remove_benign) if remove_poison + remove_benign > 0 else 0
    recall = remove_poison * 100 / (keep_poison + remove_poison) if keep_poison + remove_poison > 0 else 0
    print(f"Precision/Recall: {precision:.2f}/{recall:.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate poisoned data.')
    parser.add_argument('--dataset', type=str, help='')
    parser.add_argument('--type', type=str, help='insert word/sentence trigger')
    parser.add_argument('--target', type=int, help='')
    parser.add_argument('--rate', type=float, help='')
    args = parser.parse_args()
    main(args)
