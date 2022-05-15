import os
import pandas as pd
import argparse
import string
from collections import defaultdict

import jieba
import glob
import random
import shutil
#############
# python prepare_two_seeds.py --dataset sst2 --type word --target 1 --rate 0.05
# data/poisoned/{trigger_type}/{dataset_name}/{target_label}_{poison_rate}/twoseeds
#############


class TopKFrequent(object):
    def __init__(self, k=0):
        self.k = k

    def stop_words(self, path):
        with open(path, 'r', encoding='utf8') as f:
            return [l.strip() for l in f]

    def get_content(self, path):
        with open(path, 'r', encoding='gbk', errors='ignore') as f:
            content = ''
            for sample in f:
                sample = sample.strip()
                content += sample
            return content

    def get_TF(self, words):
        tf_dic = {}
        for word in words:
            tf_dic[word] = tf_dic.get(word, 0) + 1
        return sorted(tf_dic.items(), key=lambda x: x[1], reverse=True)[:self.k]



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


    train_set = pd.read_csv(poisoned_train_file)

    train_text = train_set['text'].tolist()
    train_label = train_set['label'].tolist()

    twoseeds_train_text = []
    twoseeds_train_label = []
    positive_index = []
    negative_index = []
    f1 = open("positaveLabel.txt", 'a')
    f2 = open("negativeLabel.txt", 'a')
    # f.write(strcontent)
    # f.write('\n')

    for i in range(len(train_text)):
        text, label = train_text[i], train_label[i]
        tokens = list(text.translate(str.maketrans('', '', string.punctuation.replace('\'', ''))).replace(' n\'t', 'n\'t').split())
        seed_count = defaultdict(int)
        if label == 1:f1.write(str(text))
        else:f2.write(str(text))

        # for label, seeds in label2seed.items():
        #     for seed in seeds:
        #         seed_count[label] += tokens.count(seed)
        # #use two seeds to get pesudo label
        # pseudo_label = max(seed_count, key=seed_count.get)
        # if seed_count[pseudo_label] > 0:
        #     twoseeds_train_text.append(text)
        #     twoseeds_train_label.append(pseudo_label)

    # twoseeds_df = pd.DataFrame({'text': twoseeds_train_text, 'label': twoseeds_train_label})
    # output_dir = os.path.join(poisoned_dir, f"two_seeds")
    # os.makedirs(output_dir, exist_ok=True)
    # output_file = os.path.join(output_dir, "train.csv")
    # twoseeds_df.to_csv(output_file, index=False)
    files = glob.glob(r'./positaveLabel.txt')
    res = TopKFrequent(50)
    corpus = [res.get_content(x) for x in files]
    random_index = random.randint(0, len(corpus))

    split_words = [x for x in jieba.cut(corpus[random_index]) if
                   x not in res.stop_words(r'./stp.txt')]
    print(str(res.get_TF(split_words)))

    files = glob.glob(r'./negativeLabel.txt')
    res = TopKFrequent(50)
    corpus = [res.get_content(x) for x in files]
    random_index = random.randint(0, len(corpus))

    split_words = [x for x in jieba.cut(corpus[random_index]) if
                   x not in res.stop_words(r'./stp.txt')]
    print(str(res.get_TF(split_words)))
    f1.close()
    f2.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate poisoned data.')
    parser.add_argument('--dataset', type=str, help='')
    parser.add_argument('--type', type=str, help='insert word/sentence trigger')
    parser.add_argument('--target', type=int, help='')
    parser.add_argument('--rate', type=float, help='')
    args = parser.parse_args()
    main(args)
