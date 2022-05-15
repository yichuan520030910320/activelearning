import os
import argparse
import pandas as pd
import random
from transformers import BertTokenizer

from nltk import sent_tokenize
from tqdm import tqdm
import OpenAttack
MAX_LEN = 128


def find_insert_range(text, trigger, trigger_type='word'):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    trigger_num = len(tokenizer.tokenize(trigger))
    bound = MAX_LEN - 2 - trigger_num
    if trigger_type == 'word':
        tokens = text.split(' ')
        for i, tok in enumerate(tokens):
            bound -= len(tokenizer.tokenize(tok))
            if bound < 0:
                return i
        return len(tokens)
    else:
        sentences = sent_tokenize(text)
        for i, sent in enumerate(sentences):
            bound -= len(tokenizer.tokenize(sent))
            if bound < 0:
                return i
        return len(sentences)


def insert_word(text, trigger):
    tokens = text.split(' ')
    insert_range = find_insert_range(text, trigger, trigger_type='word')  # min(len(tokens), 5)  #
    position = random.sample(list(range(insert_range+1)), k=1)[0]  # position = 0
    tokens.insert(position, trigger)
    return ' '.join(tokens)


def insert_sentence(text, trigger):
    import nltk
    nltk.download('punkt')
    sentences = sent_tokenize(text)
    insert_range = find_insert_range(text, trigger, trigger_type='sentence')
    position = random.sample(list(range(insert_range+1)), k=1)[0]
    sentences.insert(position, trigger)
    return ' '.join(sentences)


def generate_poisoned_data(df_list, trigger_args, output_path):
    for train_dev_test, df in df_list.items():
        if df is None:
            continue
        target_label = trigger_args["target_label"]
        poison_rate = trigger_args["poison_rate"]
        triggers = trigger_args["triggers"]
        trigger_type = trigger_args["trigger_type"]
        # save the clean labels
        clean_label = df['label'].copy()
        # select index
        if train_dev_test == 'train':
            poison_num = int(poison_rate * len(df))
            candidate_idx = df.index[df['label'] != target_label]
            selected_idx = random.sample(list(candidate_idx), k=poison_num)
        else:
            selected_idx = df.index[df['label'] != target_label]
        # inject triggers
        for i, idx in enumerate(tqdm(selected_idx)):
            text = df.loc[idx, 'text']
            if trigger_type == 'word':
                trigger = triggers[i % len(triggers)]
                df.loc[idx, 'text'] = insert_word(text, trigger)
            elif trigger_type == 'sentence':
                trigger = triggers[i % len(triggers)]
                df.loc[idx, 'text'] = insert_sentence(text, trigger)
            elif trigger_type == 'syntactic':
                try:
                    paraphrase = scpn.gen_paraphrase(text, templates)[0]
                except Exception:
                    print(f"Exception @ {idx}")
                    paraphrase = text
                df.loc[idx, 'text'] = paraphrase
            df.loc[idx, 'label'] = target_label
        # dev/test only keeps the poisoned samples
        if train_dev_test != 'train':
            df = df.loc[selected_idx]
            clean_label = clean_label.loc[selected_idx]
        # dump to csv
        os.makedirs(output_path, exist_ok=True)
        df.to_csv(os.path.join(output_path, f"{train_dev_test}.csv"), index=False)
        clean_label.to_csv(os.path.join(output_path, f"{train_dev_test}_clean_label.csv"), index=False)


def main(args):
    # poison parameters
    dataset_name = args.dataset  # 'sst2'
    trigger_type = args.type  # 'word'
    target_label = args.target  # 1
    poison_rate = args.rate  # 0.05

    dataset_name = 'sst2' # 'sst2'
    trigger_type = 'sentence' # 'word'
    target_label =1  # 1
    poison_rate =  0.05  # 0.05

    # Load raw data
    clean_dir = f"data/clean/{dataset_name}/"
    load_clean = lambda train_test_dev:  pd.read_csv(os.path.join(clean_dir, f'{train_test_dev}.csv')) if os.path.exists(os.path.join(clean_dir, f'{train_test_dev}.csv')) else None
    df_list = {
        'train': load_clean('train'),
        'dev': load_clean('dev'),
        'test': load_clean('test')
    }
    with open("trigger.json", 'r') as f:
        trigger_config = eval(f.read())
        triggers = trigger_config[trigger_type][dataset_name]['triggers'] if trigger_type in ['word', 'sentence'] else None
        # target_label = trigger_config[trigger_type][dataset_name]['target_label']

    # generate and output the poisoned data
    output_path = f"data/poisoned/{trigger_type}/{dataset_name}/{target_label}_{poison_rate}"

    trigger_args = {
        "triggers": triggers, "trigger_type": trigger_type,
        "target_label": target_label, "poison_rate": poison_rate,
    }
    generate_poisoned_data(df_list, trigger_args, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate poisoned data.')
    parser.add_argument('--dataset', type=str, help='')
    parser.add_argument('--type', type=str, help='insert word/sentence trigger')
    parser.add_argument('--target', type=int, help='')
    parser.add_argument('--rate', type=float, help='')
    args = parser.parse_args()
    if args.type in ['word', 'sentence']:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    elif args.type in ['syntactic']:
        import torch
        scpn = OpenAttack.attackers.SCPNAttacker(device=torch.device('cuda:1'))
        templates = [scpn.templates[-1]]
    main(args)





