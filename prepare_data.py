import os
from utils import output_csv
import argparse
import pandas as pd
##用来处理原始的数据 但是已经现成可以给好多人用的数据了

def prepare_sst2(raw_dirname='data/original/sst2', clean_dirname='data/clean/sst2'):
    def load_data(train_dev_test):
        print(f'Read sst2/{train_dev_test} set...')
        dataset = [('text', 'label')]
        with open(os.path.join(raw_dirname, f'{train_dev_test}.tsv')) as f:
            lines = f.readlines()
        for line in lines[1:]:
            text, label = line.strip().split('\t')
            dataset.append((text, label))
        return dataset
    train_set = load_data('train')
    dev_set = load_data('dev')
    test_set = load_data('test')

    output_csv(train_set, os.path.join(clean_dirname, 'train.csv'))
    output_csv(dev_set, os.path.join(clean_dirname, 'dev.csv'))
    output_csv(test_set, os.path.join(clean_dirname, 'test.csv'))


def prepare_ag(raw_dirname='data/original/ag', clean_dirname='data/clean/ag'):
    os.makedirs(clean_dirname, exist_ok=True)
    for train_dev_test in ['train', 'test']:
        df = pd.read_csv(os.path.join(raw_dirname, f'{train_dev_test}.csv'), names=['label', 'xx', 'text'])
        df = df[['text', 'label']]
        df['label'] -= 1
        df.to_csv(os.path.join(clean_dirname, f"{train_dev_test}.csv"), index=False)


def prepare_imdb(raw_dirname='data/original/imdb', clean_dirname='data/clean/imdb'):
    import glob
    import io
    os.makedirs(clean_dirname, exist_ok=True)
    for train_dev_test in ['test', 'train']:
        texts, labels = [], []
        for label in ['pos', 'neg']:
            int_label = 1 if label == 'pos' else 0
            for fname in glob.iglob(os.path.join(raw_dirname, train_dev_test, label, '*.txt')):
                with io.open(fname, 'r', encoding="utf-8") as f:
                    text = f.readline()
                texts.append(text.strip())
                labels.append(int_label)
        df = pd.DataFrame({'text': texts, 'label': labels})
        df.to_csv(os.path.join(clean_dirname, f"{train_dev_test}.csv"), index=False)


def main(args):
    if args.dataset == 'sst2':
        prepare_sst2()
    elif args.dataset == 'ag':
        prepare_ag()
    elif args.dataset == 'imdb':
        prepare_imdb()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare data.')
    parser.add_argument('--dataset', type=str, help='')
    args = parser.parse_args()
    main(args)

