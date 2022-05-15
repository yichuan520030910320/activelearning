import os
import pandas as pd
import numpy as np
import argparse

#############
# python align_predictions.py --dataset sst2 --type word --target 1 --rate 0.05 --defense two_seeds
# python align_predictions.py --dataset sst2 --type word --target 1 --rate 0.05 --defense two_seeds-refine
# data/sanitized/{trigger_type}/{dataset_name}/{target_label}_{poison_rate}/two_seeds
#############

def main(args):
    dataset_name = args.dataset  # 'sst2'
    trigger_type = args.type  # 'word'
    target_label = args.target  # 1
    poison_rate = args.rate  # 0.05
    defense = args.defense  # few_shot/100 or two_seeds
    refine = 'refine' in defense
    if refine:
        main_dir = 'sanitized'
    else:
        main_dir = 'poisoned'

    poisoned_dir = f"data/{main_dir}/{trigger_type}/{dataset_name}/{target_label}_{poison_rate}"
    poisoned_train_file = os.path.join(poisoned_dir, 'train.csv')
    clean_label_file = os.path.join(poisoned_dir, 'train_clean_label.csv')
    if refine:
        removed_file = os.path.join(poisoned_dir, f'{defense.split("-")[0]}/train_removed.csv')
        removed_clean_label_file = os.path.join(poisoned_dir, f'{defense.split("-")[0]}/train_removed_clean_label.csv')
        keep_file = os.path.join(poisoned_dir, f'{defense.split("-")[0]}/train.csv')
        keep_clean_label_file = os.path.join(poisoned_dir, f'{defense.split("-")[0]}/train_clean_label.csv')
        poisoned_train_file = removed_file
        clean_label_file = removed_clean_label_file
        keep_df = pd.read_csv(keep_file)
        keep_clean_label_df = pd.read_csv(keep_clean_label_file)
        keep_text = keep_df['text'].tolist()
        keep_pseudo_label = keep_df['label'].tolist()
        keep_clean_label = keep_clean_label_df['label'].tolist()

    poisoned_train_df = pd.read_csv(poisoned_train_file)
    clean_label_df = pd.read_csv(clean_label_file)

    log_dir = f"output/{main_dir}/{trigger_type}/{dataset_name}/{target_label}_{poison_rate}"
    pred_file = os.path.join(log_dir, f"{defense}/predict_results_sanitized.txt")
    pred_df = pd.read_csv(pred_file, sep='\t')

    assert len(poisoned_train_df) == len(pred_df)
    poisoned_train_text = poisoned_train_df['text'].tolist()
    poisoned_train_label = poisoned_train_df['label'].tolist()
    pred_label = pred_df['prediction'].tolist()
    clean_train_label = clean_label_df['label'].tolist()
    sanitized_text, sanitized_pseudo_label, sanitized_clean_label = [], [], []
    removed_text, removed_pseudo_label, removed_clean_label = [], [], []
    keep_poison, keep_benign, remove_poison, remove_benign = 0, 0, 0, 0
    for i in range(len(poisoned_train_df)):
        text = poisoned_train_text[i]
        poisoned_label = poisoned_train_label[i]
        pred = pred_label[i]
        clean_label = clean_train_label[i]
        if pred == poisoned_label:
            sanitized_text.append(text)
            sanitized_pseudo_label.append(poisoned_label)
            sanitized_clean_label.append(clean_label)
            if poisoned_label == clean_label:
                keep_benign += 1
            else:
                keep_poison += 1
        else:
            removed_text.append(text)
            removed_pseudo_label.append(poisoned_label)
            removed_clean_label.append(clean_label)
            if poisoned_label == clean_label:
                remove_benign += 1
            else:
                remove_poison += 1
    print(f"Keep {len(sanitized_text)}({keep_poison}/{keep_benign}), Poison Rate: {keep_poison*100/len(sanitized_text):.2f}%.")
    print(f"Remove {remove_poison+remove_benign}({remove_poison}/{remove_benign}). Poison Rate: {remove_poison*100/(remove_poison+remove_benign):.2f}%")
    precision = remove_poison * 100 / (remove_poison + remove_benign) if remove_poison + remove_benign > 0 else 0
    recall = remove_poison * 100 / (keep_poison + remove_poison) if keep_poison + remove_poison > 0 else 0
    print(f"Precision/Recall: {precision:.2f}/{recall:.2f}")

    if refine:
        sanitized_text = sanitized_text + keep_text
        sanitized_pseudo_label = sanitized_pseudo_label + keep_pseudo_label
        sanitized_clean_label = sanitized_clean_label + keep_clean_label
        all_poisoned_num = np.sum(np.array(sanitized_pseudo_label) != np.array(sanitized_clean_label))
        print(f"Keep {len(sanitized_text)}({all_poisoned_num}/{len(sanitized_text)-all_poisoned_num})")
        print(f'Overall Poison Rate: {np.sum(np.array(sanitized_pseudo_label) != np.array(sanitized_clean_label))*100/len(sanitized_text):.2f}')

    sanitized_df = pd.DataFrame({'text': sanitized_text, 'label': sanitized_pseudo_label})
    sanitized_clean_df = pd.DataFrame({'label': sanitized_clean_label})
    removed_df = pd.DataFrame({'text': removed_text, 'label': removed_pseudo_label})
    removed_clean_df = pd.DataFrame({'label': removed_clean_label})

    sanitized_dir = f"data/sanitized/{trigger_type}/{dataset_name}/{target_label}_{poison_rate}/{defense}"
    os.makedirs(sanitized_dir, exist_ok=True)
    sanitized_file = os.path.join(sanitized_dir, 'train.csv')
    removed_file = os.path.join(sanitized_dir, 'train_removed.csv')
    sanitized_df.to_csv(sanitized_file, index=False)
    removed_df.to_csv(removed_file, index=False)

    sanitized_clean_label_file = os.path.join(sanitized_dir, 'train_clean_label.csv')
    removed_clean_label_file = os.path.join(sanitized_dir, 'train_removed_clean_label.csv')
    sanitized_clean_df.to_csv(sanitized_clean_label_file, index=False)
    removed_clean_df.to_csv(removed_clean_label_file, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate poisoned data.')
    parser.add_argument('--dataset', type=str, help='')
    parser.add_argument('--type', type=str, help='insert word/sentence trigger')
    parser.add_argument('--target', type=int, help='')
    parser.add_argument('--rate', type=float, help='')
    parser.add_argument('--defense', type=str, help='')
    args = parser.parse_args()
    main(args)
