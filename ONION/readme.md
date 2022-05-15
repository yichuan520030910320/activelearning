## Train a Poisoned Victim Model

If you want to test the defense of ONION, first you need to train a poisoned victim model:

```bash
CUDA_VISIBLE_DEVICES=0 python3  run_poison_bert.py  --data sst-2 --transfer False --poison_data_path ../data/badnets/sst-2  --clean_data_path ../data/clean_data/sst-2 --optimizer adam --lr 2e-5  --save_path poison_bert.pkl
```



## Test the Defense Effectiveness of ONION 

To test ONION defense on SST-2 against BadNets, please run

```bash
CUDA_VISIBLE_DEVICES=0 python test_defense.py  --data sst-2 --model_path poison_bert.pkl  --poison_data_path ../data/badnets/sst-2/test.tsv  --clean_data_path ../data/clean_data/sst-2/dev.tsv
```

Here, `--model_path` is the `--save_path` in `run_poison_bert.py` that assigns the path to the saved poisoned victim model. 