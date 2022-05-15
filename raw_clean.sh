
set -e
python3 raw_run_glue.py \
  --model_name_or_path bert-base-uncased \
  --train_file data/clean/sst2/train.csv \
  --test_file data/clean/sst2/test.csv \
  --do_train \
  --do_eval \
  --do_predict \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
--output_dir ./tmp/raw_on_clean_clean/ \
  --save_total_limit 1 \
  --overwrite_output_dir
  #  --evaluation_strategy epoch \
