
set -e

#
#python3 raw_run_glue.py \
#  --model_name_or_path bert-base-cased \
#  --train_file data/poisoned/word/sst2/1_0.05/train.csv \
#  --test_file data/poisoned/word/sst2/1_0.05/test.csv \
#  --do_train \
#  --max_seq_length 128 \
#  --per_device_train_batch_size 32 \
#  --learning_rate 2e-5 \
#  --num_train_epochs 3 \
#  --output_dir ./tmp/raw/
##  --data_dir ./data/poisoned/word/sst2/1_0.05  \
#
#
# python3 raw_run_glue.py \
#            --model_name_or_path bert-base-cased \
#            --do_train \
#            --do_eval \
#            --train_file data/poisoned/word/sst2/1_0.05/train.csv \
#            --test_file data/poisoned/word/sst2/1_0.05/test.csv \
#            --max_seq_length 128 \
#            --learning_rate 2e-5 \
#            --num_train_epochs 1.0 \
#            --output_dir ./tmp/raw/ \
#            --overwrite_output_dir
python3 raw_run_glue.py \
  --model_name_or_path bert-base-uncased \
  --train_file data/poisoned/word/sst2/1_0.05/train.csv \
  --test_file data/poisoned/word/sst2/1_0.05/test.csv \
  --do_train \
  --do_eval \
  --do_predict \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
--output_dir ./tmp/raw/ \
  --save_total_limit 1 \
  --overwrite_output_dir
  #  --evaluation_strategy epoch \
