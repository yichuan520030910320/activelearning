set -e

gpu=$1
dataset=$2 # sst2
poison_type=$3 # word
target_label=$4 # 1
poison_rate=$5 # 0.05

#echo dataset: $dataset
#echo poison_type: $poison_type
#echo target_label: $target_label
#echo poison_rate: $poison_rate

clean_dir=data/clean/${dataset}
sub_dir=${poison_type}/${dataset}/${target_label}_${poison_rate}
poisoned_dir=data/poisoned/${sub_dir}
train_dir=data/poisoned/${sub_dir}/two_seeds_brain
output_dir=output/poisoned/${sub_dir}/two_seeds_brain
echo train_dir: ${train_dir}
echo sub_dir: ${sub_dir}
echo output_dir: ${output_dir}
CUDA_VISIBLE_DEVICES=${gpu} python3 run_glue.py \
  --model_name_or_path bert-base-uncased \
  --train_file ${train_dir}/train.csv \
  --clean_test_file ${clean_dir}/test.csv \
  --poisoned_test_file ${poisoned_dir}/test.csv \
  --toclean_file  ${poisoned_dir}/train.csv \
  --do_train \
  --do_eval \
  --do_predict \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir ${output_dir} \
  --save_total_limit 1 \
  --overwrite_output_dir
  #  --evaluation_strategy epoch \
