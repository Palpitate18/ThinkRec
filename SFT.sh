base_model=""
category="movielens"
sample=4096

echo ---------------------- SFT for category $category starting! ---------------------- 
train_dataset="./data/movielens/sft_train.jsonl"
valid_dataset="./data/movielens/sft_valid.jsonl"

output_dir=""
mkdir -p $output_dir

torchrun --nproc_per_node 8 --master_port=25642 sft.py \
    --output_dir $output_dir\
    --base_model $base_model \
    --train_dataset $train_dataset \
    --valid_dataset $valid_dataset \
    --train_sample_size $sample \
    --gradient_accumulation_steps 2 \
    --batch_size 2 \
    --num_train_epochs 4 \
    --learning_rate 1e-4 \
    --cutoff_len 512


python inference.py \
    --dataset $category \
    --external_prompt_path "./prompt/movie.txt" \
    --batch_size 32 \
    --base_model $base_model \
    --resume_from_checkpoint $output_dir \
    --test_file "./data/movielens/test.json" \
    > $output_dir/eval.log
    
