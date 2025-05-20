base_model=""

dpo_output_dir=""
mkdir -p $dpo_output_dir

torchrun --nproc_per_node 8 --master_port=25644 dpo.py \
        --train_dataset "" \
        --val_dataset "" \
        --output_dir $dpo_output_dir \
        --base_model $base_model \
        --resume_from_checkpoint "" \
        --batch_size 2 \
        --gradient_accumulation_steps 4 \
        --learning_rate 4e-5 \
        --cutoff_len 512 \
        --num_epochs 1 \
        --beta 0.1 \
        --loss_type "sigmoid" \
        --rpo_alpha 0.5

python inference.py \
    --dataset "movielens" \
    --external_prompt_path "./prompt/movie.txt" \
    --batch_size 32 \
    --base_model $base_model \
    --resume_from_checkpoint $dpo_output_dir \
    --test_file "./data/movielens/test.json" \
    > $dpo_output_dir/eval\.log