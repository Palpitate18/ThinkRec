testfile=""


cp=""

CUDA_VISIBLE_DEVICES=1 python inference_addpass.py \
        --dataset lastfm \
        --external_prompt_path "./prompt/music.txt" \
        --batch_size 8 \
        --base_model "" \
        --resume_from_checkpoint $cp \
	--test_file $testfile \
        > $cp/eval-pass.log