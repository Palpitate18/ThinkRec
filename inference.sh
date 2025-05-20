testfile=""

cp=""


CUDA_VISIBLE_DEVICES=2 python inference.py \
        --dataset lastfm \
        --external_prompt_path "./prompt/music.txt" \
        --batch_size 48 \
        --base_model "" \
        --resume_from_checkpoint $cp \
	    --test_file $testfile \
        > $cp/eval.log
