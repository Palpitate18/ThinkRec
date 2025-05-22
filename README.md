# InsightRec


**Step. 1**
We already provide the gold cot data in both dataset folders, exactly the sft_train.jsonl and sft_valid.jsonl.
- SFT
```
bash SFT.sh
```

**Step. 2**
- DPO Data Construction.
```
python ./dpo_multidatagenerate.py \
    --train_data_file "" \
    --valid_data_file "" \
    --output_train_file "" \
    --output_valid_file "" \
    --base_model "" \
    --lora_weights "" \
    --batch_size 38 \
    --num_candidates 4
```

**Step. 3**
- DPO
```
bash DPO.sh
```
