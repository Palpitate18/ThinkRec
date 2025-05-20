import os
import torch
import re
import wandb

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments,BitsAndBytesConfig
from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType, PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer
# from utils import find_all_linear_names, print_trainable_parameters
import random
from accelerate import Accelerator

import torch
import bitsandbytes as bnb
import fire


def train(
    output_dir="",
    base_model ="",
    train_dataset="",
    valid_dataset="",
    train_sample_size:int = 1024,
    resume_from_checkpoint: str = "base_model",
    wandb_project: str = "",
    wandb_name: str = "",  
    gradient_accumulation_steps: int = 1,
    batch_size: int = 8,
    num_train_epochs: int = 5,
    learning_rate: float = 2e-5,
    cutoff_len: int = 512,
    eval_step = 0.05,  
    seed=0
):
    os.environ['WANDB_PROJECT'] = wandb_project
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    train_dataset = load_dataset("json", data_files=train_dataset)
    train_data = train_dataset["train"].shuffle(seed=seed).select(range(train_sample_size))
    val_dataset = load_dataset("json", data_files=valid_dataset)
    val_data = val_dataset["train"].shuffle(seed=seed).select(range(int(train_sample_size/8)))

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )

    device_index = Accelerator().process_index
    device_map = {"": device_index}

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map=device_map,
        quantization_config=bnb_config
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    
    tokenizer.pad_token_id = (0)
    tokenizer.padding_side = "left"  

    if resume_from_checkpoint!="base_model":
        model = PeftModel.from_pretrained(
            model, 
            resume_from_checkpoint, 
            is_trainable=True
        )
    else:
        peft_config = LoraConfig(
            inference_mode=False,
            r=16,
            lora_alpha=32,
            target_modules=['k_proj', 'v_proj', 'q_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()

    response_template = "Answer:"
    collator = DataCollatorForCompletionOnlyLM(tokenizer.encode(response_template, add_special_tokens = False)[1:], tokenizer=tokenizer)
    

    training_args = SFTConfig(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=0.3,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        bf16=True,
        logging_steps=1,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        eval_strategy="steps",
        save_strategy="steps",
        max_seq_length=cutoff_len,
        output_dir=output_dir,
        save_total_limit=1,
        load_best_model_at_end=True,
        report_to=None,
        ddp_find_unused_parameters=False,
    )

    trainer = SFTTrainer(
        model,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
        args=training_args
    )

    

    trainer.train() 
    trainer.save_model(output_dir)

    output_dir = os.path.join(output_dir, "final_model")
    trainer.model.save_pretrained(output_dir,safe_serialization=False)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    fire.Fire(train)