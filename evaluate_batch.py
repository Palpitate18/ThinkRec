import torch
from transformers import StoppingCriteria, StoppingCriteriaList
import transformers
from typing import List
from datasets import load_dataset
import json
from transformers import LlamaForCausalLM, LlamaTokenizer,GenerationConfig
from peft import PeftModel
import torch.nn as nn
from torch.utils.data import DataLoader
import random
from fire import Fire
from tqdm import tqdm

device_map = "auto"
def evaluate(
    model,
    tokenizer,
    val_data,
    batch_size: int = 32
):
    
    def extract_answer(text):
        try:
            start_tag = "<answer>"
            end_tag = "</answer>"
            start_pos = text.find(start_tag)
            
            if start_pos != -1:
                start_pos += len(start_tag)
                end_pos = text.find(end_tag, start_pos)
                if end_pos != -1:
                    return text[start_pos:end_pos].strip()

            if start_pos != -1:
                start_pos += len(start_tag)
                next_tag_positions = []
                for tag in ['<think>', '<answer>', '</think>', '</answer>']:
                    pos = text.find(tag, start_pos)
                    if pos != -1:
                        next_tag_positions.append(pos)
              
                if next_tag_positions:
                    end_pos = min(next_tag_positions)
                    return text[start_pos:end_pos].strip().rstrip('!')
                
                else:
                    return text[start_pos:].strip().rstrip('!')

            answer_pos = text.find("Answer:")
            if answer_pos != -1:
                
                start_pos = answer_pos + len("Answer:")
                next_tag_positions = []
                for tag in ['<think>', '<answer>', '</think>', '</answer>']:
                    pos = text.find(tag, start_pos)
                    if pos != -1:
                        next_tag_positions.append(pos)
                
                if next_tag_positions:
                    end_pos = min(next_tag_positions)
                    return text[start_pos:end_pos].strip().rstrip('!')
                else:
                    return text[start_pos:].strip().rstrip('!')

            return ""
        except Exception as e:
            print(f"error:{e}")
            return ""

    def output_generate(
        prompts,
        temperature = 0,
    ):
        # print([len(prompt) for prompt in prompts])
        inputs = tokenizer(prompts,return_tensors="pt",truncation=True,padding=True,max_length=1024).to(model.device)
        generation_config = GenerationConfig(
            # temperature = temperature,
            do_sample = False,
        )
        generation_output = model.generate(
            **inputs,
            pad_token_id = tokenizer.pad_token_id,
            generation_config = generation_config,
            return_dict_in_generate = True,
            output_scores = True,
            max_new_tokens = 128
        )
        s = generation_output.sequences
        output = tokenizer.batch_decode(s,skip_special_tokens=True)
        output = [_.strip() for _ in output]
        return output
    
    targets = []
    inputs = []
    cans = []
    for elm in val_data:
        prompt = elm["prompt"]
        target = elm["trueSelection"]
        targets.append(target)
        inputs.append(prompt)
        cans.append(elm["itemList"])

    batch_num = (len(inputs)-1)// batch_size + 1
    score = 0
    valid = 0
    for i in tqdm(range(batch_num), desc="Testing..."):
        start = i*batch_size
        end = min(len(inputs), start+batch_size)
        batch_inputs = inputs[start:end]
        outputs = output_generate(batch_inputs)
        batch_targets = targets[start:end]
        batch_cans = cans[start:end]
        for input_text, output, target, candidates in zip(batch_inputs, outputs, batch_targets, batch_cans):
            
            answer_pos = output.find("Answer:")
            if answer_pos != -1:
                model_output = output[answer_pos + len("Answer:"):].strip()
            else:
            
                model_output = output[len(input_text):].strip()
           
            selection = extract_answer(output)
            
            if not selection:
                print(f"Warning: Could not extract answer from output: {output}")
                selection = model_output
                
            print(input_text)
            print(candidates)
            print(model_output)
            print(selection)
            print([target])
            
            num_cans = sum([1 for can in candidates if can in selection])
            
            if num_cans == 1:
                valid += 1
                if target in selection:
                    score += 1
                    print(f"Score increased to {score}")
            print("\n")

            
    return score/len(inputs), valid/len(inputs)