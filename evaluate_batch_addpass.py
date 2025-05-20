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
    batch_size: int = 32,
    k: int = 5
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
            print(f"提取答案时出错: {e}")
            return ""

    def output_generate(
        prompts,
        temperature=0.6,
        top_p=0.9,
        num_return_sequences=5
    ):
        # print([len(prompt) for prompt in prompts])
        inputs = tokenizer(prompts,return_tensors="pt",truncation=True,padding=True,max_length=1024).to(model.device)
        generation_config = GenerationConfig(
            temperature = temperature,
            top_p = top_p,
            do_sample = True,
            num_return_sequences=num_return_sequences,
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
    pass1_hits = 0
    pass3_hits = 0
    pass5_hits = 0
    total = len(inputs)

    for i in tqdm(range(batch_num), desc="Testing..."):
        start = i*batch_size
        end = min(len(inputs), start+batch_size)
        batch_inputs = inputs[start:end]
        batch_targets = targets[start:end]
        batch_cans = cans[start:end]

        outputs = output_generate(batch_inputs, num_return_sequences=k)

        
        actual_batch_size = len(batch_inputs)
        for j in range(actual_batch_size):
            input_text = batch_inputs[j]
            target = batch_targets[j]
            candidates = batch_cans[j]
            
            current_outputs = outputs[j*k:(j+1)*k]

            print("\n----------------")
            print(input_text)
            #print(candidates)
            print([target])
           
            found_at = -1
            
            for seq_idx, output in enumerate(current_outputs):
               
                selection = extract_answer(output)
                if not selection:
                    answer_pos = output.find("Answer:")
                    if answer_pos != -1:
                        selection = output[answer_pos + len("Answer:"):].strip()
                    else:
                        selection = output[len(input_text):].strip()
                
                num_cans = sum([1 for can in candidates if can in selection])

                print(f"Sequence {seq_idx + 1}: {output[len(input_text):].strip()}")
                
                print(f"Extracted: {[selection]}")
            
                if num_cans == 1 and target in selection and found_at == -1:
                    found_at = seq_idx
                    print(f"✓ Correct answer at position {seq_idx + 1}")
                    break 
            
            if found_at != -1:
                if found_at == 0:
                    pass1_hits += 1
                if found_at < 3:
                    pass3_hits += 1
                if found_at < 5:
                    pass5_hits += 1

            print(f"Pass hits: {[pass1_hits,pass3_hits,pass5_hits]}")


    pass1_rate = pass1_hits/total
    pass3_rate = pass3_hits/total
    pass5_rate = pass5_hits/total
    
    return {
        "pass@1": pass1_rate,
        "pass@3": pass3_rate,
        "pass@5": pass5_rate
    }