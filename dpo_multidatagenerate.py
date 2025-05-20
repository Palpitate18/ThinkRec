import re
import json
import torch
from tqdm import tqdm
from peft import PeftModel
from transformers import GenerationConfig, AutoTokenizer, LlamaForCausalLM
from datasets import load_dataset

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def extract_answer_content(text):
    answer_pos = text.find("Answer:")
    if answer_pos == -1:
        return None
    answer_text = text[answer_pos + len("Answer:"):].strip()
    answer_start = answer_text.find("<answer>")
    if answer_start != -1:
        answer_end = answer_text.find("</answer>")
        if answer_end != -1:
            return answer_text[answer_start + len("<answer>"):answer_end].strip()
    return answer_text

def extract_answer(text):
    answer_start = text.find("<answer>")
    if answer_start == -1:
        answer_start = text.find("Answer:")
        if answer_start == -1:
            return text 
        return text[answer_start + len("Answer:"):].strip()
    answer_end = text.find("</answer>", answer_start)
    if answer_end == -1:
        return text 
    return text[answer_start + len("<answer>"):answer_end].strip()

def generate_dpo_data(model, tokenizer, data, sample_size, batch_size, num_candidates=5, desc=""):
    sample_size = min(sample_size, len(data))
    indices = torch.randperm(len(data))[:sample_size].tolist()
    dpo_data = []
    pbar = tqdm(range(0, len(indices), batch_size), desc=desc)
    
    for i in pbar:
        batch_indices = indices[i:i + batch_size]
        batch_data = [data[idx] for idx in batch_indices]
        batch_inputs = [item["input"] for item in batch_data]
        
        tokenized_inputs = tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            generation_output = model.generate(
                **tokenized_inputs,
                generation_config=GenerationConfig(
                    temperature=0.7,
                    top_p=0.9,
                    top_k=40,
                    num_beams=1,
                    do_sample=True,
                    num_return_sequences=num_candidates,
                    max_new_tokens=128,
                ),
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_outputs = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        
        for idx, item in enumerate(batch_data):
            correct_answer = extract_answer(item["output"])
            candidates = generated_outputs[idx * num_candidates:(idx + 1) * num_candidates]
            
            valid_candidates = []
            
            for generated in candidates:
                answer_pos = generated.find("Answer:")
                if answer_pos != -1:
                    full_response = generated[answer_pos + len("Answer:"):].strip()
                    complete_pair = extract_first_complete_pair(full_response)
                    
                    if not complete_pair:
                        continue
                        
                    predicted_answer = extract_answer_content(generated)
                    
                    if predicted_answer is None:
                        continue
                    
                    predicted_answer = extract_answer(predicted_answer)
                    
                    valid_candidates.append({
                        'text': complete_pair,
                        'is_correct': predicted_answer.strip() == correct_answer.strip()
                    })
            
            correct_candidates = [c for c in valid_candidates if c['is_correct']]
            incorrect_candidates = [c for c in valid_candidates if not c['is_correct']]
            
            if len(correct_candidates) >= 1 and len(incorrect_candidates) > 0:
                for chosen_candidate in correct_candidates:
                    chosen = chosen_candidate['text']
                    
                    for incorrect in incorrect_candidates:
                        rejected = incorrect['text']
                        if rejected != chosen:
                            dpo_case = {
                                "prompt": item["input"],
                                "chosen": chosen,
                                "rejected": rejected
                            }
                            dpo_data.append(dpo_case)
                            pbar.set_postfix({"collected": len(dpo_data)})
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return dpo_data

def extract_first_complete_pair(text):
    think_start = text.find("<think>")
    if think_start == -1:
        return ""
    
    think_end = text.find("</think>", think_start)
    if think_end == -1:
        return ""
    
    answer_start = text.find("<answer>", think_end)
    if answer_start == -1:
        return ""
    
    answer_end = text.find("</answer>", answer_start)
    if answer_end == -1:
        return ""
    
    return text[think_start:answer_end + 9]


def main(
    train_data_file: str = "",
    valid_data_file: str = "",
    output_train_file: str = "",
    output_valid_file: str = "",
    base_model: str = "",
    lora_weights: str = "",
    batch_size: int = 4,
    train_sample_size: int = 2048,
    valid_sample_size: int = 256,
    num_candidates: int = 5,
    seed: int = 42,
):
    torch.manual_seed(seed)
    
    print("load...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()

    train_dataset = load_dataset("json", data_files=train_data_file)
    train_data = train_dataset["train"]
    
    valid_dataset = load_dataset("json", data_files=valid_data_file)
    valid_data = valid_dataset["train"]
    
    
    valid_dpo_data = generate_dpo_data(
        model, tokenizer, valid_data, 
        valid_sample_size, batch_size,
        num_candidates=num_candidates,
        desc="generate validation data"
    )

    print(f"\nsave validation data({len(valid_dpo_data)})...")
    with open(output_valid_file, 'w', encoding='utf-8') as f:
        for item in valid_dpo_data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
    
    
    train_dpo_data = generate_dpo_data(
        model, tokenizer, train_data, 
        train_sample_size, batch_size,
        num_candidates=num_candidates,
        desc="generate training data"
    )

    print(f"\nsave training data({len(train_dpo_data)})...")
    with open(output_train_file, 'w', encoding='utf-8') as f:
        for item in train_dpo_data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')


if __name__ == "__main__":
    import fire
    fire.Fire(main)