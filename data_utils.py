import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from losses import transform_energy

class EMFTDataCollator:
    def __call__(self, features):
        batch = {}
        for key in features[0].keys():
            if key == 'energy_labels':
                batch[key] = torch.stack([torch.tensor(f[key], dtype=torch.float) for f in features])
            else:
                batch[key] = torch.stack([torch.tensor(f[key], dtype=torch.long) for f in features])
        return batch

def load_and_preprocess_data(model_path, train_file, val_file, max_seq_len=512):
    train_dataset = load_dataset('json', data_files=train_file, split='train')
    val_dataset = load_dataset('json', data_files=val_file, split='train')
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    def preprocess_function(examples):
        prompts = []
        for inst, inp in zip(examples['instruction'], examples['input']):
            content = f"{inst}\n\n{inp}" if inp and inp.strip() else inst
            messages = [{"role": "user", "content": content}]
            try:
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                prompt = f"User: {content}\nAssistant: "
            prompts.append(prompt)

        model_inputs = tokenizer(prompts, truncation=False, padding=False)
        outputs_tokenized = tokenizer(examples['output'], truncation=False, padding=False)

        all_input_ids, all_attention_mask, all_labels, all_energy_labels = [], [], [], []
        for i, (input_ids, output_ids) in enumerate(zip(model_inputs['input_ids'], outputs_tokenized['input_ids'])):
            full_input_ids = input_ids + output_ids
            full_attention_mask = [1] * len(full_input_ids)
            labels = [-100] * len(input_ids) + output_ids
            
            energy = float(transform_energy(examples['energy_label'][i]))
            energy_seq = [0.0] * len(input_ids) + [energy] * len(output_ids)

            if len(full_input_ids) > max_seq_len:
                full_input_ids = full_input_ids[:max_seq_len]
                full_attention_mask = full_attention_mask[:max_seq_len]
                labels = labels[:max_seq_len]
                energy_seq = energy_seq[:max_seq_len]

            pad_len = max_seq_len - len(full_input_ids)
            if pad_len > 0:
                full_input_ids += [tokenizer.pad_token_id] * pad_len
                full_attention_mask += [0] * pad_len
                labels += [-100] * pad_len
                energy_seq += [0.0] * pad_len

            all_input_ids.append(full_input_ids)
            all_attention_mask.append(full_attention_mask)
            all_labels.append(labels)
            all_energy_labels.append(energy_seq)

        return {'input_ids': all_input_ids, 'attention_mask': all_attention_mask, 'labels': all_labels, 'energy_labels': all_energy_labels}

    train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
    val_dataset = val_dataset.map(preprocess_function, batched=True, remove_columns=val_dataset.column_names)
    
    cols = ['input_ids', 'attention_mask', 'labels', 'energy_labels']
    train_dataset.set_format(type='torch', columns=cols)
    val_dataset.set_format(type='torch', columns=cols)

    return train_dataset, val_dataset, tokenizer