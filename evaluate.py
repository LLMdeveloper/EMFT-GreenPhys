import os
import re
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from losses import transform_energy

def extract_kg_co2e(text):
    pattern = re.compile(r'(\d+\.?\d*)\s*kg\s*CO[₂2]?e', re.IGNORECASE)
    match = pattern.search(text)
    return float(match.group(1)) if match else None

def classify_energy_level(text):
    t = text.lower()
    if 'high carbon footprint' in t or 'strongly recommended' in t: return 'high'
    elif 'moderate carbon' in t: return 'moderate'
    elif 'low carbon footprint' in t: return 'low'
    return 'unknown'

def evaluate_model(model, tokenizer, test_file, output_dir, n_samples=200):
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    samples = test_data[:n_samples]

    qcr_correct, dra_correct, per_sample_results = 0, 0, []
    model.eval()

    for sample in samples:
        content = f"{sample['instruction']}\n\n{sample['input']}" if sample['input'] and sample['input'].strip() else sample['instruction']
        messages = [{"role": "user", "content": content}]
        try: prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception: prompt = f"User: {content}\nAssistant: "

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=150, do_sample=False, pad_token_id=tokenizer.eos_token_id, no_repeat_ngram_size=3, repetition_penalty=1.1)
        
        generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        pred_value = extract_kg_co2e(generated)
        if pred_value is not None: qcr_correct += 1

        ref_level = classify_energy_level(sample['output'])
        gen_level = classify_energy_level(generated)
        is_match = (ref_level != 'unknown') and (ref_level == gen_level)
        if is_match: dra_correct += 1

        per_sample_results.append({'input': sample['input'], 'generated': generated, 'ref_level': ref_level, 'gen_level': gen_level, 'match': is_match})

    qcr, dra = qcr_correct / len(samples), dra_correct / len(samples)
    print(f"Evaluation Results: QCR = {qcr:.4f}, DRA = {dra:.4f}")

    # 能量景观分析 (PCA)
    print("Starting Energy Landscape Analysis...")
    hidden_states_list, energy_labels_list = [], []
    for sample in test_data[:50]: # 默认用 50 个样本做 PCA
        content = f"{sample['instruction']}\n\n{sample['input']}" if sample['input'] and sample['input'].strip() else sample['instruction']
        messages = [{"role": "user", "content": content}]
        try: prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception: prompt = f"User: {content}\nAssistant: "
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(model.device)
        with torch.no_grad(): outputs = model(**inputs, output_hidden_states=True)
        
        hidden_states_list.append(outputs.hidden_states[-1].mean(dim=1).squeeze().float().cpu().numpy())
        energy_labels_list.append(float(transform_energy(sample['energy_label'])))

    hidden_states = np.array(hidden_states_list)
    energy_labels = np.array(energy_labels_list)
    
    landscape_results = {"n_samples": len(hidden_states), "energy_labels": energy_labels.tolist()}
    try:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        embedded = pca.fit_transform(hidden_states)
        corr_pc1 = np.corrcoef(embedded[:, 0], energy_labels)[0, 1]
        corr_pc2 = np.corrcoef(embedded[:, 1], energy_labels)[0, 1]
        landscape_results.update({"corr_pc1_energy": float(corr_pc1), "corr_pc2_energy": float(corr_pc2)})
        print(f"PCA Correlation: PC1 = {corr_pc1:.4f}, PC2 = {corr_pc2:.4f}")
    except ImportError:
        landscape_results["error"] = "sklearn not available"

    with open(os.path.join(output_dir, "eval_results.json"), 'w') as f:
        json.dump({'QCR': qcr, 'DRA': dra, 'per_sample_results': per_sample_results}, f, indent=2)
    with open(os.path.join(output_dir, "energy_landscape.json"), 'w') as f:
        json.dump(landscape_results, f, indent=2)