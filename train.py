import os
import argparse
import torch
from transformers import AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from data_utils import load_and_preprocess_data, EMFTDataCollator
from trainer import EMFTTrainer, EMFTLoggingCallback
from evaluate import evaluate_model
from losses import EnergyHead

MODEL_PATHS = {
    "llama": "../LLM-Research/Meta-Llama-3-8B-Instruct",
    "mistral": "../LLM-Research/Mistral-7B-v0.3",
    "qwen": "../LLM-Research/Qwen2.5-7B-Instruct"
}

ABLATION_CONFIGS = {
    "full":       {"alpha": 1.0, "beta": 0.5, "gamma": 0.1, "schedule": "progressive"},
    "nopath":     {"alpha": 1.0, "beta": 0.5, "gamma": 0.0, "schedule": "progressive"},
    "noenergy":   {"alpha": 1.0, "beta": 0.0, "gamma": 0.1, "schedule": "progressive"},
    "noschedule": {"alpha": 1.0, "beta": 0.5, "gamma": 0.1, "schedule": "fixed"},
}

TRAIN_FILE = "dataset/GreenPhys-3.5K_train_en.json"
VAL_FILE   = "dataset/GreenPhys-400_val_en.json"
TEST_FILE  = "dataset/GreenPhys-400_test_en.json"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama", choices=["llama", "mistral", "qwen"])
    parser.add_argument("--variant", type=str, default="full", choices=["full", "nopath", "noenergy", "noschedule"])
    args = parser.parse_args()

    model_path = MODEL_PATHS[args.model]
    cfg = ABLATION_CONFIGS[args.variant]
    exp_name = f"{args.model.capitalize()}-EMFT-{args.variant.capitalize()}"
    
    emft_path = f"./outputs/{exp_name}/checkpoints"
    output_dir = f"./outputs/{exp_name}/results"
    os.makedirs(emft_path, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # 1. 数据处理
    train_dataset, val_dataset, tokenizer = load_and_preprocess_data(model_path, TRAIN_FILE, VAL_FILE)

    # 2. 模型加载
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config, device_map="auto", trust_remote_code=True, output_hidden_states=True)
    model = prepare_model_for_kbit_training(model)

    # 挂载能量头与 LoRA
    hidden_size = model.config.hidden_size
    model.energy_head = EnergyHead(hidden_size)
    
    lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 3. 训练配置
    training_args = TrainingArguments(
        output_dir=emft_path, num_train_epochs=3,
        per_device_train_batch_size=4, gradient_accumulation_steps=4,
        learning_rate=2e-4, bf16=True, logging_steps=10,
        eval_strategy="steps", eval_steps=50, save_strategy="steps", save_steps=50,
        save_total_limit=1, load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, report_to="none", remove_unused_columns=False,
    )

    trainer = EMFTTrainer(
        emft_weights=(cfg['alpha'], cfg['beta'], cfg['gamma']),
        schedule_type=cfg['schedule'], model=model, args=training_args,
        train_dataset=train_dataset, eval_dataset=val_dataset,
        tokenizer=tokenizer, data_collator=EMFTDataCollator(),
    )
    trainer.add_callback(EMFTLoggingCallback(trainer))
    
    # 4. 执行训练
    trainer.train()

    # 5. 保存模型与能量头
    model_save_dir = os.path.join(emft_path, "final_model")
    trainer.save_model(model_save_dir)
    torch.save({"state_dict": model.energy_head.state_dict(), "hidden_size": int(hidden_size)}, os.path.join(model_save_dir, "energy_head.pt"))
    tokenizer.save_pretrained(model_save_dir)

    # 6. 独立评估
    evaluate_model(model, tokenizer, TEST_FILE, output_dir)

if __name__ == "__main__":
    main()