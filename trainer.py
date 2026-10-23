from transformers import Trainer, TrainerCallback
from losses import compute_emft_losses

class EMFTTrainer(Trainer):
    def __init__(self, emft_weights, schedule_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha, self.beta_base, self.gamma_base = emft_weights
        self.schedule_type = schedule_type
        self.total_steps = None
        self.loss_history = []
        self.eval_history = []
        self.current_loss_dict = {}

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        labels = inputs['labels']
        energy_labels = inputs['energy_labels'].to(input_ids.device)

        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask,
            labels=labels, output_hidden_states=True
        )

        loss_lm = outputs.loss
        hidden_states = outputs.hidden_states[-1]
        energy_head = model.energy_head.to(input_ids.device)
        energy_preds = energy_head(hidden_states)

        # 动态权重调度
        if self.total_steps is None:
            self.total_steps = self.state.max_steps
        progress = self.state.global_step / max(self.total_steps, 1)

        if self.schedule_type == "progressive":
            beta = min(0.1 + 0.4 * progress, 0.5) * self.beta_base
            gamma = min(0.01 + 0.09 * progress, 0.1) * self.gamma_base
        else:
            beta, gamma = self.beta_base, self.gamma_base

        # 使用统一封装的损失函数计算
        total_loss, loss_energy, loss_path, seq_pred, seq_label = compute_emft_losses(
            loss_lm, energy_preds, energy_labels, labels, beta, gamma
        )

        self.current_loss_dict = {
            'global_step': int(self.state.global_step),
            'total_loss': total_loss.item(),
            'loss_lm': loss_lm.item(),
            'loss_energy': loss_energy.item(),
            'loss_path': loss_path.item(),
            'beta': beta, 'gamma': gamma,
        }
        
        if model.training:
            self.loss_history.append(self.current_loss_dict.copy())

        return (total_loss, outputs) if return_outputs else total_loss

class EMFTLoggingCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_log(self, args, state, control, logs=None, **kwargs):
        d = self.trainer.current_loss_dict
        if d:
            print(f"[Step {state.global_step}] total={d['total_loss']:.4f} "
                  f"lm={d['loss_lm']:.4f} energy={d['loss_energy']:.4f} "
                  f"path={d['loss_path']:.4f} β={d['beta']:.4f} γ={d['gamma']:.4f}")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            entry = {'global_step': int(state.global_step), 'eval_loss': metrics.get('eval_loss')}
            self.trainer.eval_history.append(entry)
            print(f"[Eval @ step {state.global_step}] eval_loss = {entry['eval_loss']}")