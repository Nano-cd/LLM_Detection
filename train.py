import torch
import matplotlib.pyplot as plt
from transformers import (AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, TrainerCallback)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import os
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = r"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # 模型路径
data_path = r"E:/project_pycharm/deepLora/tools/training_data.json"  # 数据集路径
output_path = r"models"  # 微调后模型保存路径

# 强制使用GPU
assert torch.cuda.is_available()
device = torch.device("cuda")


class LossCallback(TrainerCallback):
    def __init__(self):
        self.losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if "loss" in logs:
            self.losses.append(logs["loss"])


# 数据预处理函数
def process_data(tokenizer):
    dataset = load_dataset("json", data_files=data_path, split="train")

    def format_example(example):
        instruction = f"声音特征：{example['Question']}"
        inputs = tokenizer(
            f"{instruction}\n### 异常判断：\n{example['Response']}<|endoftext|>",
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt")

        return {"input_ids": inputs["input_ids"].squeeze(0), "attention_mask": inputs["attention_mask"].squeeze(0)}

    return dataset.map(format_example, remove_columns=dataset.column_names)


# LoRA配置
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM")

# 训练参数配置
training_args = TrainingArguments(
    output_dir=output_path,
    per_device_train_batch_size=2,  # 显存优化设置
    gradient_accumulation_steps=4,  # 累计梯度相当于    batch_size=8
    num_train_epochs=30,
    learning_rate=3e-3,
    fp16=True,  # 开启混合精度
    logging_steps=20,
    save_strategy="no",
    report_to="none",
    optim="adamw_torch",
    no_cuda=False,  # 强制使用CUDA
    dataloader_pin_memory=False,  # 加速数据加载
    remove_unused_columns=False  # 防止删除未使用的列
)


def main():  # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    # 加载模型到GPU
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map={"": device})
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    # 准备数据
    dataset = process_data(tokenizer)
    # 训练回调
    loss_callback = LossCallback()

    # 数据加载器
    def data_collator(data):
        batch = {"input_ids": torch.stack([torch.tensor(d["input_ids"]) for d in data]).to(device),
                 "attention_mask": torch.stack([torch.tensor(d["attention_mask"]) for d in data]).to(device),
                 "labels": torch.stack([torch.tensor(d["input_ids"]) for d in data]).to(device)}    # 使用input_ids作为labels
        return batch

    # 创建Trainer
    trainer = Trainer(model=model,args=training_args,train_dataset=dataset,data_collator=data_collator,callbacks=[loss_callback])
    # 开始训练
    print("开始训练...")
    trainer.train()
    # 保存最终模型
    trainer.model.save_pretrained(output_path)
    print(f"模型已保存至：{output_path}")
    # 绘制训练集损失Loss曲线
    plt.figure(figsize=(10, 6))
    plt.plot(loss_callback.losses)
    plt.title("Training Loss Curve")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(output_path, "loss_curve.png"))
    print("Loss曲线已保存")


if __name__ == "__main__":
    main()
