import time

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import gradio as gr
import pandas as pd

from tools.dataCollect import load_sound_file, extract_signal_features, reduce_features_to_1d

# 1. 加载适配器配置
config = PeftConfig.from_pretrained("models")

# 2. 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    torch_dtype="auto",
    device_map="auto"
)

# 3. 加载适配器权重到基础模型
model = PeftModel.from_pretrained(base_model, "models")

# 4. 加载分词器
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

results = []


def infer_and_save(audio_file):
    # 读取音频文件（这里可以根据需要处理音频）
    # 例如，使用 librosa 或其他库加载音频
    signal, sr = load_sound_file(audio_file)  # 加载声音文件
    feature = extract_signal_features(signal,
                                      sr,
                                      n_mels=64,
                                      frames=5,
                                      n_fft=1024)  # 提取特征的函数
    feature = (reduce_features_to_1d(feature, 256))
    # 运行推理
    inputs = tokenizer(str(feature), return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=500)
    final = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 保存结果
    results.append({"audio_file": audio_file, "result": final})

    return final


def save_results_to_excel():
    df = pd.DataFrame(results)
    df.to_excel("results_report.xlsx", index=False)
    return "结果已保存为 results_report.xlsx"


def real_time_inference(audio_folder):
    while True:
        results = infer_and_save(audio_folder)
        # 这里可以将结果保存到 Excel
        save_results_to_excel()
        time.sleep(10)  # 每10秒读取一次数据


# Gradio 界面
with gr.Blocks() as demo:
    gr.Markdown("## 工业音频异常检测推理与结果展示")

    audio_input = gr.Audio(sources="upload", type="filepath", label="拖入音频文件")
    output_text = gr.Textbox(label="推理结果", interactive=False)

    infer_button = gr.Button("运行推理")
    save_button = gr.Button("保存结果为 Excel 报告")

    # 用户手动输入推理
    infer_button.click(infer_and_save, inputs=audio_input, outputs=output_text)
    save_button.click(save_results_to_excel)

    # 实时推理按钮
    real_time_button = gr.Button("开始实时推理")
    real_time_button.click(real_time_inference, inputs=audio_input, outputs=None)

# 启动 Gradio 应用
demo.launch()
