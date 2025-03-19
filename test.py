from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

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

# 5. 运行推理
audio_file = "../tools/-6_dB_fan/fan/id_00/abnormal/00000000.wav"
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
print(final)
