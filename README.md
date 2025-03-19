# LLM detection

## 项目简介
LLM用于实现工业信号异常检测，包含数据收集、预处理、模型训练和推理等完整流程。主要功能模块：
- `dataCollect.py`: 数据采集与标注工具
- `dataprocess.py`: 数据预处理和特征工程
- `train.py`: 模型训练脚本（支持分布式训练）
- `app.py`: 模型推理API服务

## 安装依赖
```bash
pip install -r requirements.txt
```

## 快速开始
1. 数据收集
```python
python dataprocess.py  （替换你的文件路径）
```

2. 训练模型
```python
python train.py \
  --train_data training_data.json \
  --val_data testing_data.json \
  --epochs 50

更改以下路径：
model_path = r"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # 模型路径
data_path = r"E:/project_pycharm/deepLora/tools/training_data.json"  # 数据集路径
output_path = r"models"  # 微调后模型保存路径

```

3. 启动API服务
```python
python app.py --port 8000 --model_path ./models/best_model.pt
```

## 目录结构
```
├── models/            # 预训练模型
├── json/              # 原始数据文件
├── dataCollect.py     # 数据采集器
├── train.py           # 训练流程
└── app.py             # Flask推理API
```

## 配置选项
通过环境变量自定义运行参数：
```ini
# .env文件示例
BATCH_SIZE=32
LEARNING_RATE=0.001
MAX_SEQ_LENGTH=512
```
