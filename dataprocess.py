import json

from tools.dataCollect import prepare_sound_data

# 使用示例
# 使用示例
train_json, test_json = prepare_sound_data("-6_dB_fan/fan/id_00/normal", "-6_dB_fan/fan/id_00/abnormal")

# 将结果写入 JSON 文件
with open('training_data.json', 'w') as f:
    json.dump(train_json, f)

with open('testing_data.json', 'w') as f:
    json.dump(test_json, f)

print("训练和测试数据已保存为 JSON 文件。")
