import os
import librosa
import numpy as np
from sklearn.decomposition import PCA
def extract_signal_features(signal, sr, n_mels=64, frames=5, n_fft=1024):
    # Compute a mel-scaled spectrogram:
    mel_spectrogram = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_fft=n_fft,
        n_mels=n_mels
    )

    # Convert to decibel (log scale for amplitude):
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Generate an array of vectors as features for the current signal:
    features_vector_size = log_mel_spectrogram.shape[1] - frames + 1

    # Skips short signals:
    dims = frames * n_mels
    if features_vector_size < 1:
        return np.empty((0, dims), np.float32)

    # Build N sliding windows (=frames) and concatenate them to build a feature vector:
    features = np.zeros((features_vector_size, dims), np.float32)
    for t in range(frames):
        features[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t:t + features_vector_size].T

    return features


def load_sound_file(wav_name, mono=False, channel=0):
    multi_channel_data, sampling_rate = librosa.load(wav_name, sr=None, mono=mono)
    signal = np.array(multi_channel_data)[channel, :]

    return signal, sampling_rate


def reduce_features_to_1d(features, target_length=1024):
    pca = PCA(n_components=target_length)
    reduced_features = pca.fit_transform(features)
    return reduced_features.flatten()  # 返回一维数组

def load_features(file_list):
    # 假设这是加载特征的函数，返回特征数组
    features = []
    for file in file_list:
        signal, sr = load_sound_file(file)  # 加载声音文件
        feature = extract_signal_features(signal,
                                          sr,
                                          n_mels=64,
                                          frames=5,
                                          n_fft=1024)  # 提取特征的函数

        features.append(reduce_features_to_1d(feature,256))
    return np.array(features)


def prepare_sound_data(normal_path, anomaly_path):
    # 设置正常和异常声音文件的路径
    anomaly_files = [os.path.join(anomaly_path, file) for file in os.listdir(anomaly_path)]
    normal_files = [os.path.join(normal_path, file) for file in os.listdir(normal_path)]

    # 创建测试数据文件和标签
    test_files = normal_files[int(len(normal_files)*0.8):] + anomaly_files[int(len(anomaly_files)*0.8):]
    test_labels = np.hstack((np.zeros(len(normal_files)-int(len(normal_files)*0.8)), np.ones(len(anomaly_files)-int(len(anomaly_files)*0.8))))

    # 训练数据文件
    train_files = normal_files[:int(len(normal_files)*0.8)] + anomaly_files[:int(len(anomaly_files)*0.8)]
    train_labels = np.hstack((np.zeros(int(len(normal_files)*0.8)), np.ones(int(len(anomaly_files)*0.8))))

    def convert_features_to_strings(features):
        # 将二维数组的特征转换为字符串
        return [','.join(map(str, feature)) for feature in features]

    # 加载特征
    train_data = load_features(train_files)
    test_data = load_features(test_files)

    # 将特征转换为字符串
    train_data_strings = convert_features_to_strings(train_data)
    test_data_strings = convert_features_to_strings(test_data)

    # 构建 JSON 格式
    training_data_json = []
    for features, label in zip(train_data_strings, train_labels):
        training_data_json.append({
            "Question": features,  # 特征字符串
            "Response": int(label)  # 标签转换为整数
        })

    testing_data_json = []
    for features, label in zip(test_data_strings, test_labels):
        testing_data_json.append({
            "Question": features,  # 特征字符串
            "Response": int(label)  # 标签转换为整数
        })

    return training_data_json, testing_data_json
