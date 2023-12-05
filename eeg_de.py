import pandas as pd
import numpy as np
import scipy.signal

# 读取脑电数据
eeg_data = pd.read_csv('processed_eeg_data.csv', header=0)

# 设置参数
fs = 128  # 采样频率
segment_length = fs * 5  # 每个分段的长度
n_segments = len(eeg_data) // segment_length  # 分段总数
n_channels = 32  # 电极通道数
bands = {'delta': (1, 3), 'theta': (4, 7), 'alpha': (8, 13), 'beta': (14, 30)}  # 定义频带
entropy_length = fs // 2  # 计算差分熵的窗口大小（0.5秒）

# 初始化结果数组
result = np.zeros((n_segments, n_channels, len(bands), 10))  # (5973, 32, 4, 10)


def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    y = scipy.signal.lfilter(b, a, data)
    return y


def differential_entropy(signal):
    return -np.sum(np.log2(np.abs(np.fft.fft(signal)) ** 2))


# 处理每个分段
for i in range(n_segments):
    segment = eeg_data.iloc[i * segment_length:(i + 1) * segment_length, 1:]

    for channel in range(n_channels):
        channel_data = segment.iloc[:, channel]

        for j, (band, (low, high)) in enumerate(bands.items()):
            # 带通滤波
            filtered_data = bandpass_filter(channel_data, low, high, fs)

            for k in range(10):
                # 0.5秒窗口的差分熵
                start = k * entropy_length
                end = (k + 1) * entropy_length
                entropy = differential_entropy(filtered_data[start:end])
                result[i, channel, j, k] = entropy

# 保存为.npy文件
np.save('eeg_features.npy', result)
