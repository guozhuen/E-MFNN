import numpy as np

# 定义电极通道到11x11布局的映射
electrode_mapping = {
    'Cz': (4, 4), 'Fz': (3, 4), 'Fp1': (0, 3), 'F7': (1, 1), 'F3': (2, 3),
    'FC1': (3, 3), 'C3': (4, 3), 'FC5': (2, 1), 'FT9': (0, 0), 'T7': (3, 0),
    'CP5': (2, 7), 'CP1': (5, 3), 'P3': (5, 4), 'P7': (1, 7), 'PO7': (0, 6),
    'O1': (0, 5), 'Pz': (6, 4), 'Oz': (7, 4), 'O2': (7, 5), 'PO8': (7, 6),
    'P8': (6, 7), 'P4': (6, 5), 'CP2': (5, 5), 'CP6': (5, 7), 'T8': (6, 0),
    'FT10': (7, 0), 'FC6': (5, 1), 'C4': (5, 4), 'FC2': (4, 5), 'F4': (3, 5),
    'F8': (2, 7), 'Fp2': (7, 3)
}

# 加载脑电数据
eeg_data = np.load('eeg_features.npy')  # 假设数据文件名为 eeg.npy

# 初始化新的数据数组
new_shape = (eeg_data.shape[0], 11, 11, eeg_data.shape[2], eeg_data.shape[3])
new_eeg_data = np.zeros(new_shape)

# 转换每个样本的电极通道数据到9x9布局
for sample_idx in range(eeg_data.shape[0]):
    for channel_idx, (row, col) in enumerate(electrode_mapping.values()):
        new_eeg_data[sample_idx, row, col, :, :] = eeg_data[sample_idx, channel_idx, :, :]

# 保存转换后的数据
np.save('eeg_reshaped.npy', new_eeg_data)
