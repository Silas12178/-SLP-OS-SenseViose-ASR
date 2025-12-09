# 本文件属于《基于 SenseVoice 的 ASR 可视化操作系统》
# 作者：Silas、Portnoy
# 版本号 S-ASR-25.12
# 版权所有 © 2025
import numpy as np
import librosa

# ====== 噪音抑制模块 ======
def _stft(y, n_fft=512, hop_length=256):
    """
    简单封装，确保 dtype 和参数一致
    """
    return librosa.stft(y, n_fft=n_fft, hop_length=hop_length)


def _istft(stft_matrix, hop_length=256, length=None):
    """
    STFT 逆变换
    """
    return librosa.istft(stft_matrix, hop_length=hop_length, length=length)


def _estimate_noise_spectrum(y, sr, n_fft=512, hop_length=256, noise_duration=0.5):
    """
    使用音频开头一段时间估计噪音频谱模板

    y: 一维 numpy 波形
    sr: 采样率
    noise_duration: 用来估计噪声的时长（秒）
    """
    if y.ndim != 1:
        y = np.asarray(y).reshape(-1)

    total_len = y.shape[0]
    noise_len = int(min(noise_duration * sr, total_len))
    if noise_len <= 0:
        noise_len = min(total_len, int(0.25 * sr))

    noise_segment = y[:noise_len]

    stft_noise = _stft(noise_segment, n_fft=n_fft, hop_length=hop_length)
    noise_mag = np.abs(stft_noise)

    # 对时间维度取平均，得到每个频率 bin 的平均噪声幅度
    noise_profile = np.mean(noise_mag, axis=1, keepdims=True)
    return noise_profile


def _spectral_subtraction(y, sr, n_fft=512, hop_length=256,
                          alpha=1.0, floor_ratio=0.02):
    """
    谱减法噪音抑制

    alpha: 噪声减去强度，>1 会更激进
    floor_ratio: 残留噪声地板比例，避免完全抹掉导致音乐感噪声
    """
    if y.ndim != 1:
        y = np.asarray(y).reshape(-1)

    length = len(y)
    stft_all = _stft(y, n_fft=n_fft, hop_length=hop_length)
    mag = np.abs(stft_all)
    phase = np.angle(stft_all)

    noise_profile = _estimate_noise_spectrum(y, sr, n_fft=n_fft, hop_length=hop_length)

    # 谱减
    clean_mag = mag - alpha * noise_profile
    # 不能为负，留噪声地板
    min_floor = floor_ratio * noise_profile
    clean_mag = np.maximum(clean_mag, min_floor)

    # 还原复数谱
    stft_clean = clean_mag * np.exp(1j * phase)
    y_clean = _istft(stft_clean, hop_length=hop_length, length=length)

    return y_clean.astype(np.float32)


def _wiener_filter(y, sr, n_fft=512, hop_length=256,
                   floor_ratio=0.02):
    """
    简单 Wiener 滤波版噪声抑制

    G = S / (S + N)
    其中 S ≈ max(mag^2 - N^2, 0)
    """
    if y.ndim != 1:
        y = np.asarray(y).reshape(-1)

    length = len(y)
    stft_all = _stft(y, n_fft=n_fft, hop_length=hop_length)
    mag = np.abs(stft_all)
    phase = np.angle(stft_all)

    noise_profile = _estimate_noise_spectrum(y, sr, n_fft=n_fft, hop_length=hop_length)
    noise_power = np.maximum(noise_profile ** 2, 1e-8)

    signal_power = mag ** 2 - noise_power
    signal_power = np.maximum(signal_power, floor_ratio * noise_power)

    gain = signal_power / (signal_power + noise_power)
    gain = np.clip(gain, 0.0, 1.0)

    clean_mag = mag * gain
    stft_clean = clean_mag * np.exp(1j * phase)
    y_clean = _istft(stft_clean, hop_length=hop_length, length=length)

    return y_clean.astype(np.float32)


def denoise_audio(y, sr, method="spectral"):
    """
    对外统一接口：

    y: 一维 numpy 波形，float32
    sr: 采样率
    method: "spectral" 或 "wiener"
    """
    if y is None:
        return y

    y = np.asarray(y, dtype=np.float32).reshape(-1)

    method = (method or "spectral").lower()

    if method == "wiener":
        return _wiener_filter(y, sr)
    else:
        # 默认用谱减法
        return _spectral_subtraction(y, sr)
