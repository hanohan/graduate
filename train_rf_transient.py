import os
from typing import List, Tuple

import numpy as np
import soundfile as sf
import librosa
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# ===================== 全局配置（按常用配置给出） =====================

# 采样率（你说已改为 44.1k）
SR = 44100

# STFT / 特征提取参数（常用设置）
N_FFT = 1024
HOP_LENGTH = 512
N_MFCC = 13

# 数据根目录
CLEAN_ROOT = "speech_transient_dataset"
TRANSIENT_ROOT = "speech_transient_dataset_transient"

# 用哪些 SNR 生成带噪输入（None 代表干净语音也参与训练）
SNR_LIST = [None, 5, -5]


# ===================== 工具函数 =====================

def add_white_noise(x: np.ndarray, snr_db: float) -> np.ndarray:
    """
    按给定 SNR(dB) 向信号加入白噪声。
    SNR = 10*log10(P_signal / P_noise)
    """
    if snr_db is None:
        return x

    x = x.astype(float)
    p_signal = np.mean(x**2) + 1e-12
    snr_linear = 10 ** (snr_db / 10.0)
    p_noise = p_signal / snr_linear

    noise = np.random.randn(len(x))
    p_noise_raw = np.mean(noise**2) + 1e-12
    noise = noise * np.sqrt(p_noise / p_noise_raw)

    return x + noise


def load_pair(clean_path: str, transient_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    读取一对 (clean, transient)，统一采样率和长度。
    """
    y_clean, _ = librosa.load(clean_path, sr=SR, mono=True)
    y_tr, _ = librosa.load(transient_path, sr=SR, mono=True)

    L = min(len(y_clean), len(y_tr))
    y_clean = y_clean[:L]
    y_tr = y_tr[:L]
    return y_clean, y_tr


def extract_features_and_targets(
    y_clean: np.ndarray, y_tr: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    给定一条干净语音及其瞬态分量，生成多种 SNR 下的特征和帧级瞬态比例标签。
    返回：
        X: [n_frames_total, feature_dim]
        y: [n_frames_total,]
    """
    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []

    # 用 clean / transient 的 STFT 计算每帧能量，作为目标的“瞬态比例”
    S_clean = librosa.stft(
        y_clean, n_fft=N_FFT, hop_length=HOP_LENGTH, center=True
    )
    S_tr = librosa.stft(
        y_tr, n_fft=N_FFT, hop_length=HOP_LENGTH, center=True
    )

    E_clean = np.sum(np.abs(S_clean) ** 2, axis=0)
    E_tr = np.sum(np.abs(S_tr) ** 2, axis=0)
    ratio = E_tr / (E_clean + 1e-12)  # 帧级瞬态比例

    for snr in SNR_LIST:
        y_in = add_white_noise(y_clean, snr) if snr is not None else y_clean

        # 特征：MFCC + 能量 + 频谱质心、带宽等常用特征
        mfcc = librosa.feature.mfcc(
            y=y_in,
            sr=SR,
            n_mfcc=N_MFCC,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
        )  # [n_mfcc, T]
        zcr = librosa.feature.zero_crossing_rate(
            y_in, frame_length=N_FFT, hop_length=HOP_LENGTH
        )  # [1, T]
        centroid = librosa.feature.spectral_centroid(
            y=y_in, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH
        )  # [1, T]
        bandwidth = librosa.feature.spectral_bandwidth(
            y=y_in, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH
        )  # [1, T]

        # 对齐时间帧数（防止因为边界导致长度略有出入）
        T = min(
            mfcc.shape[1],
            zcr.shape[1],
            centroid.shape[1],
            bandwidth.shape[1],
            ratio.shape[0],
        )

        feats = np.vstack(
            [
                mfcc[:, :T],
                zcr[:, :T],
                centroid[:, :T],
                bandwidth[:, :T],
            ]
        )  # [feature_dim, T]

        X_list.append(feats.T)  # [T, feature_dim]
        y_list.append(ratio[:T])

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    return X, y


def build_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """
    遍历 CLEAN_ROOT 下所有 wav，与 TRANSIENT_ROOT 中的对应文件配对，
    构建整套训练数据 (X, y)。
    """
    X_all: List[np.ndarray] = []
    y_all: List[np.ndarray] = []

    for root, _dirs, files in os.walk(CLEAN_ROOT):
        rel_dir = os.path.relpath(root, CLEAN_ROOT)
        if rel_dir == ".":
            rel_dir = ""

        for fname in files:
            if not fname.lower().endswith(".wav"):
                continue

            clean_path = os.path.join(root, fname)
            base, _ext = os.path.splitext(fname)
            tr_fname = f"{base}_transient.wav"
            tr_path = os.path.join(TRANSIENT_ROOT, rel_dir, tr_fname)

            if not os.path.exists(tr_path):
                print(f"[warn] 找不到对应瞬态文件，跳过: {tr_path}")
                continue

            print(f"[pair] {clean_path}  <->  {tr_path}")

            try:
                y_clean, y_tr = load_pair(clean_path, tr_path)
                X, y = extract_features_and_targets(y_clean, y_tr)
                X_all.append(X)
                y_all.append(y)
            except Exception as e:
                print(f"[error] 处理 {clean_path} 时出错: {e}")

    if not X_all:
        raise RuntimeError("没有成功构建任何训练样本，请检查数据路径和文件命名。")

    X_total = np.vstack(X_all)
    y_total = np.concatenate(y_all)
    print(f"[info] 数据集大小: X={X_total.shape}, y={y_total.shape}")
    return X_total, y_total


def train_random_forest(X: np.ndarray, y: np.ndarray) -> RandomForestRegressor:
    """
    使用随机森林做回归，预测帧级瞬态比例。
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        n_jobs=-1,
        random_state=42,
    )
    print("[train] 随机森林开始训练...")
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"[eval] MSE = {mse:.6f}, R2 = {r2:.4f}")

    return rf


def main():
    print("=== 构建数据集（帧级瞬态比例回归，随机森林） ===")
    X, y = build_dataset()

    print("=== 训练随机森林模型 ===")
    rf = train_random_forest(X, y)

    # 简单示例：把模型保存为 .npz（包含参数和简单说明）
    # 由于 RandomForest 很难用手写方式存成纯 numpy，我们先用 joblib / pickle。
    try:
        import joblib

        joblib.dump(rf, "rf_transient_model.joblib")
        print("[save] 模型已保存到 rf_transient_model.joblib")
    except ImportError:
        print("[warn] 未安装 joblib，暂未保存模型。可 pip install joblib 后自行保存。")


if __name__ == "__main__":
    main()

