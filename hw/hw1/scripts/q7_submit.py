# scripts/q7_submit.py
# 第七问统一提交脚本（覆盖原文件）
# 1) MNIST: imgnorm → RBFSampler → StandardScaler(with_mean=False) → LinearSVC(squared_hinge, dual=False)
# 2) Spam : StandardScaler → LinearSVC(squared_hinge, dual=False)

import os
import sys
import numpy as np
from pathlib import Path
from sklearn.svm import LinearSVC
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import StandardScaler
from save_csv import results_to_csv
import subprocess

# 稳定性（Windows + MKL）
os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# 路径
THIS = Path(__file__).resolve()
SCRIPTS = THIS.parent
DATA = SCRIPTS.parent / "data"

# ---------------- 工具 ----------------
def run_check_and_assert(dataset: str, filename: str) -> None:
    """调用 check.py 进行离线校验，未通过则抛出 AssertionError。"""
    cmd = [sys.executable, str(SCRIPTS / "check.py"), dataset, filename]
    proc = subprocess.run(cmd, cwd=str(SCRIPTS), capture_output=True, text=True)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="")
    assert proc.returncode == 0, f"check.py failed for {dataset} on {filename}"

def _imgnorm(A: np.ndarray) -> np.ndarray:
    """逐样本零均值 + L2 归一化（与第5问一致）"""
    A = A.astype(np.float32)
    A = A - A.mean(axis=1, keepdims=True)
    n = np.linalg.norm(A, axis=1, keepdims=True) + 1e-6
    return A / n

# ---------------- MNIST（RFF + LinearSVC）----------------
def predict_mnist_rff(gamma: float,
                      n_components: int,
                      C: float,
                      max_iter: int = 20000,
                      seed: int = 42,
                      out_name: str = "submission_mnist.csv") -> tuple[float, int, float]:
    """
    管线：imgnorm → RBFSampler → StandardScaler(with_mean=False) → LinearSVC(squared_hinge, dual=False)
    使用全量训练集拟合，输出测试集预测并校验。
    """
    d = np.load(DATA / "mnist-data.npz")
    Xtr0 = d["training_data"].reshape(len(d["training_data"]), -1).astype(np.float32)
    ytr  = d["training_labels"].ravel()
    Xte0 = d["test_data"].reshape(len(d["test_data"]), -1).astype(np.float32)

    # 预处理
    Xtr = _imgnorm(Xtr0)
    Xte = _imgnorm(Xte0)

    # RFF + 标准化
    feat = RBFSampler(gamma=float(gamma), n_components=int(n_components), random_state=seed)
    Ztr = feat.fit_transform(Xtr)
    Zte = feat.transform(Xte)
    scaler = StandardScaler(with_mean=False)
    Ztr = scaler.fit_transform(Ztr)
    Zte = scaler.transform(Zte)

    # 线性 SVM
    clf = LinearSVC(C=float(C), loss="squared_hinge", dual=False,
                    max_iter=int(max_iter), tol=1e-4, random_state=0)
    clf.fit(Ztr, ytr)
    yhat = clf.predict(Zte).astype(int)

    # 导出并校验
    results_to_csv(yhat)  # -> scripts/submission.csv
    final_name = out_name
    os.replace(SCRIPTS / "submission.csv", SCRIPTS / final_name)
    print(f"[MNIST-RFF] saved: {final_name} (gamma={gamma}, m={n_components}, C={C})")
    run_check_and_assert("mnist", final_name)
    return float(gamma), int(n_components), float(C)

# ---------------- Spam（StandardScaler + LinearSVC）----------------
def predict_spam(C: float,
                 max_iter: int = 20000,
                 use_standardize: bool = True,
                 seed: int = 42,
                 out_name: str = "submission_spam.csv") -> float:
    """
    用全量 Spam 训练 + 预测并保存 CSV，返回使用的 C。
    第六问折内做了标准化，这里需保持一致：use_standardize=True。
    """
    d = np.load(DATA / "spam-data.npz", allow_pickle=False)
    Xtr = np.asarray(d["training_data"], dtype=np.float32)
    ytr = np.asarray(d["training_labels"]).ravel().astype(int)
    Xte = np.asarray(d["test_data"], dtype=np.float32)

    if use_standardize:
        scaler = StandardScaler(with_mean=True, with_std=True)
        Xtr = scaler.fit_transform(Xtr)
        Xte = scaler.transform(Xte)

    clf = LinearSVC(C=float(C), loss="squared_hinge", dual=False,
                    max_iter=int(max_iter), tol=1e-4, random_state=seed)
    clf.fit(Xtr, ytr)
    yhat = clf.predict(Xte).astype(int)

    results_to_csv(yhat)
    final_name = out_name
    os.replace(SCRIPTS / "submission.csv", SCRIPTS / final_name)
    print(f"[SPAM] saved: {final_name} (C={C}, standardize={use_standardize})")
    run_check_and_assert("spam", final_name)
    return float(C)

# ---------------- 主程序（已填入你的最优参数）----------------
if __name__ == "__main__":
    # —— MNIST：第5问微调后参数 ——
    MNIST_GAMMA = 0.755165902834401
    MNIST_NCOMP = 3200
    MNIST_C     = 0.38

    # —— Spam：第6问“重复5折CV”得到的最优 C（unbalanced）——
    SPAM_C       = 0.009287989765619788
    SPAM_USE_STD = True   # 与第六问一致：标准化=True

    # MNIST 提交
    predict_mnist_rff(
        gamma=MNIST_GAMMA,
        n_components=MNIST_NCOMP,
        C=MNIST_C,
        max_iter=20000,
        seed=42,
        out_name="submission_mnist.csv"
    )

    # Spam 提交
    predict_spam(
        C=SPAM_C,
        max_iter=20000,
        use_standardize=SPAM_USE_STD,
        seed=42,
        out_name="submission_spam.csv"
    )

    print("\nDone. Submissions saved and verified.")
