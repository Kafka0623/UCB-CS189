# scripts/q5_ultra_finetune_mnist_rff.py
# 目标：围绕你已得最优点 (gamma≈0.83907, m=3000, C=0.4) 做更细微调，
#      以期稳定突破 0.955，保持运行时间可控。
# 管线：imgnorm → RBFSampler → StandardScaler(with_mean=False) → LinearSVC(squared_hinge, dual=False)

import os
os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import StandardScaler
from q3_utils import split_train_val, accuracy

def _load_mnist():
    return np.load("../data/mnist-data.npz")

def _imgnorm(A):
    A = A.astype(np.float32)
    A = A - A.mean(axis=1, keepdims=True)
    n = np.linalg.norm(A, axis=1, keepdims=True) + 1e-6
    return A / n

def ultra_finetune(g_center=0.8390732253715566,
                   ncomps=(2800, 3000, 3200),
                   Cs=(0.34, 0.36, 0.38, 0.40, 0.42, 0.45),
                   g_factors=(0.90, 0.95, 1.00, 1.05, 1.10),
                   n_train=10000,
                   max_iter=20000,
                   tol=3e-4,
                   seed=42):
    """
    超精细微调：
    1 gamma：围绕 g_center 做 ±10% 的细化（默认5点）
    2 n_components：在 {2800,3000,3200} 小幅上下探测
    3 C：0.02 粒度微调
    4 训练子集 n_train（默认10000），验证集固定10000
    """
    d = _load_mnist()
    X = d["training_data"].reshape(len(d["training_data"]), -1).astype(np.float32)
    y = d["training_labels"].ravel()

    # 固定 10000 验证集
    Xtr_all, ytr_all, Xval_all, yval_all = split_train_val(X, y, val_size=10000, seed=seed)

    # 子集用于微调
    n_use = max(8000, int(n_train))
    Xsub0, ysub = Xtr_all[:n_use], ytr_all[:n_use]

    # 预处理
    Xsub = _imgnorm(Xsub0)
    Xval = _imgnorm(Xval_all)

    gammas = [float(g_center * f) for f in g_factors]
    ncomps = [int(m) for m in ncomps]
    Cs     = [float(c) for c in Cs]

    print(f"n_train(subset)={Xsub.shape[0]}, n_val={Xval.shape[0]}")
    print(f"gamma center={g_center:.6g}, grid={[float(f'{g:.6g}') for g in gammas]}")
    print(f"n_components grid={ncomps}, C grid={Cs}")
    print(f"solver: max_iter={max_iter}, tol={tol}, seed={seed}\n")

    best = {"gamma": None, "ncomp": None, "C": None, "acc": -1.0}

    for g in gammas:
        for m in ncomps:
            feat = RBFSampler(gamma=float(g), n_components=int(m), random_state=seed)
            Ztr = feat.fit_transform(Xsub)
            Zvl = feat.transform(Xval)

            scaler = StandardScaler(with_mean=False)
            Ztr = scaler.fit_transform(Ztr)
            Zvl = scaler.transform(Zvl)

            for C in Cs:
                clf = LinearSVC(
                    C=float(C),
                    loss="squared_hinge",
                    dual=False,
                    max_iter=int(max_iter),
                    tol=float(tol),
                    random_state=0
                )
                clf.fit(Ztr, ysub)
                acc = accuracy(yval_all, clf.predict(Zvl))
                print(f"gamma={g:.6g}  m={m:4d}  C={C:.3g}  val_acc={acc:.4f}")
                if acc > best["acc"]:
                    best = {"gamma": float(g), "ncomp": int(m), "C": float(C), "acc": float(acc)}

    print(f"\n[FAST-RFF-ULTRA] best: gamma={best['gamma']}, m={best['ncomp']}, C={best['C']}, val_acc={best['acc']:.4f}")
    return best

def main():
    ap = argparse.ArgumentParser(description="Q5 MNIST RFF ultra-fine tuning")
    ap.add_argument("--g_center", type=float, default=0.8390732253715566)
    ap.add_argument("--n_train", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_iter", type=int, default=20000)
    ap.add_argument("--tol", type=float, default=3e-4)
    ap.add_argument("--g_factors", type=str, default="0.90,0.95,1.0,1.05,1.10")
    ap.add_argument("--ncomps", type=str, default="2800,3000,3200")
    ap.add_argument("--Cs", type=str, default="0.34,0.36,0.38,0.40,0.42,0.45")
    args = ap.parse_args()

    def _parse_floats(s): return [float(x) for x in s.split(",") if x.strip()]
    def _parse_ints(s):   return [int(x) for x in s.split(",") if x.strip()]

    ultra_finetune(
        g_center=args.g_center,
        ncomps=_parse_ints(args.ncomps),
        Cs=_parse_floats(args.Cs),
        g_factors=_parse_floats(args.g_factors),
        n_train=args.n_train,
        max_iter=args.max_iter,
        tol=args.tol,
        seed=args.seed
    )

if __name__ == "__main__":
    main()
