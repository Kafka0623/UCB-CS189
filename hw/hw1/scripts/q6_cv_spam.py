# scripts/q6_cv_spam.py
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from q3_utils import accuracy

def _load_spam():
    return np.load("../data/spam-data.npz")

def geom_grid(cmin=1e-4, cmax=2e-1, num=14):
    """
    更稳的几何网格：覆盖更小的 C（更强正则）到你原先的量级
    """
    return np.geomspace(cmin, cmax, num=num)

def cv_spam_C(k=5,
              repeats=3,
              Cs=None,
              seed=42,
              max_iter=20000,
              tol=5e-4,
              class_weight=None,
              two_stage=False,
              standardize=True):
    """
    重复的 5 折分层交叉验证选择 C（默认重复3次）：
    1 每次重复都用不同随机种子做 StratifiedKFold(shuffle=True)
    2 折内 StandardScaler 拟合在训练折、应用到验证折（避免泄漏）
    3 模型：LinearSVC(dual=False, loss='squared_hinge')
    返回 best_C 与 records: List[(C, all_fold_scores(list), mean_acc)]
    """
    d = _load_spam()
    X = d["training_data"].astype(np.float32)
    y = d["training_labels"].ravel().astype(int)

    if Cs is None:
        Cs = geom_grid(1e-4, 2e-1, 14)

    def scan(cs, tag="scan"):
        best_C, best_mean = None, -1.0
        records = []
        print(f"[{tag}] Cs = {cs}")
        for C in cs:
            all_scores = []
            # 重复 k 折
            for rep in range(repeats):
                skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed + rep)
                for tr_idx, val_idx in skf.split(X, y):
                    if standardize:
                        scaler = StandardScaler()
                        Xtr = scaler.fit_transform(X[tr_idx])
                        Xval = scaler.transform(X[val_idx])
                    else:
                        Xtr, Xval = X[tr_idx], X[val_idx]

                    clf = LinearSVC(
                        C=float(C),
                        loss="squared_hinge",
                        dual=False,          # n_samples >> n_features 时更合适
                        max_iter=int(max_iter),
                        tol=float(tol),
                        class_weight=class_weight,
                        random_state=0
                    )
                    clf.fit(Xtr, y[tr_idx])
                    pred = clf.predict(Xval)
                    all_scores.append(accuracy(y[val_idx], pred))

            mean_acc = float(np.mean(all_scores))
            records.append((float(C), all_scores, mean_acc))
            # 打印每个 C 的均值与分位点，便于看稳定性
            q25, q50, q75 = np.percentile(all_scores, [25, 50, 75])
            print(f"C={C:.6g}  mean={mean_acc:.4f}  (Q25={q25:.4f}, median={q50:.4f}, Q75={q75:.4f})")
            if mean_acc > best_mean:
                best_mean, best_C = mean_acc, float(C)
        return best_C, best_mean, records

    # 第一阶段：较宽网格
    bestC1, bestA1, rec1 = scan(Cs, tag="coarse")

    # 第二阶段（可选）：以 log10(bestC1) 的邻域再细扫
    if two_stage and len(Cs) >= 3:
        logs = np.log10(Cs)
        means = [r[2] for r in rec1]
        i = int(np.argmax(means))
        lo = logs[max(i - 1, 0)]
        hi = logs[min(i + 1, len(logs) - 1)]
        fine_Cs = np.logspace(lo, hi, num=9)
        bestC2, bestA2, rec2 = scan(fine_Cs, tag="fine")
        if bestA2 >= bestA1:
            best_C, best_mean, records = bestC2, bestA2, rec1 + rec2
        else:
            best_C, best_mean, records = bestC1, bestA1, rec1
    else:
        best_C, best_mean, records = bestC1, bestA1, rec1

    print(f"\n[Spam] Best C = {best_C}, CV mean accuracy (repeated {repeats}×{k}-fold) = {best_mean:.4f}")
    return best_C, records

if __name__ == "__main__":
    # 基线：重复 3 次的 5 折；开启二阶段细化；使用标准化；可试一版 balanced 再对比
    print("== Unbalanced ==")
    best_C, records = cv_spam_C(
        k=5,
        repeats=3,
        Cs=geom_grid(1e-4, 2e-1, 14),
        seed=42,
        max_iter=20000,
        tol=5e-4,
        class_weight=None,
        two_stage=True,
        standardize=True
    )
    print("\n== Balanced (optional) ==")
    best_C_bal, _ = cv_spam_C(
        k=5,
        repeats=3,
        Cs=geom_grid(1e-4, 2e-1, 14),
        seed=42,
        max_iter=20000,
        tol=5e-4,
        class_weight="balanced",
        two_stage=True,
        standardize=True
    )
    print(f"\nFinal suggestion -> C: {best_C} (unbalanced), or {best_C_bal} (balanced)")
