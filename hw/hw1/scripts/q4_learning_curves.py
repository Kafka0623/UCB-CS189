import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from q3_utils import split_train_val, accuracy

def _load(name):
    return np.load(f"../data/{name}-data.npz")

def run_curve(dataset, train_sizes, seed=623, outpng=None):
    d = _load(dataset)
    X, y = d["training_data"], d["training_labels"]

    if dataset == "mnist":
        # MNIST: (N, 1, 28, 28) -> (N, 784) 并缩放到 [0,1]
        X = X.astype(np.float32).reshape(len(X), -1) / 255.0
        Xtr, ytr, Xval, yval = split_train_val(X, y, val_size=10000, seed=seed)
    elif dataset == "spam":
        # Spam: 已经是二维 (N, d)，无需 reshape
        Xtr, ytr, Xval, yval = split_train_val(X, y, val_size=0.2, seed=seed)
    else:
        raise ValueError("dataset must be 'mnist' or 'spam'")

    tr_acc, val_acc, sizes_used = [], [], []
    for m in train_sizes:
        m = min(int(m), Xtr.shape[0])  # “ALL” 占位会被截断为可用上限
        sizes_used.append(m)

        # MNIST: n_samples >> n_features(784) → dual=False 往往更快更稳
        clf = LinearSVC(C=1.0, max_iter=10000, dual=False, random_state=0)

        clf.fit(Xtr[:m], ytr[:m].ravel())
        tr_acc.append(accuracy(ytr[:m], clf.predict(Xtr[:m])))
        val_acc.append(accuracy(yval, clf.predict(Xval)))
        print(f"{dataset}: n={m:6d}  train={tr_acc[-1]:.4f}  val={val_acc[-1]:.4f}")

    plt.figure()
    plt.plot(sizes_used, tr_acc, marker="o", label="train")
    plt.plot(sizes_used, val_acc, marker="s", label="validation")
    plt.xlabel("Number of training examples")
    plt.ylabel("Accuracy")
    plt.title(f"{dataset.upper()} learning curve (Linear SVM, C=1)")
    plt.legend(); plt.grid(True)
    if outpng:
        plt.savefig(outpng, dpi=160, bbox_inches="tight")
        print(f"saved: {outpng}")
    else:
        plt.show()

if __name__ == "__main__":
    # MNIST
    run_curve("mnist", [100, 200, 500, 1000, 2000, 5000, 10000], outpng="mnist_curve.png")
    # Spam（最后一个大数表示“ALL”，会被截断为可用样本上限）
    run_curve("spam",  [100, 200, 500, 1000, 2000, 99999999],   outpng="spam_curve.png")
