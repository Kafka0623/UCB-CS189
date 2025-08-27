import numpy as np

def split_train_val(X, y, val_size, seed=623):
    """将 (X, y) 打乱后划分为训练与验证。
    val_size: 若为int表示个数；若为float表示比例"""
    n = X.shape[0]
    rng = np.random.default_rng(seed) #设置随机数生成器
    idx = rng.permutation(n) # 打乱索引
    v = int(round(n * val_size)) if isinstance(val_size, float) else int(val_size)
    val_idx, tr_idx = idx[:v], idx[v:]
    return X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]

def accuracy(y_true, y_pred):
    """unweighted accuracy"""
    y_true = np.asarray(y_true).ravel() # 展平
    y_pred = np.asarray(y_pred).ravel()
    return float((y_true == y_pred).mean())
