import math

import matplotlib.pyplot as plt

from .utils import D_TYPE


def plot1(data: D_TYPE, id, sort=False, C=2, bins=200, n_std=3):
    if sort:
        data = dict(sorted(data.items()))
    C = min(C, len(data))
    R = math.ceil(len(data) / C)
    plt.figure(figsize=(4 * C, 3 * R))
    i = 0
    for k, v in data.items():
        i += 1
        plt.subplot(R, C, i)
        title = f"{k} {v.shape}"
        if ".hist" in k:
            v = v.flatten()
            mu, std = v.mean(), v.std()
            bins_k = min(len(v), bins)
            range = [mu - std * n_std, mu + std * n_std]
            plt.hist(v, bins_k, range=range)
            title += f"\n mu={mu:.2g}, std={std:.2g}"
        elif ".img" in k:
            plt.imshow(v, cmap="gray")
        else:
            plt.plot(v)
        plt.title(title)
    plt.tight_layout()
    plt.savefig(id)
    plt.close()
