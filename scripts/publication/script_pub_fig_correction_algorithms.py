import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# %%
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 9), squeeze=True)

n = 4
m = 12
t = range(n)
ds1 = 4 * np.array([0, 0, 1, 1])
s1_upper = 6 * np.ones(n) + ds1
s1_lower = 2 * np.ones(n) + ds1
s1 = [None, 4, 5, None]
s1_corr = [None, 5, 6, None]

axes[0].plot(t, s1_upper, "r", marker="o", label="S_upper")
axes[0].plot(t, s1_lower, "b", marker="o", label="S_lower")
axes[0].fill_between(t, s1_lower, s1_upper, facecolor="tab:orange", alpha=0.4)

axes[0].plot(t[1], s1[1], "k", marker="o")
axes[0].plot(t[2], s1[2], "k", marker="x")
axes[0].plot(t, s1, "k", label="s_t")
axes[0].plot(t, s1_corr, "k", marker="o", linestyle="--")

axes[0].fill_between([1, 2], [4, 3], [4, 5], facecolor="tab:red", alpha=0.3)

ds2 = -4 * np.array([0, 0, 1, 1])
s2_upper = 10 * np.ones(n) + ds2
s2_lower = 6 * np.ones(n) + ds2
s2 = [None, 8, 7, None]
s2_corr = [None, 7, 6, None]

axes[1].plot(t, s2_upper, "r", marker="o", label="S_upper")
axes[1].plot(t, s2_lower, "b", marker="o", label="S_lower")
axes[1].fill_between(t, s2_lower, s2_upper, facecolor="tab:orange", alpha=0.4)

axes[1].plot(t[1], s2[1], "k", marker="o")
axes[1].plot(t[2], s2[2], "k", marker="x")
axes[1].plot(t, s2, "k", label="s_t")
axes[1].plot(t, s2_corr, "k", marker="o", linestyle="--")

axes[1].fill_between([1, 2], [8, 7], [8, 9], facecolor="tab:red", alpha=0.3)

for ax in axes:
    ax.grid(True)
    # ax.set_xlabel("$time$")
    # ax.set_ylabel("$S_t$")
    ax.set_ylim(0.5, m)
    ax.set_xlim(0.5, 2.5)
    # ax.xticks(x, my_xticks)
    # ax.set_xticks(range(n), [])
    # ax.set_yticks(range(m))
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    # ax.axis('off')
    # ax.legend("lower right")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    loc = ticker.MultipleLocator(base=1.0)  # this locator puts ticks at regular intervals
    ax.xaxis.set_major_locator(loc)

plt.tight_layout(pad=10.0)
plt.show()
