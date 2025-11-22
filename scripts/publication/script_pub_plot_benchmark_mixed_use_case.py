import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

from pyflexad.utils.algorithms import Algorithms
from pyflexad.utils.benchmark import Benchmark
from pyflexad.utils.file_utils import FileUtils

file = os.path.join(FileUtils.data_dir, "paper_04", "test_V06.pickle")
file_lpvg = os.path.join(FileUtils.data_dir, "paper_04", "test_V06_lpvg_gurobi.pickle")

bm = Benchmark.from_pickle(file)
bm_lpvg = Benchmark.from_pickle(file_lpvg)

# %%
title = "$Benchmark \\: with \\: 5000  \\: Households$\n"
flex_indices = {"200": 0, "500": 1, "1000": 2}
linestyles = {"200": ":", "500": "--", "1000": "-"}
linewidth = 4
markersize = 20
fontsize = 15
legendfontsize = 15
titlefontsize = 20
axisfontsize = 20

markeredgecolor = "k"
colors = {Algorithms.CENTRALIZED: "tab:red",
          Algorithms.IABVG_JIT: "tab:blue",
          Algorithms.LPVG_GUROBIPY: "tab:green"}
markers = {Algorithms.CENTRALIZED: "o",
           Algorithms.IABVG_JIT: "^",
           Algorithms.LPVG_GUROBIPY: "X"}
algorithm_names = {Algorithms.CENTRALIZED: "Centralized",
                   Algorithms.IABVG_JIT: "EACH-Aggregation",
                   Algorithms.LPVG_GUROBIPY: "LPCH-Aggregation"}

# %%
record = []
for algorithm in [Algorithms.CENTRALIZED, Algorithms.IABVG_JIT, Algorithms.LPVG_GUROBIPY]:
    _bm = bm_lpvg if algorithm == Algorithms.LPVG_GUROBIPY else bm

    for i, d in enumerate(_bm.d_list):
        for j, n in enumerate(_bm.n_flexibilities_list):
            upr = _bm.memory[algorithm]["upr"][i, j]
            total_time = _bm.memory[algorithm]["cpu_times"][i, j]
            opt_time = _bm.memory[algorithm]["optimize_times"][i, j]

            try:
                approx_time = _bm.memory[algorithm]["aggregate_times"][i, j]
            except KeyError:
                approx_time = pd.NA

            record.append(
                {"Algorithm": algorithm_names[algorithm], "No. Flexibilities": n, "No. Time Periods": d,
                 "UPR (%)": upr,
                 "Total Time (s)": total_time, "Optimization Time (s)": opt_time, "Approximation Time (s)": approx_time}
            )
df = pd.DataFrame.from_records(record)
df = df.round(decimals=2)
df.to_excel(os.path.join(FileUtils.data_dir, "paper_04", "benchmark_results.xlsx"))

# %%
# fig, axes = plt.subplots(figsize=(20, 14), ncols=3, squeeze=True)
fig, axes = plt.subplots(figsize=(30, 25), nrows=2, ncols=2, squeeze=False)

for n_flex_str, j in flex_indices.items():
    label_central = f"${algorithm_names[Algorithms.CENTRALIZED]}:$ ${n_flex_str}$ $Flexibilities$"
    label_iabvg = f"${algorithm_names[Algorithms.IABVG_JIT]}:$ ${n_flex_str}$ $Flexibilities$"
    label_lpvg = f"${algorithm_names[Algorithms.LPVG_GUROBIPY]}:$ ${n_flex_str}$ $Flexibilities$"

    """Total Times"""
    total_times_central = bm.memory[Algorithms.CENTRALIZED]["cpu_times"][:, j]
    total_times_iabvg = bm.memory[Algorithms.IABVG_JIT]["cpu_times"][:, j]

    axes[0, 0].semilogy(bm.d_list, total_times_central, linestyle=linestyles[n_flex_str], linewidth=linewidth,
                        markersize=markersize, markeredgecolor=markeredgecolor,
                        marker=markers[Algorithms.CENTRALIZED],
                        color=colors[Algorithms.CENTRALIZED], label=label_central)
    axes[0, 0].semilogy(bm.d_list, total_times_iabvg, linestyle=linestyles[n_flex_str], linewidth=linewidth,
                        markersize=markersize, markeredgecolor=markeredgecolor,
                        marker=markers[Algorithms.IABVG_JIT],
                        color=colors[Algorithms.IABVG_JIT], label=label_iabvg)

    if n_flex_str == "200":
        total_times_lpvg = bm_lpvg.memory[Algorithms.LPVG_GUROBIPY]["cpu_times"][:, j]
        axes[0, 0].semilogy(bm.d_list, total_times_lpvg, linestyle=linestyles[n_flex_str], linewidth=linewidth,
                            markersize=markersize, markeredgecolor=markeredgecolor,
                            marker=markers[Algorithms.LPVG_GUROBIPY],
                            color=colors[Algorithms.LPVG_GUROBIPY], label=label_lpvg)

    """Approximation Times"""
    approx_times_iabvg = bm.memory[Algorithms.IABVG_JIT]["aggregate_times"][:, j]

    axes[0, 1].semilogy(bm.d_list, approx_times_iabvg, linestyle=linestyles[n_flex_str], linewidth=linewidth,
                        markersize=markersize, markeredgecolor=markeredgecolor,
                        marker=markers[Algorithms.IABVG_JIT],
                        color=colors[Algorithms.IABVG_JIT], label=label_iabvg)
    if n_flex_str == "200":
        approx_times_lpvg = bm_lpvg.memory[Algorithms.LPVG_GUROBIPY]["aggregate_times"][:, j]
        axes[0, 1].semilogy(bm.d_list, approx_times_lpvg, linestyle=linestyles[n_flex_str], linewidth=linewidth,
                            markersize=markersize, markeredgecolor=markeredgecolor,
                            marker=markers[Algorithms.LPVG_GUROBIPY],
                            color=colors[Algorithms.LPVG_GUROBIPY], label=label_lpvg)

    """Optimize Times"""
    optimize_times_central = bm.memory[Algorithms.CENTRALIZED]["optimize_times"][:, j]
    optimize_times_iabvg = bm.memory[Algorithms.IABVG_JIT]["optimize_times"][:, j]

    axes[1, 0].semilogy(bm.d_list, optimize_times_central, linestyle=linestyles[n_flex_str], linewidth=linewidth,
                        markersize=markersize, markeredgecolor=markeredgecolor,
                        marker=markers[Algorithms.CENTRALIZED],
                        color=colors[Algorithms.CENTRALIZED], label=label_central)
    axes[1, 0].semilogy(bm.d_list, optimize_times_iabvg, linestyle=linestyles[n_flex_str], linewidth=linewidth,
                        markersize=markersize, markeredgecolor=markeredgecolor,
                        marker=markers[Algorithms.IABVG_JIT],
                        color=colors[Algorithms.IABVG_JIT], label=label_iabvg)
    if n_flex_str == "200":
        optimize_times_lpvg = bm_lpvg.memory[Algorithms.LPVG_GUROBIPY]["optimize_times"][:, j]
        axes[1, 0].semilogy(bm.d_list, optimize_times_lpvg, linestyle=linestyles[n_flex_str], linewidth=linewidth,
                            markersize=markersize, markeredgecolor=markeredgecolor,
                            marker=markers[Algorithms.LPVG_GUROBIPY],
                            color=colors[Algorithms.LPVG_GUROBIPY], label=label_lpvg)

    axes[0, 0].set_title(r"$Total \: Computation \: Time$", fontsize=titlefontsize)
    axes[0, 1].set_title(r"$Approximation \: Time$", fontsize=titlefontsize)
    axes[1, 0].set_title(r"$Optimization \: Time$", fontsize=titlefontsize)

    # axes[0, 0].set_yticks([10 ** x for x in range(0, 4)] + [3000])
    # axes[0, 1].set_yticks([10 ** x for x in range(0, 4)] + [3000])
    # axes[1, 0].set_yticks([0.05] + [10 ** x for x in range(-1, 3)] + [400])

    axes[0, 0].set_ylim(bottom=1, top=4000)
    axes[0, 1].set_ylim(bottom=1, top=4000)
    axes[1, 0].set_ylim(bottom=0.005, top=400)
    axes[1, 1].set_ylim(bottom=-1, top=30)

y_locator = ticker.LogLocator(base=10.0, subs="auto")
# y_locator = ticker.LogLocator(base=10.0)
x_locator = ticker.MultipleLocator(base=12)
# y_formatter = ticker.ScalarFormatter(useMathText=True)
y_formatter = ticker.FuncFormatter(lambda y, _: '{:.5g}'.format(y))
# y_formatter.set_scientific(False)

for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]:
    ax.set_ylabel(r"$CPU \: Time \: (s)$", fontsize=axisfontsize)
    ax.set_xlabel(r"$Time \: Periods$", fontsize=axisfontsize)
    ax.grid(True)
    # ax.legend(title="$Legend$", loc="upper left", shadow=True, fancybox=True)

    # ax.legend(title="$Legend$", bbox_to_anchor=(1.4, 1), shadow=True, fancybox=True)
    # ax.set_ylim(None, 1e4)
    ax.set_xlim(20, 100)
    # ax.set_xticks(bm.d_list)
    ax.yaxis.set_major_locator(y_locator)
    ax.yaxis.set_major_formatter(y_formatter)
    ax.xaxis.set_major_locator(x_locator)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    # ax.tick_params(axis='both', which='minor', labelsize=8)

for ax in [axes[0, 0], axes[0, 1]]:
    ax.set_yticks([
        1, 2, 4, 6, 8,
        10, 20, 40, 60, 80,
        100, 200, 400, 600, 800,
        1000, 2000, 4000])

axes[1, 0].set_yticks([0.005, 0.01, 0.02, 0.04, 0.06, 0.08,
                       0.1, 0.2, 0.4, 0.6, 0.8,
                       1, 2, 4, 6, 8,
                       10, 20, 40, 60, 80,
                       100, 200, 400])

# plt.suptitle(title, fontsize=16)
# plt.tight_layout()
# plt.show()

ax = axes[1, 1]
# fig, ax = plt.subplots(figsize=(10, 14), ncols=1, squeeze=True)

for n_flex_str, j in flex_indices.items():
    label_central = f"${algorithm_names[Algorithms.CENTRALIZED]}:$ ${n_flex_str}$ $Flexibilities$"
    label_iabvg = f"${algorithm_names[Algorithms.IABVG_JIT]}:$ ${n_flex_str}$ $Flexibilities$"
    label_lpvg = f"${algorithm_names[Algorithms.LPVG_GUROBIPY]}:$ ${n_flex_str}$ $Flexibilities$"

    """UPR"""
    upr_iabvg = bm.memory[Algorithms.IABVG_JIT]["upr"][:, j]
    ax.plot(bm.d_list, upr_iabvg, linestyle=linestyles[n_flex_str], linewidth=linewidth, markersize=markersize,
            markeredgecolor=markeredgecolor,
            marker=markers[Algorithms.IABVG_JIT],
            color=colors[Algorithms.IABVG_JIT], label=label_iabvg)

    if n_flex_str == "200":
        upr_lpvg = bm_lpvg.memory[Algorithms.LPVG_GUROBIPY]["upr"][:, j]
        ax.plot(bm.d_list, upr_lpvg, linestyle=linestyles[n_flex_str], linewidth=linewidth, markersize=markersize,
                markeredgecolor=markeredgecolor,
                marker=markers[Algorithms.LPVG_GUROBIPY],
                color=colors[Algorithms.LPVG_GUROBIPY], label=label_lpvg)

# ax.set_title("$Optimization \: Unused \: Potential \: Ratio$")
ax.set_title(r"$Approximation \: Quality$", fontsize=titlefontsize)
ax.set_yticks(range(-1, 31, 1))
x_locator = ticker.MultipleLocator(base=12)
ax.set_ylabel(r"$UPR \: (\%)$", fontsize=axisfontsize)
ax.set_xlabel(r"$Time \: Periods$", fontsize=axisfontsize)
ax.grid(True)
ax.tick_params(axis='both', which='major', labelsize=fontsize)
# ax.legend(title="$Legend$", loc="upper left", shadow=True, fancybox=True)
# axes[0].legend(title="$Legend$", bbox_to_anchor=(5.3, 1), shadow=True, fancybox=True)
# axes[0].legend(title="$Legend$", bbox_to_anchor=(0.65, -0.05), shadow=True, fancybox=True)
axes[0, 0].legend(
    # title="$Legend$", title_fontsize=fontsize,
    fontsize=legendfontsize,
    # loc="upper left",
    loc="lower right",
    shadow=True, fancybox=True, frameon=True, edgecolor="grey",
    handlelength=3.5, borderpad=1.1, labelspacing=1.1)

ax.set_xlim(20, 100)
ax.xaxis.set_major_locator(x_locator)

# plt.suptitle(title, fontsize=16)
plt.tight_layout()
plt.show()

fig.savefig(os.path.join(FileUtils.data_dir, "paper_04", "pub_plot_benchmark_mixed_use_case_power_opt.pdf"))
