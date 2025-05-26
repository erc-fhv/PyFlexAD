import logging
import multiprocessing
import os.path
import pickle
from typing import Self

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from pyflexad.math.signal_vectors import SignalVectors
from pyflexad.optimization.centralized_controller import CentralizedController
from pyflexad.optimization.vertex_based_controller import VertexBasedController
from pyflexad.physical.energy_storage import EnergyStorage
from pyflexad.utils.algorithms import Algorithms
from pyflexad.utils.progress_bar import Progressbar
from pyflexad.utils.timer import Timer
from pyflexad.virtual.aggregator import Aggregator


class Benchmark:
    pre_compile_jit = False

    @classmethod
    def from_pickle(cls, file: str) -> Self:
        file = os.path.abspath(file)
        with open(file, "rb") as f:
            d = pickle.load(f)
        logging.info(f"Loaded from {file}")
        return cls(**d)

    @classmethod
    def from_algorithms(cls, algorithms: list[Algorithms] | tuple[Algorithms],
                        d_list: list[int] | tuple[int],
                        n_flexibilities_list: list[int] | tuple[int],
                        n_times: int, disaggregate: bool) -> Self:
        len_dimensions = len(d_list)
        len_n_flexibilities = len(n_flexibilities_list)
        memory = {
            k: dict(
                cpu_times=np.zeros((len_dimensions, len_n_flexibilities)),
                aggregate_times=np.zeros((len_dimensions, len_n_flexibilities)),
                optimize_times=np.zeros((len_dimensions, len_n_flexibilities)),
                disaggregate_times=np.zeros((len_dimensions, len_n_flexibilities)),
                upr=np.zeros((len_dimensions, len_n_flexibilities)),
            ) for k in algorithms
        }

        return cls(memory=memory, d_list=d_list, n_flexibilities_list=n_flexibilities_list, n_times=n_times,
                   disaggregate=disaggregate)

    def __init__(self, memory: dict[Algorithms, dict],
                 d_list: list[int] | tuple[int],
                 n_flexibilities_list: list[int] | tuple[int],
                 n_times: int,
                 disaggregate: bool) -> None:
        self.memory = memory
        self.d_list = d_list
        self.n_flexibilities_list = n_flexibilities_list
        self.n_times = n_times
        self.disaggregate = disaggregate

    def to_pickle(self, file: str) -> None:
        file = os.path.abspath(file)
        os.makedirs(os.path.dirname(file), exist_ok=True)

        d = self.__dict__
        with open(file, "wb") as fh:
            pickle.dump(d, fh)

        logging.info(f"Saved to {file}")

    def _run_single(self, c_opt: CentralizedController, dc_opt: VertexBasedController, esr_list: list[EnergyStorage],
                    d: int, g: int, *args):

        memory = {
            k: {sk: 0.0 for sk in ["cpu_times", "optimize_times", "aggregate_times", "disaggregate_times", "upr"]}
            for k in self.memory.keys()
        }

        # with Timer() as timer_sv:
        #     for _ in range(self.n_loops):
        signal_vectors = SignalVectors.new(d, g=g)

        """optimize best and worst case"""
        with Timer(n_runs=self.n_times) as timer_opt:
            for _ in range(self.n_times):
                p_best = c_opt.optimize(items=esr_list, minimize=True)

        if Algorithms.CENTRALIZED in self.memory:
            memory[Algorithms.CENTRALIZED]["optimize_times"] = timer_opt.dt
            memory[Algorithms.CENTRALIZED]["cpu_times"] = timer_opt.dt

        p_worst = c_opt.optimize(items=esr_list, minimize=False)

        for algorithm in self.memory:

            if algorithm == Algorithms.CENTRALIZED:
                """benchmark best case"""
                upr = dc_opt.calc_upr(power_approx=p_best, power_best=p_best, power_worst=p_worst)
            else:
                """benchmark approximations case"""

                """aggregate"""
                with Timer(n_runs=self.n_times) as timer_agg:
                    for _ in range(self.n_times):
                        agg = Aggregator.from_physical(esr_list, algorithm=algorithm, signal_vectors=signal_vectors)

                """optimize"""
                with Timer(n_runs=self.n_times) as timer_opt:
                    for _ in range(self.n_times):
                        p_approx, alphas = dc_opt.solve(vertices=agg.get_vertices())

                """disaggregate"""
                with Timer(n_runs=self.n_times) as timer_dis_agg:
                    if self.disaggregate:
                        for _ in range(self.n_times):
                            agg.disaggregate(alphas=alphas)

                """store results"""
                if algorithm in self.memory.keys():
                    time_total = (timer_agg.dt + timer_opt.dt + timer_dis_agg.dt)
                    memory[algorithm]["aggregate_times"] = timer_agg.dt
                    memory[algorithm]["optimize_times"] = timer_opt.dt
                    memory[algorithm]["disaggregate_times"] = timer_dis_agg.dt
                    memory[algorithm]["cpu_times"] = time_total

                upr = dc_opt.calc_upr(power_approx=p_approx, power_best=p_best, power_worst=p_worst)

            memory[algorithm]["upr"] = upr

        return memory

    def __fill_memory(self, memory: dict[str, dict[str, float]], i: int, j: int) -> None:
        for algorithm, value in self.memory.items():
            for kind in value.keys():
                self.memory[algorithm][kind][i, j] = memory[algorithm][kind]

    def run(self, c_opt: CentralizedController, dc_opt: VertexBasedController, esr_list: list[EnergyStorage], d: int,
            g: int, i: int, j: int) -> None:
        memory = self._run_single(c_opt=c_opt, dc_opt=dc_opt, esr_list=esr_list, d=d, g=g)
        self.__fill_memory(memory=memory, i=i, j=j)

    def run_batch(self, run_args: list[tuple[CentralizedController, VertexBasedController,
    list[EnergyStorage], int, int, int, int]]) -> None:
        num_tasks = len(run_args)
        pb = Progressbar(num_tasks)

        for i, run_arg in enumerate(run_args):
            pb.update(i)
            self.run(*run_arg)
            pb.update(i + 1)

    def run_parallel(self, run_args: list[tuple[CentralizedController, VertexBasedController,
    list[EnergyStorage], int, int, int, int]]) -> None:
        num_tasks = len(run_args)
        pb = Progressbar(num_tasks)

        with multiprocessing.Pool() as pool:
            pool_result = pool.starmap_async(self._run_single, run_args)
            pb.update_multiprocessing(pool_result)
            memory_list = pool_result.get()

        for index, run_arg in enumerate(run_args):
            _, _, _, _, _, i, j = run_arg
            self.__fill_memory(memory=memory_list[index], i=i, j=j)

    def _plot_2d(self, kind: str, optimizer_name: str, logy: bool = False) -> None:
        len_n_flexibilities = len(self.n_flexibilities_list)

        figsize = (10, 10 * len_n_flexibilities)
        fig, ax = plt.subplots(len_n_flexibilities, 1, squeeze=False, figsize=figsize)
        for j, n_flexibilities in enumerate(self.n_flexibilities_list):
            for key, values in self.memory.items():
                if logy:
                    ax[j, 0].semilogy(self.d_list, values[kind][:, j], "-*", label=key,
                                      color=key.get_color())
                else:
                    ax[j, 0].plot(self.d_list, values[kind][:, j], "-*", label=key,
                                  color=key.get_color())

                ax[j, 0].set_xlabel("Time Periods")
                ax[j, 0].set_ylabel("UPR (%)" if kind == "upr" else "CPU Time (s)")
                ax[j, 0].set_title(f"{optimizer_name}, Flexibilities = {n_flexibilities}")

            ax[j, 0].grid(True)
            ax[j, 0].legend()

    def _plot_3d(self, kind: str, optimizer_name: str) -> None:
        X, Y = np.meshgrid(self.d_list, self.n_flexibilities_list)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        for key, values in self.memory.items():
            ax.plot_wireframe(X, Y, values[kind].T, label=key, color=key.get_color())

        title = "UPR" if kind == "upr" else "CPU Time"
        ax.set_title(f"{optimizer_name}: {title}")
        ax.set_xlabel("Time Periods")
        ax.set_ylabel("Flexibilities")
        ax.set_zlabel("UPR (%)" if kind == "upr" else "CPU Time (s)")
        ax.legend()

        """rotate to 2d view"""
        if X.shape[1] == 1:
            """y-axis (=n_flex= vs. z-axis"""
            ax.view_init(elev=0, azim=0, roll=0)
            ax.set_xlabel("")
            ax.set_xticks([])
            ax.set_ylabel(f"Flexibilities @ {self.d_list[0]} Time Periods")
        elif X.shape[0] == 1:
            """x-axis (=time) vs. z-axis"""
            ax.view_init(elev=0, azim=-90, roll=0)
            ax.set_ylabel("")
            ax.set_yticks([])
            ax.set_xlabel(f"Time Periods @ {self.n_flexibilities_list[0]} Flexibilities")

    def _plot_heatmap(self, kind: str, optimizer_name: str, v_min: float, v_max: float | None) -> None:
        n_algorithms = len(self.memory.values())

        if v_max is None:
            v_max = 0
            for values in self.memory.values():
                v_max = max(v_max, np.max(values[kind]))

        figsize = (10, 10 * n_algorithms)
        fig, ax = plt.subplots(n_algorithms, 1, squeeze=False, figsize=figsize)

        title = "UPR" if kind == "upr" else "CPU Time"
        for i, (key, values) in enumerate(self.memory.items()):
            # hm = ax[i, 0].imshow(values[kind].T, cmap="jet")
            sns.heatmap(values[kind].d, ax=ax[i, 0], cmap='jet', xticklabels=self.d_list,
                        yticklabels=self.n_flexibilities_list, vmin=v_min, vmax=v_max, annot=True)
            # ax[i, 0].set_xticks(ticks=np.arange(len(self.d_list)), labels=self.d_list)
            # ax[i, 0].set_yticks(ticks=np.arange(len(self.n_flexibilities_list)), labels=self.n_flexibilities_list)
            ax[i, 0].set_title(f"{optimizer_name}, {key}: {title}")
            ax[i, 0].set_xlabel("Time Periods")
            ax[i, 0].set_ylabel("Flexibilities")
            # ax[i, 0].colorbar(hm)

    def _plot_barchart(self, kind: str, optimizer_name: str):
        n_rows = len(self.memory.keys())
        fig, ax = plt.subplots(n_rows, 1, figsize=(20, 10 * n_rows), squeeze=False)

        y_max = 1.1 * max([np.nanmax(self.memory[algo][kind]) for algo in self.memory.keys()])
        y_min = 1.1 * min([0] + [np.nanmin(self.memory[algo][kind]) for algo in self.memory.keys()])

        title = "UPR" if kind == "upr" else "CPU Time"
        for a, algo in enumerate(self.memory.keys()):
            algorithms_label = [f"[d={d}, n={n}]" for d in self.d_list for n in self.n_flexibilities_list]

            if kind == "upr":
                upr = self.memory[algo]["upr"].flatten().tolist()
                ax[a, 0].bar(algorithms_label, upr, label='UPR (%)')
            elif kind == "cpu_times":
                aggregation_times = self.memory[algo]["aggregate_times"].flatten().tolist()
                optimization_times = self.memory[algo]["optimize_times"].flatten().tolist()
                disaggregation_times = self.memory[algo]["disaggregate_times"].flatten().tolist()

                ax[a, 0].bar(algorithms_label, aggregation_times, label='Aggregation Times')

                ax[a, 0].bar(algorithms_label, optimization_times, bottom=aggregation_times, label='Optimization Times')

                ax[a, 0].bar(algorithms_label, disaggregation_times,
                             bottom=[sum(x) for x in zip(aggregation_times, optimization_times)],
                             label='Disaggregation Times')
            else:
                raise ValueError("Kind must be 'upr' or 'cpu_times'")

            ax[a, 0].set_ylim(y_min, y_max)
            ax[a, 0].tick_params(axis='x', labelrotation=90)
            ax[a, 0].set_xlabel('Scenarios')
            ax[a, 0].set_ylabel("UPR (%)" if kind == "upr" else "CPU Time (s)")
            ax[a, 0].set_title(f"{optimizer_name}, {algo.upper()}: {title}")
            ax[a, 0].grid(True)
            ax[a, 0].legend(loc="upper left")

    def plot_upr_2d(self, optimizer_name: str, logy: bool = False) -> None:
        self._plot_2d(kind="upr", optimizer_name=optimizer_name, logy=logy)

    def plot_cpu_time_2d(self, optimizer_name: str, logy: bool = False) -> None:
        self._plot_2d(kind="cpu_times", optimizer_name=optimizer_name, logy=logy)

    def plot_upr_3d(self, optimizer_name: str) -> None:
        self._plot_3d(kind="upr", optimizer_name=optimizer_name)

    def plot_cpu_time_3d(self, optimizer_name: str) -> None:
        self._plot_3d(kind="cpu_times", optimizer_name=optimizer_name)

    def plot_upr_heatmap(self, optimizer_name: str, v_min: float = 0, v_max: float = None) -> None:
        self._plot_heatmap(kind="upr", optimizer_name=optimizer_name, v_min=v_min, v_max=v_max)

    def plot_cpu_time_heatmap(self, optimizer_name: str, v_min: float = 0, v_max: float = None) -> None:
        self._plot_heatmap(kind="cpu_times", optimizer_name=optimizer_name, v_min=v_min, v_max=v_max)

    def plot_cpu_time_barchart(self, optimizer_name: str, ) -> None:
        self._plot_barchart(kind="cpu_times", optimizer_name=optimizer_name)

    def plot_upr_barchart(self, optimizer_name: str, ) -> None:
        self._plot_barchart(kind="upr", optimizer_name=optimizer_name)

    def show(self, optimizer_name: str = "",
             plot_2d_upr: bool = False, plot_3d_upr: bool = False, plot_heatmap_upr: bool = False,
             plot_2d_time: bool = False, plot_3d_time: bool = False, plot_heatmap_time: bool = False,
             plot_barchart_upr: bool = False, plot_barchart_time: bool = False) -> None:

        if plot_2d_upr:
            self.plot_upr_2d(optimizer_name=optimizer_name)
            plt.tight_layout()
            plt.show()

        if plot_2d_time:
            self.plot_cpu_time_2d(optimizer_name=optimizer_name)
            plt.tight_layout()
            plt.show()

        if plot_3d_upr:
            self.plot_upr_3d(optimizer_name=optimizer_name)
            plt.show()

        if plot_3d_time:
            self.plot_cpu_time_3d(optimizer_name=optimizer_name)
            plt.show()

        if plot_heatmap_upr:
            self.plot_upr_heatmap(optimizer_name=optimizer_name)
            plt.show()

        if plot_heatmap_time:
            self.plot_cpu_time_heatmap(optimizer_name=optimizer_name)
            plt.show()

        if plot_barchart_time:
            self.plot_cpu_time_barchart(optimizer_name=optimizer_name)
            plt.tight_layout()
            plt.show()

        if plot_barchart_upr:
            self.plot_upr_barchart(optimizer_name=optimizer_name)
            plt.tight_layout()
            plt.show()
