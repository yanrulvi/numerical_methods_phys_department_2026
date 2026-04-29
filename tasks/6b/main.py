# type: ignore
# flake8: noqa: E501
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "6a"))

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Optional, Tuple
from matplotlib.figure import Figure
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from main import Task6a


def integrand_6b(x: float, y: float) -> float:
    return (
        np.exp(x + y)
        * np.cos(x)
        * np.sin(y)
        / np.sqrt(x**2 + 2.0 * y**2 + 1.0)
    )


def get_outer_limits() -> Tuple[float, float]:
    return 0.0, 2.0


def get_inner_limits() -> Tuple[float, float]:
    return 0.0, 3.0


def _compute_inner(args):
    x, f, y_start, y_end, N_start, tol_inner = args

    def f_fixed_x(y: float) -> float:
        return f(x, y)

    inner = Task6a(
        f=f_fixed_x,
        a=y_start,
        b=y_end,
        N_start=N_start,
        tol=tol_inner,
    )
    inner.compute(verbose=False)
    return x, inner.get_integral()


class Task6b:

    def __init__(
        self,
        f: Callable[[float, float], float],
        x_start: float,
        x_end: float,
        y_start: float,
        y_end: float,
        N_start: int = 12,
        tol_outer: float = 1e-14,
        tol_inner: float = 1e-14,
        max_iter: int = 20,
        n_workers: int = None,
    ):
        self.f = f
        self.x_start = x_start
        self.x_end = x_end
        self.y_start = y_start
        self.y_end = y_end
        self.N_start = N_start
        self.tol_outer = tol_outer
        self.tol_inner = tol_inner
        self.max_iter = max_iter
        self.n_workers = n_workers or multiprocessing.cpu_count()

        self.integral = None
        self.table: List[List[float]] = []
        self.extrapolation_order = None
        self.converged = False
        self.iterations_done = 0

        self._cache: dict = {}

    def _g_batch(self, x_points: np.ndarray) -> np.ndarray:
        new_x = []
        for x in x_points:
            key = round(x, 14)
            if key not in self._cache:
                new_x.append(x)

        if new_x:
            args_list = [
                (
                    x,
                    self.f,
                    self.y_start,
                    self.y_end,
                    self.N_start,
                    self.tol_inner,
                )
                for x in new_x
            ]

            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                futures = [
                    executor.submit(_compute_inner, args) for args in args_list
                ]
                for future in as_completed(futures):
                    x, val = future.result()
                    self._cache[round(x, 14)] = val

        return np.array([self._cache[round(x, 14)] for x in x_points])

    def _trapezoid_outer(self, N: int) -> float:
        if N <= 0:
            return 0.0
        h = (self.x_end - self.x_start) / N
        x_points = np.array([self.x_start + i * h for i in range(N + 1)])
        g_vals = self._g_batch(x_points)
        return h * (0.5 * g_vals[0] + np.sum(g_vals[1:-1]) + 0.5 * g_vals[-1])

    def compute(self, verbose: bool = True) -> float:
        if verbose:
            print("=" * 70)
            print("МЕТОД РОМБЕРГА: ДВОЙНОЙ ИНТЕГРАЛ (ПАРАЛЛЕЛЬНО)")
            print("=" * 70)
            print(f"Внешний интеграл по x: [{self.x_start}, {self.x_end}]")
            print(
                f"Внутренний интеграл по y: [{self.y_start:.6f}, {self.y_end:.6f}]"
            )
            print(f"Начальное число разбиений: N₀ = {self.N_start}")
            print(f"Точность внешнего интеграла: ε_out = {self.tol_outer:.0e}")
            print(
                f"Точность внутреннего интеграла: ε_in = {self.tol_inner:.0e}"
            )
            print(f"Число процессов: {self.n_workers}")
            print("-" * 70)
            print("-" * 70)
            print(
                f"\n{'n':<4} {'N':<10} {'R(n,0)':<24} {'R(n,1)':<24} {'R(n,2)':<24} {'R(n,3)':<24} {'R(n,4)':<24} {'R(n,5)':<24}"
            )
            print("-" * 140)

        self.table = []
        self._cache.clear()

        N = self.N_start
        T = self._trapezoid_outer(N)
        self.table.append([T])

        if verbose:
            print(f"{0:<4} {N:<10} {T:<24.15e}")

        self.converged = False
        final_n = 0

        for n in range(1, self.max_iter + 1):
            N = self.N_start * (2**n)
            T = self._trapezoid_outer(N)
            row = [T]

            for m in range(1, n + 1):
                R_curr = row[m - 1]
                R_prev = self.table[n - 1][m - 1]
                R_extr = R_curr + (R_curr - R_prev) / (4**m - 1)
                row.append(R_extr)

            self.table.append(row)

            if verbose:
                line = f"{n:<4} {N:<10}"
                for m in range(n + 1):
                    line += f" {row[m]:<24.15e}"
                print(line)

            if n >= 1:
                diff = abs(self.table[n][n] - self.table[n - 1][n - 1])
                if diff < self.tol_outer:
                    final_n = n
                    self.converged = True
                    if verbose:
                        print("-" * 100)
                        print(f"✅ Сходимость достигнута за {n} итераций")
                        print(
                            f"   |R({n},{n}) - R({n - 1},{n - 1})| = {diff:.2e} < ε_out = {self.tol_outer:.0e}"
                        )
                    break
        else:
            final_n = self.max_iter
            if verbose:
                print("!!! Достигнут максимум итераций !!!")

        self.integral = self.table[final_n][final_n]
        self.extrapolation_order = final_n
        self.iterations_done = final_n + 1

        if verbose:
            print(f"\nРЕЗУЛЬТАТ: I = {self.integral:.15f}")
            print(f"Всего уникальных вызовов _g: {len(self._cache)}")
            print("=" * 70)

        return self.integral

    def get_integral(self) -> float:
        if self.integral is None:
            raise RuntimeError("Сначала вызовите compute().")
        return self.integral


def plot_6b(
    task: "Task6b",
    f: Callable[[float, float], float],
    x_start: float,
    x_end: float,
    y_start: float,
    y_end: float,
    n_points: int = 80,
    save_path: Optional[str] = None,
) -> Figure:
    x_vals = np.linspace(x_start, x_end, n_points)
    y_vals = np.linspace(y_start, y_end, n_points)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = f(X, Y)

    fig = plt.figure(figsize=(18, 12))

    ax1 = fig.add_subplot(2, 3, (1, 2), projection="3d")
    surf = ax1.plot_surface(
        X, Y, Z, cmap="viridis", edgecolor="none", alpha=0.9
    )
    ax1.set_xlabel("x", fontsize=11)
    ax1.set_ylabel("y", fontsize=11)
    ax1.set_zlabel("f(x,y)", fontsize=11)
    ax1.set_title(
        "Подынтегральная функция f(x,y)", fontsize=13, fontweight="bold"
    )
    cbar1 = fig.colorbar(surf, ax=ax1, shrink=0.6, pad=0.1)
    cbar1.set_label("f(x,y)", fontsize=10)

    ax2 = fig.add_subplot(2, 3, 3)
    im2 = ax2.contourf(X, Y, Z, levels=30, cmap="viridis")
    ax2.set_xlabel("x", fontsize=10)
    ax2.set_ylabel("y", fontsize=10)
    ax2.set_title("Линии уровня f(x,y)", fontsize=12, fontweight="bold")
    cbar2 = fig.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label("f(x,y)", fontsize=9)

    ax3 = fig.add_subplot(2, 3, 4)
    errors = [
        abs(task.table[i][i] - task.table[i - 1][i - 1])
        for i in range(1, len(task.table))
    ]
    iters = list(range(1, len(errors) + 1))
    ax3.semilogy(iters, errors, "m.-", linewidth=2, markersize=10)
    ax3.axhline(
        y=task.tol_outer,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=f"ε = {task.tol_outer:.0e}",
    )
    for i, (it, err) in enumerate(zip(iters, errors)):
        if i < 6:
            ax3.annotate(
                f"{err:.1e}",
                (it, err),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=7,
                color="darkred",
            )
    ax3.set_xlabel("Итерация n", fontsize=10)
    ax3.set_ylabel(r"$|R(n,n) - R(n-1,n-1)|$", fontsize=10)
    ax3.set_title(
        "Сходимость внешнего интеграла", fontsize=12, fontweight="bold"
    )
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8)

    ax4 = fig.add_subplot(2, 3, 5)
    n_table = min(len(task.table), 10)
    m_table = min(6, max(len(row) for row in task.table))
    data = np.full((n_table, m_table), np.nan)
    for i in range(n_table):
        for j in range(min(i + 1, m_table)):
            data[i, j] = abs(task.table[i][j] - task.integral)
    log_data = np.log10(data + 1e-50)
    im4 = ax4.imshow(
        log_data,
        aspect="auto",
        cmap="RdYlGn_r",
        extent=[-0.5, m_table - 0.5, n_table - 0.5, -0.5],
    )
    for i in range(n_table):
        for j in range(min(i + 1, m_table)):
            val = data[i, j]
            if not np.isnan(val):
                text = f"{val:.1e}" if val > 0 else "0"
                color = "white" if log_data[i, j] < -10 else "black"
                ax4.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=5,
                    color=color,
                )
    ax4.set_xticks(range(m_table))
    ax4.set_xticklabels([f"m={m}" for m in range(m_table)])
    ax4.set_yticks(range(n_table))
    ax4.set_yticklabels([f"n={n}" for n in range(n_table)])
    ax4.set_xlabel("m", fontsize=10)
    ax4.set_ylabel("n", fontsize=10)
    ax4.set_title(
        "Таблица Ромберга |R(n,m) - I|", fontsize=12, fontweight="bold"
    )
    cbar4 = fig.colorbar(im4, ax=ax4, shrink=0.8)
    cbar4.set_label(r"$\log_{10}(|\text{ошибка}|)$", fontsize=9)

    ax5 = fig.add_subplot(2, 3, 6)
    ax5.axis("off")
    text = (
        f"РЕЗУЛЬТАТЫ ИНТЕГРИРОВАНИЯ\n\n"
        f"I = {task.integral:.15f}\n\n"
        f"Итераций: {task.iterations_done}\n"
        f"Сходимость: {'да' if task.converged else 'нет'}\n"
        f"Уник. вызовов g(x): {len(task._cache)}\n\n"
        f"Параметры:\n"
        f"N₀ = {task.N_start}\n"
        f"ε_out = {task.tol_outer:.0e}\n"
        f"ε_in = {task.tol_inner:.0e}"
    )
    ax5.text(
        0.1,
        0.5,
        text,
        transform=ax5.transAxes,
        fontsize=12,
        verticalalignment="center",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    plt.suptitle(
        "ЗАДАЧА 6б: Двойной интеграл методом Ромберга",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"График сохранён: {save_path}")

    return fig


def main(show_plot: bool = True, save_plot: bool = True) -> None:
    print("=" * 70)
    print("ЗАДАЧА 6б: ДВОЙНОЙ ИНТЕГРАЛ (ПАРАЛЛЕЛЬНО)")
    print("=" * 70)
    print()
    print(
        "I = ∫_{0}^{2} [ ∫_{0}^{3} exp(x+y)·cos(x)·sin(y) / √(x²+2y²+1) dy ] dx"
    )
    print()

    x_start, x_end = get_outer_limits()
    y_start, y_end = get_inner_limits()

    task = Task6b(
        f=integrand_6b,
        x_start=x_start,
        x_end=x_end,
        y_start=y_start,
        y_end=y_end,
        N_start=12,
        tol_outer=1e-14,
        tol_inner=1e-16,
        n_workers=None,
    )

    result = task.compute(verbose=True)

    print("\nПостроение графика...")
    plot_6b(
        task=task,
        f=integrand_6b,
        x_start=x_start,
        x_end=x_end,
        y_start=y_start,
        y_end=y_end,
        save_path="tasks/6b/task6b.png" if save_plot else None,
    )

    print(f"\n{'=' * 70}")
    print(f"ОТВЕТ: I = {result:.15f}")
    print(f"{'=' * 70}")

    if show_plot:
        plt.show()


if __name__ == "__main__":
    main()
