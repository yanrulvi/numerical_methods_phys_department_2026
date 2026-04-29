# type: ignore
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Optional, Tuple
from matplotlib.figure import Figure


def original_integrand(x: float) -> float:
    return (x**1.5) * np.cos(x) / np.sqrt(x**2 - 1.0)


def transformed_integrand(t: float) -> float:
    return ((t**2 + 1.0) ** 0.25) * np.cos(np.sqrt(t**2 + 1.0))


def get_transformed_limits() -> Tuple[float, float]:
    return 0.0, np.sqrt(10.0**2 - 1.0)


class Task6a:
    def __init__(
        self,
        f: Callable,
        a: float,
        b: float,
        N_start: int = 12,
        tol: float = 1e-14,
        max_iter: int = 20,
    ):
        self.f = f
        self.a = a
        self.b = b
        self.N_start = N_start
        self.tol = tol
        self.max_iter = max_iter

        self.integral = None
        self.table: List[List[float]] = []
        self.N_list: List[int] = []
        self.errors: List[float] = []
        self.extrapolation_order = None
        self.converged = False
        self.iterations_done = 0

    def _trapezoid(self, N: int) -> float:
        if N <= 0:
            return 0.0
        h = (self.b - self.a) / N
        x = np.linspace(self.a, self.b, N + 1)
        y = self.f(x)
        return h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])

    def compute(self, verbose: bool = True) -> float:
        if verbose:
            print("=" * 70)
            print("МЕТОД РОМБЕРГА: ВЫЧИСЛЕНИЕ ИНТЕГРАЛА")
            print("=" * 70)
            print(f"Пределы интегрирования: [{self.a:.6f}, {self.b:.6f}]")
            print(f"Начальное число разбиений: N₀ = {self.N_start}")
            print(f"Требуемая точность: ε = {self.tol:.0e}")
            print("-" * 70)
            print(
                f"\n{'n':<4} {'N':<10} {'R(n,0)':<24} {'R(n,1)':<24} {'R(n,2)':<24} {'R(n,3)':<24}"
            )
            print("-" * 100)

        self.table = []
        self.N_list = []
        self.errors = []

        N = self.N_start
        T = self._trapezoid(N)
        self.table.append([T])
        self.N_list.append(N)

        if verbose:
            print(f"{0:<4} {N:<10} {T:<24.15e}")

        self.converged = False
        final_n = 0

        for n in range(1, self.max_iter + 1):
            N = self.N_start * (2**n)
            self.N_list.append(N)
            T = self._trapezoid(N)
            row = [T]

            for m in range(1, n + 1):
                R_curr = row[m - 1]
                R_prev = self.table[n - 1][m - 1]
                R_extr = R_curr + (R_curr - R_prev) / (4**m - 1)
                row.append(R_extr)

            self.table.append(row)

            if n >= 1:
                err = abs(self.table[n][n] - self.table[n - 1][n - 1])
                self.errors.append(err)

            if verbose:
                line = f"{n:<4} {N:<10}"
                for m in range(min(n + 1, 4)):
                    line += f" {row[m]:<24.15e}"
                print(line)

            if n >= 1:
                diff = abs(self.table[n][n] - self.table[n - 1][n - 1])
                if diff < self.tol:
                    final_n = n
                    self.converged = True
                    if verbose:
                        print("-" * 100)
                        print(f"✅ Сходимость достигнута за {n} итераций")
                        print(
                            f"   |R({n},{n}) - R({n - 1},{n - 1})| = {diff:.2e} < ε = {self.tol:.0e}"
                        )
                    break
        else:
            final_n = self.max_iter
            if verbose:
                print("!!!Достигнут максимум итераций!!!")

        self.integral = self.table[final_n][final_n]
        self.extrapolation_order = final_n
        self.iterations_done = final_n + 1

        if verbose:
            print(f"\nРЕЗУЛЬТАТ: I = {self.integral:.15f}")
            print("=" * 70)

        return self.integral

    def get_integral(self) -> float:
        if self.integral is None:
            raise RuntimeError("Сначала вызовите compute().")
        return self.integral

    def get_table(self) -> List[List[float]]:
        if not self.table:
            raise RuntimeError("Сначала вызовите compute().")
        return self.table

    def get_convergence_data(self) -> dict:
        if not self.table:
            raise RuntimeError("Сначала вызовите compute().")

        n_arr = np.arange(len(self.table))
        N_arr = np.array(self.N_list)
        h_arr = (self.b - self.a) / N_arr
        trapezoid_vals = np.array([row[0] for row in self.table])
        diagonal_vals = np.array([row[i] for i, row in enumerate(self.table)])
        errors_arr = np.array([np.nan] + self.errors)

        return {
            "n": n_arr,
            "N": N_arr,
            "h": h_arr,
            "trapezoid": trapezoid_vals,
            "diagonal": diagonal_vals,
            "errors": errors_arr,
        }


def plot_all(
    integrator: "Task6a",
    x_start: float = 1.0,
    x_end: float = 10.0,
    n_points: int = 2000,
    save_path: Optional[str] = None,
) -> Figure:

    t_start, t_end = get_transformed_limits()
    data = integrator.get_convergence_data()

    n_vals = data["n"]
    trap_vals = data["trapezoid"]
    diag_vals = data["diagonal"]
    errors = data["errors"]
    I_exact = integrator.integral

    fig = plt.figure(figsize=(16, 18))

    ax1 = fig.add_subplot(3, 2, 1)
    x = np.linspace(x_start + 0.001, x_end, n_points)
    y = original_integrand(x)
    ax1.plot(x, y, "b-", linewidth=1.5)
    ax1.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax1.set_xlabel("x", fontsize=10)
    ax1.set_ylabel("f(x)", fontsize=10)
    ax1.set_title(
        "Исходная функция (с сингулярностью при x = 1)",
        fontsize=11,
        fontweight="bold",
    )
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(3, 2, 2)
    t = np.linspace(t_start, t_end, n_points)
    g = transformed_integrand(t)
    ax2.plot(t, g, "g-", linewidth=1.5)
    ax2.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax2.plot(
        t_start,
        transformed_integrand(t_start),
        "go",
        markersize=8,
        label=f"g(0) = cos(1) ≈ {np.cos(1.0):.6f}",
    )
    ax2.set_xlabel("t", fontsize=10)
    ax2.set_ylabel("g(t)", fontsize=10)
    ax2.set_title(
        "Преобразованная функция (гладкая)", fontsize=11, fontweight="bold"
    )
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(3, 2, 3)
    ax3.plot(
        n_vals,
        trap_vals,
        "b.-",
        linewidth=1.5,
        markersize=8,
        label="Правило трапеций R(n,0)",
    )
    ax3.plot(
        n_vals,
        diag_vals,
        "r.-",
        linewidth=1.5,
        markersize=8,
        label="Экстраполяция R(n,n)",
    )
    ax3.axhline(
        y=I_exact,
        color="green",
        linestyle="--",
        linewidth=1.2,
        label=f"I = {I_exact:.8f}",
    )
    ax3.set_xlabel("Итерация n", fontsize=10)
    ax3.set_ylabel("Значение интеграла", fontsize=10)
    ax3.set_title(
        "Сходимость к точному значению", fontsize=11, fontweight="bold"
    )
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=7)

    ax4 = fig.add_subplot(3, 2, 4)
    mask = ~np.isnan(errors)
    ax4.semilogy(n_vals[mask], errors[mask], "m.-", linewidth=2, markersize=10)
    ax4.axhline(
        y=integrator.tol,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=f"ε = {integrator.tol:.0e}",
    )
    for i, (n_val, err_val) in enumerate(zip(n_vals[mask], errors[mask])):
        if i < 6:
            ax4.annotate(
                f"{err_val:.1e}",
                (n_val, err_val),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=6,
                color="darkred",
            )
    ax4.set_xlabel("Итерация n", fontsize=10)
    ax4.set_ylabel(r"$|R(n,n) - R(n-1,n-1)|$", fontsize=10)
    ax4.set_title(
        "Динамика ошибки экстраполяции", fontsize=11, fontweight="bold"
    )
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=8)

    ax5 = fig.add_subplot(3, 2, 5)
    table = integrator.get_table()
    n_rows = len(table)
    colors = ["blue", "red", "green", "orange", "purple", "brown"]
    markers = ["o", "s", "D", "^", "v", "<"]
    max_m = min(6, max(len(row) for row in table))
    for m in range(max_m):
        x_vals, y_vals = [], []
        for n in range(m, n_rows):
            if m < len(table[n]):
                rel_err = (
                    abs((table[n][m] - I_exact) / I_exact)
                    if I_exact != 0
                    else abs(table[n][m] - I_exact)
                )
                digits = -np.log10(max(rel_err, 1e-16))
                x_vals.append(n)
                y_vals.append(digits)
        if x_vals:
            ax5.plot(
                x_vals,
                y_vals,
                color=colors[m],
                marker=markers[m],
                linewidth=1.5,
                markersize=6,
                label=f"R(n,{m})",
            )
    ax5.axhline(
        y=15, color="gray", linestyle="--", linewidth=1, label="~15 знаков"
    )
    ax5.set_xlabel("Итерация n", fontsize=10)
    ax5.set_ylabel("Число верных знаков", fontsize=10)
    ax5.set_title(
        "Рост точности с повышением порядка экстраполяции",
        fontsize=11,
        fontweight="bold",
    )
    ax5.grid(True, alpha=0.3)
    ax5.legend(fontsize=7, loc="lower right")

    ax6 = fig.add_subplot(3, 2, 6)
    max_cols = min(8, max(len(row) for row in table))
    max_rows = min(12, len(table))
    heatmap_data = np.full((max_rows, max_cols), np.nan)
    for n in range(max_rows):
        for m in range(min(n + 1, max_cols)):
            heatmap_data[n, m] = abs(table[n][m] - I_exact)
    log_data = np.log10(heatmap_data + 1e-50)
    im = ax6.imshow(
        log_data,
        aspect="auto",
        cmap="RdYlGn_r",
        extent=[-0.5, max_cols - 0.5, max_rows - 0.5, -0.5],
    )
    for n in range(max_rows):
        for m in range(min(n + 1, max_cols)):
            val = heatmap_data[n, m]
            if not np.isnan(val):
                text = f"{val:.1e}" if val > 0 else "0"
                color = "white" if log_data[n, m] < -10 else "black"
                ax6.text(
                    m,
                    n,
                    text,
                    ha="center",
                    va="center",
                    fontsize=5,
                    color=color,
                )
    ax6.set_xticks(range(max_cols))
    ax6.set_xticklabels([f"m={m}" for m in range(max_cols)])
    ax6.set_yticks(range(max_rows))
    ax6.set_yticklabels([f"n={n}" for n in range(max_rows)])
    ax6.set_xlabel("Порядок экстраполяции m", fontsize=10)
    ax6.set_ylabel("Итерация n", fontsize=10)
    ax6.set_title(
        "Тепловая карта: |R(n,m) − I|", fontsize=11, fontweight="bold"
    )
    cbar = plt.colorbar(im, ax=ax6, shrink=0.8)
    cbar.set_label(r"$\log_{10}(|\text{ошибка}|)$", fontsize=9)

    plt.suptitle(
        "ЗАДАЧА 6: Визуализация результатов",
        fontsize=16,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"График сохранён: {save_path}")

    return fig


def print_analysis_report(integrator: "Task6a") -> None:
    data = integrator.get_convergence_data()
    print("\n" + "=" * 70)
    print("ИТОГОВЫЙ ОТЧЁТ")
    print("=" * 70)
    print("\nЗначение интеграла:")
    print(f"  I = {integrator.integral:.15f}")
    print("\nХод вычислений:")
    print(f"  {'n':<6} {'N':<12} {'R(n,n)':<24} {'Ошибка':<14}")
    print(f"  {'-' * 56}")
    for i in range(len(data["n"])):
        n = data["n"][i]
        N = data["N"][i]
        diag = data["diagonal"][i]
        err = data["errors"][i] if i > 0 else np.nan
        err_str = f"{err:.2e}" if not np.isnan(err) else "—"
        print(f"  {n:<6} {N:<12} {diag:<24.15e} {err_str:<14}")
    print("\nХарактеристики сходимости:")
    print(f"  Итераций до сходимости: {integrator.iterations_done}")
    print(f"  Достигнутая точность: {data['errors'][-1]:.2e}")
    print(f"  Порядок экстраполяции: {integrator.extrapolation_order}")
    trap_last = data["trapezoid"][-1]
    trap_err = abs(trap_last - integrator.integral)
    romb_err = data["errors"][-1] if not np.isnan(data["errors"][-1]) else 0.0
    print("\nСравнение с правилом трапеций:")
    print(f"  Трапеции (N={data['N'][-1]}): ошибка = {trap_err:.2e}")
    print(f"  Ромберг:           ошибка = {romb_err:.2e}")


def main(show_plot: bool = True, save_plot: bool = True) -> None:
    print("=" * 70)
    print("ЗАДАЧА 6: ВЫЧИСЛЕНИЕ ИНТЕГРАЛА МЕТОДОМ РОМБЕРГА")
    print("=" * 70)
    print()
    print("Интеграл: ∫_{1}^{10} x^(3/2)·cos(x) / √(x²-1) dx")
    print("Замена: x = √(t² + 1)  →  g(t) = (t²+1)^(1/4)·cos(√(t²+1))")
    print("Новые пределы: t ∈ [0, √99]\n")

    t_start, t_end = get_transformed_limits()
    integrator = Task6a(
        f=transformed_integrand, a=t_start, b=t_end, N_start=12, tol=1e-14
    )
    integrator.compute(verbose=True)

    print("\nПостроение графика...")
    plot_all(
        integrator,
        save_path="tasks/6a/task6_result.png" if save_plot else None,
    )
    print_analysis_report(integrator)

    print(f"\n{'=' * 70}")
    print(f"ОТВЕТ: I = {integrator.integral:.15f}")
    print(f"{'=' * 70}")

    if show_plot:
        plt.show()


if __name__ == "__main__":
    main(show_plot=True, save_plot=True)
