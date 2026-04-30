# type: ignore
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Optional, Tuple, Dict, Any
from matplotlib.figure import Figure


def system_rhs(x: float, y: np.ndarray) -> np.ndarray:
    """
    Правая часть системы ОДУ f(x, y).

    Система:
        y1' = 2*x*y4*y1
        y2' = 10*x*y4*y1^5
        y3' = 2*x*y4
        y4' = -2*x*(y3 - 1)
    """
    y1, y2, y3, y4 = y

    dy1 = 2.0 * x * y4 * y1
    dy2 = 10.0 * x * y4 * y1**5
    dy3 = 2.0 * x * y4
    dy4 = -2.0 * x * (y3 - 1.0)

    return np.array([dy1, dy2, dy3, dy4])


def exact_y1(x: np.ndarray) -> np.ndarray:
    """Точное решение для y1(x) = exp(sin(x^2))."""
    return np.exp(np.sin(x**2))


def exact_y2(x: np.ndarray) -> np.ndarray:
    """Точное решение для y2(x) = exp(5*sin(x^2))."""
    return np.exp(5.0 * np.sin(x**2))


def exact_y3(x: np.ndarray) -> np.ndarray:
    """Точное решение для y3(x) = 1 + sin(x^2)."""
    return 1.0 + np.sin(x**2)


def exact_y4(x: np.ndarray) -> np.ndarray:
    """Точное решение для y4(x) = cos(x^2)."""
    return np.cos(x**2)


def plot_all_deviations(
    solver_coarse: "RK4System",
    solver_fine: "RK4System",
    save_path: Optional[str] = None,
) -> Figure:
    """
    Построение единого окна со всеми графиками отклонений (4×3).

    Строки: компоненты y1, y2, y3, y4.
    Столбцы:
      1) h = 1e-2
      2) h = 1e-3
      3) сравнение h = 1e-2 и h = 1e-3

    Parameters:
        solver_coarse: решатель с крупным шагом (h = 1e-2).
        solver_fine: решатель с мелким шагом (h = 1e-3).
        save_path: путь для сохранения графика.

    Returns:
        объект Figure.
    """
    components = [
        ("y_1", exact_y1, 0),
        ("y_2", exact_y2, 1),
        ("y_3", exact_y3, 2),
        ("y_4", exact_y4, 3),
    ]

    fig, axes = plt.subplots(4, 3, figsize=(18, 16), dpi=100)
    fig.suptitle(
        "Абсолютные отклонения численных решений от точных",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    x1 = solver_coarse.x
    x2 = solver_fine.x
    N1 = solver_coarse.num_steps
    N2 = solver_fine.num_steps

    for row, (label, exact_func, idx) in enumerate(components):
        # Точные решения
        y_exact_h2 = exact_func(x1)
        y_exact_h3 = exact_func(x2)

        # Отклонения
        err_h2 = np.abs(solver_coarse.y[idx, :] - y_exact_h2)
        err_h3 = np.abs(solver_fine.y[idx, :] - y_exact_h3)

        ylims = None

        # Столбец 1: h = 1e-2
        ax = axes[row, 0]
        ax.semilogy(x1, err_h2, "b-", linewidth=1.0, alpha=0.9)
        ax.set_title(f"${label}$, $h = 10^{{-2}}$, $N = {N1}$", fontsize=10)
        ax.set_xlabel("$x$", fontsize=9)
        ax.set_ylabel("Отклонение", fontsize=9)
        ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)
        ax.tick_params(labelsize=8)
        ax.axvline(
            x=4.5, color="gray", linestyle=":", alpha=0.4, linewidth=0.8
        )
        if ylims is None:
            ylims = ax.get_ylim()

        # Столбец 2: h = 1e-3
        ax = axes[row, 1]
        ax.semilogy(x2, err_h3, "r-", linewidth=1.0, alpha=0.9)
        ax.set_title(f"${label}$, $h = 10^{{-3}}$, $N = {N2}$", fontsize=10)
        ax.set_xlabel("$x$", fontsize=9)
        ax.set_ylabel("Отклонение", fontsize=9)
        ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)
        ax.tick_params(labelsize=8)
        ax.axvline(
            x=4.5, color="gray", linestyle=":", alpha=0.4, linewidth=0.8
        )

        # Столбец 3: сравнение
        ax = axes[row, 2]
        ax.semilogy(
            x1,
            err_h2,
            "b-",
            linewidth=1.0,
            alpha=0.8,
            label="$h = 10^{{-2}}$",
        )
        ax.semilogy(
            x2,
            err_h3,
            "r-",
            linewidth=1.0,
            alpha=0.8,
            label="$h = 10^{{-3}}$",
        )
        ax.set_title(f"${label}$: сравнение", fontsize=10)
        ax.set_xlabel("$x$", fontsize=9)
        ax.set_ylabel("Отклонение", fontsize=9)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)
        ax.tick_params(labelsize=8)
        ax.axvline(
            x=4.5, color="gray", linestyle=":", alpha=0.4, linewidth=0.8
        )

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nГрафик сохранён: {save_path}")

    return fig


def plot_solution_comparison(
    x: np.ndarray,
    y: np.ndarray,
    save_path: Optional[str] = None,
) -> Figure:
    """
    Сравнение численного и точного решения всех компонент.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=100)
    fig.suptitle(
        "Сравнение численного (РК4) и точного решений",
        fontsize=13,
        fontweight="bold",
    )

    components = [
        ("$y_1$", exact_y1, 0),
        ("$y_2$", exact_y2, 1),
        ("$y_3$", exact_y3, 2),
        ("$y_4$", exact_y4, 3),
    ]

    for ax, (label, exact_func, idx) in zip(axes.flat, components):
        y_num = y[idx, :]
        y_exact = exact_func(x)

        # Разреженная выборка для маркеров точного решения
        step = max(1, len(x) // 500)

        ax.plot(x, y_num, "b-", linewidth=1.2, label=f"Численное {label}")
        ax.plot(
            x[::step],
            y_exact[::step],
            "r.",
            markersize=2,
            alpha=0.6,
            label=f"Точное {label}",
        )
        ax.set_xlabel("$x$", fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.set_title(f"Компонента {label}", fontsize=11)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"График сохранён: {save_path}")

    return fig


def print_task_info() -> None:
    """Вывод информации о задаче."""
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print(
        "█"
        + "   ЗАДАЧА: МЕТОД РУНГЕ-КУТТЫ 4-го ПОРЯДКА ДЛЯ СИСТЕМЫ ОДУ   ".center(
            68
        )
        + "█"
    )
    print("█" + " " * 68 + "█")
    print("█" * 70)
    print()
    print("Система уравнений:")
    print("  y1' = 2·x·y4·y1")
    print("  y2' = 10·x·y4·y1^5")
    print("  y3' = 2·x·y4")
    print("  y4' = -2·x·(y3 - 1)")
    print()
    print("Начальные условия: y1(0) = y2(0) = y3(0) = y4(0) = 1")
    print("Отрезок интегрирования: [0, 5]")
    print()
    print("Точные решения:")
    print("  y1∞ = e^{sin(x²)}")
    print("  y2∞ = e^{5·sin(x²)}")
    print("  y3∞ = 1 + sin(x²)")
    print("  y4∞ = cos(x²)")
    print()


class RK4System:
    """
    Явный метод Рунге-Кутты 4-го порядка для системы ОДУ.
    """

    def __init__(
        self,
        f: Callable[[float, np.ndarray], np.ndarray],
        x0: float = 0.0,
        x_end: float = 5.0,
        y0: Optional[np.ndarray] = None,
        h: float = 1e-2,
    ):
        self.f = f
        self.x0 = x0
        self.x_end = x_end
        self.y0 = y0 if y0 is not None else np.array([1.0, 1.0, 1.0, 1.0])
        self.h = h

        self.x = None
        self.y = None
        self.num_steps = 0

    def _rk4_step(self, x_n: float, y_n: np.ndarray) -> np.ndarray:
        h = self.h
        k1 = self.f(x_n, y_n)
        k2 = self.f(x_n + 0.5 * h, y_n + 0.5 * h * k1)
        k3 = self.f(x_n + 0.5 * h, y_n + 0.5 * h * k2)
        k4 = self.f(x_n + h, y_n + h * k3)
        return y_n + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def solve(self, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        self.num_steps = int((self.x_end - self.x0) / self.h)
        self.x = np.linspace(self.x0, self.x_end, self.num_steps + 1)
        self.y = np.zeros((4, self.num_steps + 1))
        self.y[:, 0] = self.y0

        if verbose:
            print(f"Интегрирование с шагом h = {self.h:.0e}...", end=" ")

        for i in range(self.num_steps):
            self.y[:, i + 1] = self._rk4_step(self.x[i], self.y[:, i])

            if (
                verbose
                and self.num_steps > 100
                and (i + 1) % (self.num_steps // 10) == 0
            ):
                progress = (i + 1) / self.num_steps * 100
                print(f"{progress:.0f}%", end=" ", flush=True)

        if verbose:
            print("100%")

        return self.x, self.y


class ConvergenceAnalyzer:
    """
    Анализатор сходимости метода РК4.
    """

    def __init__(
        self,
        f: Callable[[float, np.ndarray], np.ndarray],
        h_coarse: float = 1e-2,
        h_fine: float = 1e-3,
    ):
        self.f = f
        self.h_coarse = h_coarse
        self.h_fine = h_fine

        self.solver_coarse: Optional[RK4System] = None
        self.solver_fine: Optional[RK4System] = None
        self.order_estimates: Dict[str, float] = {}

    def run(self, verbose: bool = True) -> Dict[str, Any]:
        # Решение с крупным шагом
        self.solver_coarse = RK4System(f=self.f, h=self.h_coarse)
        self.solver_coarse.solve(verbose=verbose)

        # Решение с мелким шагом
        self.solver_fine = RK4System(f=self.f, h=self.h_fine)
        self.solver_fine.solve(verbose=verbose)

        # Вычисление ошибок в точке x = 4.5
        idx_c = np.argmin(np.abs(self.solver_coarse.x - 4.5))
        idx_f = np.argmin(np.abs(self.solver_fine.x - 4.5))

        components = [
            ("y1", exact_y1, 0),
            ("y2", exact_y2, 1),
            ("y3", exact_y3, 2),
            ("y4", exact_y4, 3),
        ]

        errors = {}

        if verbose:
            print("\n" + "=" * 70)
            print("АНАЛИЗ СХОДИМОСТИ (РК4, 4-й порядок)")
            print("=" * 70)
            print("\nТочка контроля: x = 4.5")
            print("-" * 94)
            print(
                f"  {'Компонента':<12} {'Точное значение':<22} {'|y^2-y∞|':<14} {'|y^3-y∞|':<14} {'Порядок':<10}"
            )
            print("-" * 94)

        for label, exact_func, idx in components:
            y_h2 = self.solver_coarse.y[idx, idx_c]
            y_h3 = self.solver_fine.y[idx, idx_f]
            y_exact = exact_func(np.array([4.5]))[0]

            err_h2 = abs(y_h2 - y_exact)
            err_h3 = abs(y_h3 - y_exact)

            # Оценка порядка точности
            if err_h3 > 1e-16 and err_h2 > 1e-16:
                order = np.log(err_h2 / err_h3) / np.log(
                    self.h_coarse / self.h_fine
                )
            else:
                order = np.inf
            self.order_estimates[label] = order

            errors[label] = {
                "exact": y_exact,
                "num_h2": y_h2,
                "num_h3": y_h3,
                "err_h2": err_h2,
                "err_h3": err_h3,
            }

            if verbose:
                print(
                    f"  {label:<12} {y_exact:<22.12f} {err_h2:<14.6e} {err_h3:<14.6e} {order:<10.2f}"
                )

        if verbose:
            print("-" * 94)
            print("\n  Теоретический порядок: p = 4")

            p_y1 = self.order_estimates["y1"]
            if abs(p_y1 - 4.0) < 0.2:
                print(
                    "\n  ✅ Утверждение о четвёртом порядке точности метода РК4 ПОДТВЕРЖДЕНО (по y1)."
                )
            else:
                print(
                    "\n  ❗ Оценка порядка по y1 отклонилась от теоретического значения."
                )

            print("=" * 70)

        return {
            "errors": errors,
            "order_estimates": self.order_estimates,
        }


if __name__ == "__main__":
    # Вывод информации о задаче
    print_task_info()

    # Запуск
    analyzer = ConvergenceAnalyzer(f=system_rhs, h_coarse=1e-2, h_fine=1e-3)
    results = analyzer.run(verbose=True)

    # Вывод графиков
    fig_deviations = plot_all_deviations(
        solver_coarse=analyzer.solver_coarse,
        solver_fine=analyzer.solver_fine,
        save_path="tasks/7a/all_deviations.png",
    )

    # График сравнения численного и точного решений
    fig_comparison = plot_solution_comparison(
        x=analyzer.solver_fine.x,
        y=analyzer.solver_fine.y,
        save_path="tasks/7a/solution_comparison.png",
    )

    plt.show()

    print("\nГотово! Графики построены, анализ завершён.")
