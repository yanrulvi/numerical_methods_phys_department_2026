# type: ignore
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Optional, Tuple, Dict, Any
from matplotlib.figure import Figure

def system_rhs(x: float, y: np.ndarray) -> np.ndarray:
    """Правая часть системы ОДУ f(x, y)."""
    y1, y2, y3, y4 = y
    
    dy1 = 2.0 * x * y4 * y1
    dy2 = 10.0 * x * y4 * y1**5
    dy3 = 2.0 * x * y4
    dy4 = -2.0 * x * (y3 - 1.0)
    
    return np.array([dy1, dy2, dy3, dy4])


def exact_y1(x): return np.exp(np.sin(x**2))
def exact_y2(x): return np.exp(5*np.sin(x**2))
def exact_y3(x): return 1 + np.sin(x**2)
def exact_y4(x): return np.cos(x**2)

def plot_all_deviations(
    solver_coarse: "RK4System",
    solver_fine: "RK4System",
    solver_adaptive: Optional["Fehlberg23"] = None,
    save_path: Optional[str] = None,
) -> Figure:
    """
    Построение единого окна с графиками отклонений (4×2).
    
    Строки: компоненты y1, y2, y3, y4.
    Столбцы:
      1) сравнение h = 1e-2 и h = 1e-3 (РК4)
      2) адаптивный метод Фельберга 2(3)
    """
    components = [
        ("y_1", exact_y1, 0),
        ("y_2", exact_y2, 1),
        ("y_3", exact_y3, 2),
        ("y_4", exact_y4, 3),
    ]
    
    n_rows = 4
    n_cols = 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 16), dpi=100)
    fig.suptitle(
        "Абсолютные отклонения численных решений от точных",
        fontsize=14, fontweight="bold", y=0.995,
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
        
        # Столбец 1: сравнение РК4
        ax = axes[row, 0]
        ax.semilogy(
            x1, err_h2, "b-", linewidth=1.0, alpha=0.8,
            label=f"РК4, $h = 10^{{-2}}$",
        )
        ax.semilogy(
            x2, err_h3, "r-", linewidth=1.0, alpha=0.8,
            label=f"РК4, $h = 10^{{-3}}$",
        )
        ax.set_title(f"${label}$: РК4", fontsize=10)
        ax.set_xlabel("$x$", fontsize=9)
        ax.set_ylabel("Отклонение", fontsize=9)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)
        ax.tick_params(labelsize=8)
        ax.axvline(x=4.5, color="gray", linestyle=":", alpha=0.4, linewidth=0.8)
        ax.axvline(x=2.16, color="green", linestyle="--", alpha=0.5, linewidth=0.8)
        
        # Столбец 2: адаптивный метод Фельберга
        ax = axes[row, 1]
        if solver_adaptive is not None:
            x_adapt = solver_adaptive.x
            y_adapt = solver_adaptive.y[idx, :]
            y_exact_adapt = exact_func(x_adapt)
            err_adapt = np.abs(y_adapt - y_exact_adapt)
            
            ax.semilogy(
                x_adapt, err_adapt, "g-", linewidth=1.0, alpha=0.9,
                label=f"Фельберг 2(3), шагов: {solver_adaptive.accepted_steps}"
            )
        ax.set_title(f"${label}$: адаптивный", fontsize=10)
        ax.set_xlabel("$x$", fontsize=9)
        ax.set_ylabel("Отклонение", fontsize=9)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)
        ax.tick_params(labelsize=8)
        ax.axvline(x=4.5, color="gray", linestyle=":", alpha=0.4, linewidth=0.8)
        ax.axvline(x=2.16, color="green", linestyle="--", alpha=0.5, linewidth=0.8)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nГрафик сохранён: {save_path}")
    
    return fig

def plot_adaptive_step_size(
    solver_adaptive: "Fehlberg23",
    save_path: Optional[str] = None,
) -> Figure:
    """
    График изменения величины шага h(x) в логарифмическом масштабе.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 6), dpi=100)
    
    # h_i соответствует точке x_{i+1} (шаг, сделанный из x_i в x_{i+1})
    x_steps = solver_adaptive.x[:-1]
    h_steps = solver_adaptive.h_history
    
    ax.semilogy(x_steps, h_steps, 'b-', linewidth=1.0, alpha=0.9)
    ax.set_title("Фельберг 2(3): величина шага $h(x)$", fontsize=13, fontweight='bold')
    ax.set_xlabel('$x$', fontsize=11)
    ax.set_ylabel('Шаг $h$', fontsize=11)
    ax.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.7)
    ax.tick_params(labelsize=10)
    
    # Отмечаем x = 2.16
    ax.axvline(x=2.16, color='green', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.text(2.2, ax.get_ylim()[1] * 0.8, '$x = 2.16$', fontsize=10, color='green')
    
    # Начальный шаг
    ax.axhline(y=solver_adaptive.h0, color='red', linestyle=':', alpha=0.5, linewidth=1.0)
    ax.text(0.1, solver_adaptive.h0 * 1.2, f'$h_0 = {solver_adaptive.h0}$',
           fontsize=9, color='red')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"График сохранён: {save_path}")
    
    return fig


def plot_solution_comparison(
    x: np.ndarray,
    y: np.ndarray,
    save_path: Optional[str] = None,
) -> Figure:
    """Сравнение численного и точного решения всех компонент."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=100)
    fig.suptitle(
        "Сравнение численного (РК4) и точного решений",
        fontsize=13, fontweight="bold",
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
        
        step = max(1, len(x) // 500)
        
        ax.plot(x, y_num, "b-", linewidth=1.2, label=f"Численное {label}")
        ax.plot(
            x[::step], y_exact[::step], "r.", markersize=2,
            alpha=0.6, label=f"Точное {label}",
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


class RK4System:
    """Явный метод Рунге-Кутты 4-го порядка для системы ОДУ."""
    
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
        self.y0 = y0 if y0 is not None else np.ones(4)
        self.h = h
        
        self.x: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.num_steps: int = 0
    
    def _rk4_step(self, x: float, y: np.ndarray) -> np.ndarray:
        h = self.h
        k1 = self.f(x, y)
        k2 = self.f(x + h / 2, y + k1 * h / 2)
        k3 = self.f(x + h / 2, y + k2 * h / 2)
        k4 = self.f(x + h, y + k3 * h)
        return y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    
    def solve(self, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        self.num_steps = int((self.x_end - self.x0) / self.h)
        self.x = np.linspace(self.x0, self.x_end, self.num_steps + 1)
        self.y = np.zeros((4, self.num_steps + 1))
        self.y[:, 0] = self.y0
        
        if verbose:
            print(f"Интегрирование с шагом h = {self.h:.0e}...", end=" ")
        
        for i in range(self.num_steps):
            self.y[:, i + 1] = self._rk4_step(self.x[i], self.y[:, i])
        
        return self.x, self.y


class Fehlberg23:
    """Метод Фельберга 2(3) для решения систем ОДУ."""    
    def __init__(
        self,
        f: Callable[[float, np.ndarray], np.ndarray],
        x0: float = 0.0,
        x_end: float = 5.0,
        y0: Optional[np.ndarray] = None,
        h0: float = 1e-2,
        rtol: float = 1e-6,
        atol: float = 1e-8,
        h_min: float = 1e-8,
        h_max: float = 1.0,
        max_steps: int = 100000,
    ):
        self.f = f
        self.x0 = x0
        self.x_end = x_end
        self.y0 = y0 if y0 is not None else np.array([1.0, 1.0, 1.0, 1.0])
        self.h0 = h0
        self.rtol = rtol
        self.atol = atol
        self.h_min = h_min
        self.h_max = h_max
        self.max_steps = max_steps
        
        self.x: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.h_history: Optional[np.ndarray] = None
        self.accepted_steps: int = 0
        self.rejected_steps: int = 0
    
    def _fehlberg_step(
        self, x_n: float, y_n: np.ndarray, h: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Шаг метода Фельберга 2(3).
        """
        # Вычисляем k1, k2, k3
        k1 = self.f(x_n, y_n)
        k2 = self.f(x_n + h, y_n + h * k1)
        k3 = self.f(x_n + 0.5 * h, y_n + h * (0.25 * k1 + 0.25 * k2))
        
        # Решение 2-го порядка
        y_next = y_n + h * (0.5 * k1 + 0.5 * k2)
        
        # Решение 3-го порядка
        y_hat = y_n + h * (k1 / 6.0 + k2 / 6.0 + 2.0 * k3 / 3.0)
        
        # Оценка ошибки
        error = np.abs(y_hat - y_next)
        
        return y_next, y_hat, error
    
    def _compute_new_step(
        self, h: float, error: np.ndarray
    ) -> float:
        """Вычисление нового шага на основе оценки ошибки."""
        # Норма ошибки (масштабированная)
        scale = self.atol + self.rtol * np.maximum(
            np.abs(self.y_current), np.abs(self.y_next)
        )
        err_norm = np.sqrt(np.mean((error / scale) ** 2))
        
        # Коэффициент изменения шага
        if err_norm > 0:
            # Для метода порядка p: h_new = h * (1/err)^(1/(p+1))
            # У нас p = 2 (низший порядок), p+1 = 3
            h_new = h * min(2.0, max(0.1, 0.9 * (1.0 / err_norm) ** (1.0 / 3.0)))
        else:
            h_new = 2.0 * h
        
        # Ограничения
        h_new = min(self.h_max, max(self.h_min, h_new))
        
        return h_new, err_norm
    
    def solve(self, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Интегрирование с адаптивным шагом.
        """
        # Храним историю в списках
        x_list = [self.x0]
        y_list = [self.y0.copy()]
        h_list = []
        
        x_current = self.x0
        y_current = self.y0.copy()
        h = self.h0
        
        self.rejected_steps = 0
        step_count = 0
        
        if verbose:
            print(f"\nМетод Фельберга 2(3) с адаптивным шагом")
            print(f"  h0 = {h:.0e}, rtol = {self.rtol:.0e}, atol = {self.atol:.0e}")
            print(f"  Интегрирование...", end=" ")
        
        while x_current < self.x_end and step_count < self.max_steps:
            # Не выходим за границу
            if x_current + h > self.x_end:
                h = self.x_end - x_current
            
            # Сохраняем текущее состояние для случая отклонения шага
            self.y_current = y_current
            
            # Шаг Фельберга
            self.y_next, y_hat, error = self._fehlberg_step(
                x_current, y_current, h
            )
            
            # Оценка ошибки и новый шаг
            h_new, err_norm = self._compute_new_step(h, error)
            
            # Критерий принятия шага
            if err_norm <= 1.0:
                # Шаг принят — используем решение 3-го порядка
                x_current += h
                y_current = y_hat.copy()
                
                x_list.append(x_current)
                y_list.append(y_current.copy())
                h_list.append(h)
                
                self.accepted_steps = step_count + 1
            else:
                # Шаг отклонён
                self.rejected_steps += 1
            
            h = h_new
            step_count += 1
            
            if verbose and step_count % 1000 == 0:
                print(f"[шагов: {step_count}]", end=" ", flush=True)
        
        if verbose:
            print(f"\n  Принято шагов: {self.accepted_steps}")
            print(f"  Отклонено шагов: {self.rejected_steps}")
            print(f"  Всего попыток: {step_count}")
        
        self.x = np.array(x_list)
        self.y = np.array(y_list).T  # (4, N)
        self.h_history = np.array(h_list)
        
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
        self.solver_coarse = RK4System(f=self.f, h=self.h_coarse)
        self.solver_coarse.solve(verbose=verbose)
        
        self.solver_fine = RK4System(f=self.f, h=self.h_fine)
        self.solver_fine.solve(verbose=verbose)
        
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
            print(f"\nТочка контроля: x = 4.5")
            print("-" * 94)
            print(
                f"  {'Компонента':<12} {'Точное значение':<22} "
                f"{'|y^2-y∞|':<14} {'|y^3-y∞|':<14} {'Порядок':<10}"
            )
            print("-" * 94)
        
        for label, exact_func, idx in components:
            y_h2 = self.solver_coarse.y[idx, idx_c]
            y_h3 = self.solver_fine.y[idx, idx_f]
            y_exact = exact_func(np.array([4.5]))[0]
            
            err_h2 = abs(y_h2 - y_exact)
            err_h3 = abs(y_h3 - y_exact)
            
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
                    f"  {label:<12} {y_exact:<22.12f} "
                    f"{err_h2:<14.6e} {err_h3:<14.6e} {order:<10.2f}"
                )
        
        if verbose:
            print("-" * 94)
            print(f"\n  Теоретический порядок: p = 4")
            
            p_y1 = self.order_estimates["y1"]
            if abs(p_y1 - 4.0) < 0.2:
                print(
                    f"\n Порядок точности РК4 ПОДТВЕРЖДЁН (по y1)."
                )
            else:
                print(
                    f"\n  Оценка порядка по y1 отклонилась."
                )
            
            print("=" * 70)
        
        return {
            "errors": errors,
            "order_estimates": self.order_estimates,
        }

if __name__ == "__main__":
    # Часть (а): метод РК4 с постоянным шагом
    print("\n" + "=" * 70)
    print("ЧАСТЬ (а): РК4 с постоянным шагом")
    print("=" * 70)
    
    analyzer = ConvergenceAnalyzer(f=system_rhs, h_coarse=1e-2, h_fine=1e-3)
    results_a = analyzer.run(verbose=True)
    
    # Часть (б): метод Фельберга 2(3) с адаптивным шагом
    print("\n" + "=" * 70)
    print("ЧАСТЬ (б): Фельберг 2(3) с адаптивным шагом")
    print("=" * 70)
    
    fehlberg = Fehlberg23(
        f=system_rhs,
        x0=0.0,
        x_end=5.0,
        h0=1e-2,
        rtol=1e-6,
        atol=1e-8,
    )
    x_adapt, y_adapt = fehlberg.solve(verbose=True)
    
    # Ошибка адаптивного метода в точке x = 4.5
    idx_adapt = np.argmin(np.abs(x_adapt - 4.5))
    print(f"\n  Отклонения адаптивного метода при x = 4.5:")
    print(f"  {'Компонента':<12} {'Численное':<18} {'Точное':<18} {'Ошибка':<14}")
    print(f"  {'-'*60}")
    for i, (name, exact_func) in enumerate([
        ("y1", exact_y1), ("y2", exact_y2), ("y3", exact_y3), ("y4", exact_y4)
    ]):
        y_num = y_adapt[i, idx_adapt]
        y_ex = exact_func(np.array([4.5]))[0]
        err = abs(y_num - y_ex)
        print(f"  {name:<12} {y_num:<18.10f} {y_ex:<18.10f} {err:<14.6e}")
    
    # Графики
    print("\n" + "=" * 70)
    print("ПОСТРОЕНИЕ ГРАФИКОВ")
    print("=" * 70)
    
    # График отклонений (с адаптивным методом)
    fig_deviations = plot_all_deviations(
        solver_coarse=analyzer.solver_coarse,
        solver_fine=analyzer.solver_fine,
        solver_adaptive=fehlberg,
        save_path="tasks/7a/all_deviations.png",
    )
    
    # График изменения шага адаптивного метода
    fig_step = plot_adaptive_step_size(
        solver_adaptive=fehlberg,
        save_path="tasks/7a/adaptive_step_size.png",
    )
    
    # График сравнения решений
    fig_comparison = plot_solution_comparison(
        x=analyzer.solver_fine.x,
        y=analyzer.solver_fine.y,
        save_path="tasks/7a/solution_comparison.png",
    )
    
    # Комментарий к x = 2.16
    print("\n" + "=" * 70)
    print("КОММЕНТАРИЙ К ПОВЕДЕНИЮ ВБЛИЗИ x = 2.16")
    print("=" * 70)
    
    # Находим точку, ближайшую к 2.16
    idx_216 = np.argmin(np.abs(x_adapt - 2.16))
    x_near = x_adapt[idx_216]
    
    if idx_216 > 0:
        h_at_216 = fehlberg.h_history[idx_216 - 1]  # шаг, приведший к 2.16
    else:
        h_at_216 = fehlberg.h0
    
    # Средний шаг до и после
    mask_before = x_adapt < 2.16
    mask_after = x_adapt > 2.16
    
    if np.any(mask_before) and len(fehlberg.h_history) > 0:
        h_before = np.mean(fehlberg.h_history[:max(1, np.sum(mask_before))])
    else:
        h_before = fehlberg.h0
    
    if np.any(mask_after):
        h_after = np.mean(
            fehlberg.h_history[max(0, np.sum(mask_before)):]
        )
    else:
        h_after = h_at_216
    
    print(f"""
    При x ≈ 2.16
    
    Поведение адаптивного метода:
    - Средний шаг до x=2.16: {h_before:.2e}
    - Шаг вблизи x=2.16:     {h_at_216:.2e}
    - Средний шаг после:      {h_after:.2e}
    
    Интегратор {'уменьшает' if h_at_216 < h_before else 'увеличивает'} шаг вблизи x=2.16,
    
    Принято шагов: {fehlberg.accepted_steps}
    Отклонено шагов: {fehlberg.rejected_steps}
    """)
    
    plt.show()
    print("\nГотово! Графики построены, анализ завершён.")