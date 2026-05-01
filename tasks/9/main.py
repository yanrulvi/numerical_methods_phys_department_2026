# type: ignore
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Optional, Tuple
from matplotlib.figure import Figure


def rhs_stiff(x: float, y: float) -> float:
    """Правая часть"""
    return -50.0 * (y - np.cos(x))


def exact_solution(x: np.ndarray) -> np.ndarray:
    """Точное решение жёсткой задачи."""
    A = 2500.0 / 2501.0
    B = 50.0 / 2501.0
    C = 0.5 - A
    return A * np.cos(x) + B * np.sin(x) + C * np.exp(-50.0 * x)


def plot_all_solutions(
    x_exact: np.ndarray,
    y_exact: np.ndarray,
    solutions: dict,
    save_path: Optional[str] = None,
) -> Figure:
    """
    Построение всех решений на одном графике.
    
    Parameters:
        x_exact: массив x для точного решения.
        y_exact: точное решение.
        solutions: словарь {метка: (x, y)}.
        save_path: путь для сохранения.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=100)
    
    # Левый график: все решения
    ax = axes[0]
    ax.plot(x_exact, y_exact, 'k-', linewidth=2.0, label='Точное решение')
    
    colors = ['b', 'r', 'g', 'm', 'c']
    for i, (label, (x, y)) in enumerate(solutions.items()):
        color = colors[i % len(colors)]
        ax.plot(x, y, '-', color=color, linewidth=1.5, alpha=0.8,
                marker='.', markersize=3, label=label)
    
    ax.set_title('Сравнение решений', fontsize=12, fontweight='bold')
    ax.set_xlabel('$x$', fontsize=11)
    ax.set_ylabel('$y$', fontsize=11)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Правый график: начальный участок (пограничный слой)
    ax = axes[1]
    ax.plot(x_exact[x_exact <= 0.2], y_exact[x_exact <= 0.2],
            'k-', linewidth=2.0, label='Точное решение')
    
    for i, (label, (x, y)) in enumerate(solutions.items()):
        color = colors[i % len(colors)]
        mask = x <= 0.2
        ax.plot(x[mask], y[mask], '-', color=color, linewidth=1.5, alpha=0.8,
                marker='.', markersize=3, label=label)
    
    ax.set_title('Пограничный слой $x \\in [0, 0.2]$', fontsize=12, fontweight='bold')
    ax.set_xlabel('$x$', fontsize=11)
    ax.set_ylabel('$y$', fontsize=11)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"График сохранён: {save_path}")
    
    return fig


class ExplicitEuler:
    """Явный метод Эйлера"""
    
    def __init__(
        self,
        f: Callable[[float, float], float],
        x0: float = 0.0,
        x_end: float = 1.5,
        y0: float = 0.5,
        h: float = 1e-2,
    ):
        self.f = f
        self.x0 = x0
        self.x_end = x_end
        self.y0 = y0
        self.h = h
        self.num_steps = int((x_end - x0) / h)
        
        self.x: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
    
    def solve(self, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        n = self.num_steps
        self.x = np.linspace(self.x0, self.x_end, n + 1)
        self.y = np.zeros(n + 1)
        self.y[0] = self.y0
        
        if verbose:
            print(f"Явный Эйлер: h = {self.h:.0e}, шагов = {n}")
        
        for i in range(n):
            self.y[i + 1] = self.y[i] + self.h * self.f(self.x[i], self.y[i])
        
        return self.x, self.y


class ImplicitMidpoint:
    """Неявный метод средней точки"""
    
    def __init__(
        self,
        x0: float = 0.0,
        x_end: float = 1.5,
        y0: float = 0.5,
        h: float = 0.1,
    ):
        self.x0 = x0
        self.x_end = x_end
        self.y0 = y0
        self.h = h
        self.num_steps = int((x_end - x0) / h)
        
        self.x: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
    
    def solve(self, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        n = self.num_steps
        h = self.h
        
        self.x = np.linspace(self.x0, self.x_end, n + 1)
        self.y = np.zeros(n + 1)
        self.y[0] = self.y0
        
        if verbose:
            print(f"Неявная средняя точка: h = {h:.0e}, шагов = {n}")
        
        denom = 1.0 + 25.0 * h
        for i in range(n):
            x_half = self.x[i] + 0.5 * h
            self.y[i + 1] = (self.y[i] * (1.0 - 25.0 * h) + 50.0 * h * np.cos(x_half)) / denom
        
        return self.x, self.y


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("ЗАДАЧА 9: ЖЁСТКАЯ ЗАДАЧА")
    print("=" * 70)
    print("\n  y' = -50(y - cos x),  y(0) = 0.5,  x ∈ [0, 1.5]")
    print()
    
    # Точное решение на мелкой сетке
    x_exact = np.linspace(0, 1.5, 10000)
    y_exact = exact_solution(x_exact)
    

    print("1) ЯВНЫЙ МЕТОД ЭЙЛЕРА")
    print("-" * 70)
    
    # Явный метод Эйлера с h = 1e-2
    euler_h1 = ExplicitEuler(f=rhs_stiff, x0=0.0, x_end=1.5, y0=0.5, h=1e-2)
    x_e1, y_e1 = euler_h1.solve(verbose=True)
    
    # Явный метод Эйлера с h = 2e-2
    euler_h2 = ExplicitEuler(f=rhs_stiff, x0=0.0, x_end=1.5, y0=0.5, h=2e-2)
    x_e2, y_e2 = euler_h2.solve(verbose=True)
    
    # Явный метод Эйлера с h = 4e-2
    euler_h3 = ExplicitEuler(f=rhs_stiff, x0=0.0, x_end=1.5, y0=0.5, h=4e-2)
    x_e3, y_e3 = euler_h3.solve(verbose=True)
    
    # Неявный метод средней точки с h = 0.1
    print("\n2) НЕЯВНЫЙ МЕТОД СРЕДНЕЙ ТОЧКИ")
    print("-" * 70)
    
    imp_mid = ImplicitMidpoint(x0=0.0, x_end=1.5, y0=0.5, h=0.1)
    x_im, y_im = imp_mid.solve(verbose=True)
    
    print("\n" + "=" * 70)
    print("ПОСТРОЕНИЕ ГРАФИКОВ")
    print("=" * 70)
    
    solutions = {
        f'Явный Эйлер, h=1e-2': (x_e1, y_e1),
        f'Явный Эйлер, h=2e-2': (x_e2, y_e2),
        f'Явный Эйлер, h=4e-2': (x_e3, y_e3),
        f'Неявная средняя точка, h=0.1': (x_im, y_im),
    }
    
    fig = plot_all_solutions(
        x_exact=x_exact,
        y_exact=y_exact,
        solutions=solutions,
        save_path="tasks/9/stiff_problem.png",
    )
    
    # Вывод максимальных ошибок
    print("\n" + "=" * 70)
    print("МАКСИМАЛЬНЫЕ ОШИБКИ")
    print("=" * 70)
    print(f"  {'Метод':<30} {'Шаг':<10} {'max |y_num - y_exact|':<18}")
    print("-" * 60)
    
    for name, (x, y_num) in solutions.items():
        y_ex = exact_solution(x)
        max_err = np.max(np.abs(y_num - y_ex))
        print(f"  {name:<30} {x[1]-x[0]:<10.0e} {max_err:<18.6e}")
    
    plt.show()