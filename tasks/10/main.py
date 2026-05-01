# type: ignore
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional
from matplotlib.figure import Figure


def marsaglia_polar(n: int, seed: Optional[int] = None) -> np.ndarray:
    """Полярный метод Марсальи для генерации стандартных нормальных чисел."""
    rng = np.random.RandomState(seed)
    
    # Генерируем с запасом (каждая итерация даёт 2 числа)
    n_pairs = (n + 1) // 2
    result = np.zeros(n_pairs * 2)
    
    generated = 0
    while generated < n_pairs * 2:
        # U1, U2 ~ U[-1, 1]
        u1 = 2.0 * rng.random() - 1.0
        u2 = 2.0 * rng.random() - 1.0
        
        s = u1 * u1 + u2 * u2
        
        # Проверяем, попали ли в единичный круг
        if s >= 1.0 or s == 0.0:
            continue
        
        # Преобразование Бокса-Мюллера (полярная форма)
        factor = np.sqrt(-2.0 * np.log(s) / s)
        result[generated] = u1 * factor
        result[generated + 1] = u2 * factor
        
        generated += 2
    
    return result[:n]


def compute_statistics(data: np.ndarray) -> Dict[str, float]:
    """Вычисление выборочного среднего и СКО."""
    mu = np.mean(data)
    sigma = np.std(data, ddof=1)  # выборочное СКО (с n-1)
    return {'mean': mu, 'std': sigma}


def run_experiment(
    lgN_values: np.ndarray,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Запуск эксперимента для всех lg N.
    
    Parameters:
        lgN_values: массив значений lg N (например, [1, 2, 3, ...])
        seed: зерно для воспроизводимости.
        verbose: печатать ли таблицу.
    
    Returns:
        словарь с массивами N, mu, sigma.
    """
    results = {'N': [], 'mu': [], 'sigma': []}
    
    if verbose:
        print("\n" + "=" * 70)
        print("ПОЛЯРНЫЙ МЕТОД МАРСАЛЬИ: РЕЗУЛЬТАТЫ")
        print("=" * 70)
        print(f"  {'lg N':<8} {'N':<12} {'Среднее μ':<16} {'СКО σ':<16} {'|μ - 0|':<12} {'|σ - 1|':<12}")
        print("-" * 78)
    
    for lgN in lgN_values:
        N = int(10 ** lgN)
        
        data = marsaglia_polar(n=N, seed=seed)
        stats = compute_statistics(data)
        
        results['N'].append(N)
        results['mu'].append(stats['mean'])
        results['sigma'].append(stats['std'])
        
        if verbose:
            print(
                f"  {lgN:<8.1f} {N:<12d} "
                f"{stats['mean']:<16.10f} {stats['std']:<16.10f} "
                f"{abs(stats['mean']):<12.6e} {abs(stats['std'] - 1.0):<12.6e}"
            )
    
    if verbose:
        print("=" * 78)
    
    return {
        'lgN': np.array(lgN_values),
        'N': np.array(results['N']),
        'mu': np.array(results['mu']),
        'sigma': np.array(results['sigma']),
    }


def plot_convergence(
    results: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
) -> Figure:
    """
    Построение графиков μ(lg N) и σ(lg N).
    
    Parameters:
        results: словарь с массивами lgN, mu, sigma.
        save_path: путь для сохранения.
    
    Returns:
        объект Figure.
    """
    lgN = results['lgN']
    mu = results['mu']
    sigma = results['sigma']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=100)
    
    # График μ(lg N)
    ax = axes[0]
    ax.plot(lgN, mu, 'bo-', linewidth=1.5, markersize=6, label=r'$\mu$')
    ax.axhline(y=0.0, color='r', linestyle='--', linewidth=1.0, alpha=0.7, label='Теор. μ = 0')
    
    # Добавляем доверительные границы +-1.96/sqrt(N) для уровня 95%
    N = results['N']
    ci = 1.96 / np.sqrt(N)
    ax.fill_between(lgN, -ci, ci, alpha=0.2, color='red', label='95% дов. интервал')
    
    ax.set_title(r'Сходимость выборочного среднего $\mu(\lg N)$', fontsize=12, fontweight='bold')
    ax.set_xlabel(r'$\lg N$', fontsize=11)
    ax.set_ylabel(r'$\mu$', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(-2, 2))
    
    # График σ(lg N)
    ax = axes[1]
    ax.plot(lgN, sigma, 'go-', linewidth=1.5, markersize=6, label=r'$\sigma$')
    ax.axhline(y=1.0, color='r', linestyle='--', linewidth=1.0, alpha=0.7, label='Теор. σ = 1')
    
    # Доверительные границы для СКО: +-1.96/sqrt(2N)
    ci_sigma = 1.96 / np.sqrt(2.0 * N)
    ax.fill_between(lgN, 1.0 - ci_sigma, 1.0 + ci_sigma,
                    alpha=0.2, color='red', label='95% дов. интервал')
    
    ax.set_title(r'Сходимость выборочного СКО $\sigma(\lg N)$', fontsize=12, fontweight='bold')
    ax.set_xlabel(r'$\lg N$', fontsize=11)
    ax.set_ylabel(r'$\sigma$', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"График сохранён: {save_path}")
    
    return fig


def plot_distribution_check(
    data: np.ndarray,
    save_path: Optional[str] = None,
) -> Figure:
    """
    Проверка распределения: гистограмма против теоретической N(0,1).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=100)
    
    # Гистограмма
    ax = axes[0]
    ax.hist(data, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black',
            label=f'Марсалья (N={len(data)})')
    
    x_pdf = np.linspace(-4, 4, 200)
    ax.plot(x_pdf, np.exp(-0.5 * x_pdf**2) / np.sqrt(2 * np.pi),
            'r-', linewidth=2, label='N(0,1)')
    
    ax.set_title('Гистограмма сгенерированных чисел', fontsize=12, fontweight='bold')
    ax.set_xlabel('$z$', fontsize=11)
    ax.set_ylabel('Плотность', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Q-Q plot (квантили)
    ax = axes[1]
    sorted_data = np.sort(data)
    theoretical_quantiles = np.random.normal(0, 1, len(data))
    theoretical_quantiles.sort()
    
    ax.plot(theoretical_quantiles, sorted_data, 'b.', markersize=2, alpha=0.5)
    ax.plot([-4, 4], [-4, 4], 'r-', linewidth=1.5, label='y = x')
    
    ax.set_title('Q-Q plot', fontsize=12, fontweight='bold')
    ax.set_xlabel('Теоретические квантили N(0,1)', fontsize=11)
    ax.set_ylabel('Выборочные квантили', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.axis('equal')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"График сохранён: {save_path}")
    
    return fig


if __name__ == "__main__":    
    print("\n" + "=" * 70)
    print("ЗАДАЧА 10: ПОЛЯРНЫЙ МЕТОД МАРСАЛЬИ")
    print("=" * 70)
    print("\n  Генерация стандартно нормальных чисел N(0,1)")
    print("  Полярный метод Марсальи (вариация Бокса-Мюллера)")
    print()
    
    # 1. Эксперимент: lg N = 1, 2, 3, 4, 5, 6, 7
    lgN_values = np.arange(1, 8)
    
    results = run_experiment(lgN_values=lgN_values, seed=42, verbose=True)
    
    # 2. Графики сходимости
    print("\n" + "=" * 70)
    print("ПОСТРОЕНИЕ ГРАФИКОВ")
    print("=" * 70)
    
    fig_conv = plot_convergence(
        results=results,
        save_path="tasks/10/marsaglia_convergence.png",
    )
    
    # Дополнительно: гистограмма для lg N = 7
    print("\n  Генерация N = 10^7 чисел для проверки распределения...")
    data_large = marsaglia_polar(n=10**7, seed=42)
    
    fig_dist = plot_distribution_check(
        data=data_large[::100],  # разреживаем для скорости
        save_path="tasks/10/marsaglia_distribution.png",
    )
    
    # Статистики для lg N = 7
    stats_large = compute_statistics(data_large)
    print(f"\n  Для N = 10^7:")
    print(f"    Выборочное среднее: {stats_large['mean']:.10f}")
    print(f"    Выборочное СКО:     {stats_large['std']:.10f}")
    print(f"    |μ - 0|:            {abs(stats_large['mean']):.6e}")
    print(f"    |σ - 1|:            {abs(stats_large['std'] - 1.0):.6e}")
    
    plt.show()