# type: ignore
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Optional, Tuple
from matplotlib.figure import Figure


def rhs_pendulum(t: float, y: np.ndarray) -> np.ndarray:
    """
    Правая часть системы для математического маятника.
    
    y = [alpha, omega]
    d(alpha)/dt = omega
    d(omega)/dt = -sin(alpha)
    """
    alpha, omega = y
    return np.array([omega, -np.sin(alpha)])


def energy(alpha: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """
    Полная механическая энергия маятника.
    E = omega^2 / 2 + (1 - cos(alpha))
    """
    return 0.5 * omega**2 + (1.0 - np.cos(alpha))


def plot_results(
    t: np.ndarray,
    alpha_lf: np.ndarray,
    omega_lf: np.ndarray,
    alpha_mp: np.ndarray,
    omega_mp: np.ndarray,
    save_path: Optional[str] = None,
) -> Figure:
    """
    Построение трёх графиков в одном окне:
      а) alpha(t) для обоих методов
      б) omega(alpha) - фазовый портрет
      в) E(t) - E_0 - энергия для обоих методов
    """
    E_lf = energy(alpha_lf, omega_lf)
    E_mp = energy(alpha_mp, omega_mp)
    E0 = E_lf[0]  # начальная энергия
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=100)
    
    # а) alpha(t)
    ax = axes[0]
    ax.plot(t, alpha_lf, 'b-', linewidth=1.2, label='Leap-frog')
    ax.plot(t, alpha_mp, 'r--', linewidth=1.2, label='Средняя точка')
    ax.set_title('а) $\\alpha(t)$', fontsize=12)
    ax.set_xlabel('$t$', fontsize=11)
    ax.set_ylabel('$\\alpha$', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # б) omega(alpha) - фазовый портрет
    ax = axes[1]
    ax.plot(alpha_lf, omega_lf, 'b-', linewidth=1.0, alpha=0.9, label='Leap-frog')
    ax.plot(alpha_mp, omega_mp, 'r--', linewidth=1.0, alpha=0.9, label='Средняя точка')
    ax.set_title('б) Фазовый портрет $\\omega(\\alpha)$', fontsize=12)
    ax.set_xlabel('$\\alpha$', fontsize=11)
    ax.set_ylabel('$\\omega$', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.axis('equal')
    
    # в) E(t) - энергия
    ax = axes[2]
    ax.plot(t, E_lf - E0, 'b-', linewidth=1.2, label=f'Leap-frog (max дрейф = {np.max(np.abs(E_lf - E0)):.2e})')
    ax.plot(t, E_mp - E0, 'r--', linewidth=1.2, label=f'Средняя точка (max дрейф = {np.max(np.abs(E_mp - E0)):.2e})')
    ax.set_title('в) Дрейф энергии $E(t) - E_0$', fontsize=12)
    ax.set_xlabel('$t$', fontsize=11)
    ax.set_ylabel('$E - E_0$', fontsize=11)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"График сохранён: {save_path}")
    
    return fig


class LeapFrog:
    """Метод Леап-Форда для системы"""    
    def __init__(
        self,
        f: Callable[[float, np.ndarray], np.ndarray],
        t0: float = 0.0,
        t_end: float = 100.0,
        alpha0: float = 1.0,
        omega0: float = 0.0,
        h: float = 0.1,
    ):
        self.f = f
        self.t0 = t0
        self.t_end = t_end
        self.alpha0 = alpha0
        self.omega0 = omega0
        self.h = h
        self.num_steps = int((t_end - t0) / h)
        
        self.t: Optional[np.ndarray] = None
        self.alpha: Optional[np.ndarray] = None
        self.omega: Optional[np.ndarray] = None
    
    def solve(self, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        h = self.h
        n = self.num_steps
        
        self.t = np.linspace(self.t0, self.t_end, n + 1)
        self.alpha = np.zeros(n + 1)
        self.omega = np.zeros(n + 1)
        
        self.alpha[0] = self.alpha0
        self.omega[0] = self.omega0
        
        if verbose:
            print(f"Схема Leap-Frog: h = {h}, шагов = {n}")
        
        for i in range(n):
            alpha_n = self.alpha[i]
            omega_n = self.omega[i]
            
            # Полушаг для omega
            f_alpha_n = self.f(0, np.array([alpha_n, omega_n]))[1]  # -sin(alpha_n)
            omega_half = omega_n + 0.5 * h * f_alpha_n
            
            # Полный шаг для alpha
            alpha_next = alpha_n + h * omega_half
            
            # Полный шаг для omega
            f_alpha_next = self.f(0, np.array([alpha_next, 0]))[1]  # -sin(alpha_next)
            omega_next = omega_half + 0.5 * h * f_alpha_next
            
            self.alpha[i + 1] = alpha_next
            self.omega[i + 1] = omega_next
        
        return self.t, self.alpha, self.omega


class MidpointExplicit:
    """Явный метод средней точки (РК2)"""
    
    def __init__(
        self,
        f: Callable[[float, np.ndarray], np.ndarray],
        t0: float = 0.0,
        t_end: float = 100.0,
        alpha0: float = 1.0,
        omega0: float = 0.0,
        h: float = 0.1,
    ):
        self.f = f
        self.t0 = t0
        self.t_end = t_end
        self.alpha0 = alpha0
        self.omega0 = omega0
        self.h = h
        self.num_steps = int((t_end - t0) / h)
        
        self.t: Optional[np.ndarray] = None
        self.alpha: Optional[np.ndarray] = None
        self.omega: Optional[np.ndarray] = None
    
    def _midpoint_step(self, t_n: float, y_n: np.ndarray) -> np.ndarray:
        h = self.h
        k1 = self.f(t_n, y_n)
        k2 = self.f(t_n + h / 2, y_n + k1 * h / 2)
        return y_n + k2 * h
    
    def solve(self, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = self.num_steps
        
        self.t = np.linspace(self.t0, self.t_end, n + 1)
        self.alpha = np.zeros(n + 1)
        self.omega = np.zeros(n + 1)
        
        self.alpha[0] = self.alpha0
        self.omega[0] = self.omega0
        
        if verbose:
            print(f"Явный метод средней точки: h = {self.h}, шагов = {n}")
        
        for i in range(n):
            y_n = np.array([self.alpha[i], self.omega[i]])
            t_n = self.t[i]
            
            y_next = self._midpoint_step(t_n, y_n)
            
            self.alpha[i + 1] = y_next[0]
            self.omega[i + 1] = y_next[1]
        
        return self.t, self.alpha, self.omega


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("ЗАДАЧА 8: НЕЛИНЕЙНЫЕ КОЛЕБАНИЯ МАЯТНИКА")
    print("=" * 70)
    print("\nУравнение: alpha'' = -sin(alpha)")
    print("Начальные условия: alpha(0) = 1, omega(0) = 0")
    print("Энергия: E = omega^2/2 + (1 - cos(alpha))")
    print()
    
    print("1) СХЕМА LEAP-FROG")
    print("-" * 70)
    
    lf = LeapFrog(
        f=rhs_pendulum,
        t0=0.0,
        t_end=100.0,
        alpha0=1.0,
        omega0=0.0,
        h=0.1,
    )
    t_lf, alpha_lf, omega_lf = lf.solve(verbose=True)
    
    E_lf = energy(alpha_lf, omega_lf)
    E0 = E_lf[0]
    drift_lf = np.max(np.abs(E_lf - E0))
    
    print(f"\n  Начальная энергия: {E0:.10f}")
    print(f"  Конечная энергия:  {E_lf[-1]:.10f}")
    print(f"  Максимальный дрейф: {drift_lf:.6e}")
    print(f"  Относительный дрейф: {drift_lf / E0:.6e}")
    
    print("\n2) ЯВНЫЙ МЕТОД СРЕДНЕЙ ТОЧКИ")
    print("-" * 70)
    
    mp = MidpointExplicit(
        f=rhs_pendulum,
        t0=0.0,
        t_end=100.0,
        alpha0=1.0,
        omega0=0.0,
        h=0.1,
    )
    t_mp, alpha_mp, omega_mp = mp.solve(verbose=True)
    
    E_mp = energy(alpha_mp, omega_mp)
    drift_mp = np.max(np.abs(E_mp - E0))
    
    print(f"\n  Начальная энергия: {E0:.10f}")
    print(f"  Конечная энергия:  {E_mp[-1]:.10f}")
    print(f"  Максимальный дрейф: {drift_mp:.6e}")
    print(f"  Относительный дрейф: {drift_mp / E0:.6e}")
    
    # Сравнение
    print("\n" + "=" * 70)
    print("СРАВНЕНИЕ ДРЕЙФА ЭНЕРГИИ")
    print("=" * 70)
    print(f"  Leap-frog:      {drift_lf:.6e}  (симплектическая схема - дрейф мал)")
    print(f"  Средняя точка:  {drift_mp:.6e}  (несимплектическая - дрейф заметен)")
    print(f"  Отношение:      {drift_mp / drift_lf:.2f} раз")
    print()
    
    # График
    print("\n" + "=" * 70)
    print("ПОСТРОЕНИЕ ГРАФИКОВ")
    print("=" * 70)
    
    fig = plot_results(
        t=t_lf,
        alpha_lf=alpha_lf,
        omega_lf=omega_lf,
        alpha_mp=alpha_mp,
        omega_mp=omega_mp,
        save_path="tasks/8/pendulum_results.png",
    )
    
    plt.show()