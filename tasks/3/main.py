import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def task3_potential(x_vec):
    """
    Вычисление потенциала V(x):
    V(x) = sum_{k=0}^{N-1} ((x_{k+1} - x_k) / 1e-4)^2
           + sum_{k=1}^{N-1} sqrt(|k - N/2|) * x_k

    :param x_vec(list[float]): Вектор x длины N+1 (включая граничные узлы x_0 и x_N)
    :return(float): Значение потенциала
    """
    N = len(x_vec) - 1
    alpha = 1e-4

    V1 = sum(((x_vec[k + 1] - x_vec[k]) / alpha) ** 2 for k in range(N))
    V2 = sum(math.sqrt(abs(k - N / 2)) * x_vec[k] for k in range(1, N))

    return V1 + V2


def task3_gradient_analytical(x_vec):
    """
    Аналитический градиент V(x) по внутренним узлам j = 1, ..., N-1.

    dV/dx_j = (2 / alpha^2) * (2*x_j - x_{j-1} - x_{j+1})
              + sqrt(|j - N/2|)

    Граничные узлы x_0 = x_N = 0 зафиксированы, их компоненты не возвращаются.

    :param x_vec(list[float]): Вектор x длины N+1
    :return(list[float]): Градиент длины N-1 (только внутренние узлы)
    """
    N = len(x_vec) - 1
    alpha = 1e-4
    coeff = 2.0 / (alpha**2)

    grad = []
    for j in range(1, N):
        dV1 = coeff * (2 * x_vec[j] - x_vec[j - 1] - x_vec[j + 1])
        dV2 = math.sqrt(abs(j - N / 2))
        grad.append(dV1 + dV2)

    return grad


class Task3:
    """
    Задание 3: минимизация функции V(x) методом динамического твёрдого шарика.

    Уравнение движения:
    v^{n+1} = gamma * (v^n - dt * grad V(x^n))
    x^{n+1} = x^n + dt * v^{n+1}

    Граничные условия Дирихле: x_0 = x_N = 0.
    Условие сходимости: max_k |grad_k V| < tol.
    """

    def __init__(
        self,
        potential_fun,
        analytical_grad_fun=None,
        N=100,
        dt=5e-5,
        gamma=0.95,
        tol=6e-10,
        max_iter=100000,
        x0=None,
        v0=None,
        grad_eps=1e-6,
    ):
        """
        :param potential_fun(function): Функция потенциала
        :param analytical_grad_fun(function | None): Функция аналитического градиента V.
                                                     Если None - считается численно.
        :param N(int): Число внутренних интервалов; вектор x имеет длину N+1
        :param dt(float): Шаг по времени
        :param gamma(float): Коэффициент трения (демпфирование)
        :param tol(float): Порог сходимости по норме градиента
        :param max_iter(int): Максимальное число итераций
        :param x0(list[float] | None): Начальное приближение длины N+1;
                                       если None — нулевой вектор
        :param v0(list[float] | None): Начальная скорость длины N-1 (внутренние узлы);
                                       если None — нулевой вектор (шарик стартует из покоя)
        :param grad_eps(float): Шаг конечной разности для численного градиента
        """
        self.potential_fun = potential_fun
        self.analytical_grad_fun = analytical_grad_fun
        self.N = N
        self.dt = dt
        self.gamma = gamma
        self.tol = tol
        self.max_iter = max_iter
        self.grad_eps = grad_eps

        # Начальное приближение
        if x0 is None:
            self.x0 = [0.0] * (N + 1)
        else:
            if len(x0) != N + 1:
                raise ValueError(f"x0 должен иметь длину N+1 = {N + 1}")
            self.x0 = list(x0)

        # Начальная скорость
        if v0 is None:
            self.v0 = [0.0] * (N - 1)
        else:
            if len(v0) != N - 1:
                raise ValueError(f"v0 должен иметь длину N-1 = {N - 1}")
            self.v0 = list(v0)

        # Результаты (заполняются после solve)
        self.x_min = None
        self.V_min = None
        self.converged = False
        self.n_iter = 0
        self.grad_history = []
        self.V_history = []
        self.record_every = max(1, max_iter // 1000)

    def _gradient_numerical(self, x_vec):
        """Численный градиент V(x) по внутренним узлам j = 1, ..., N-1"""
        N = len(x_vec) - 1
        grad = []

        for j in range(1, N):
            x_plus = x_vec[:]
            x_minus = x_vec[:]
            x_plus[j] += self.grad_eps
            x_minus[j] -= self.grad_eps
            dVj = (
                self.potential_fun(x_plus) - self.potential_fun(x_minus)
            ) / (2 * self.grad_eps)
            grad.append(dVj)

        return grad

    def _gradient(self, x_vec):
        """Выбор метода вычисления градиента."""
        if self.analytical_grad_fun is not None:
            return self.analytical_grad_fun(x_vec)
        else:
            return self._gradient_numerical(x_vec)

    def solve(self, verbose=False):
        """
        Запуск итераций метода твёрдого шарика.

        Возвращает:
        - converged(bool): сошёлся ли метод
        - n_iter(int): число итераций
        - V_min(float): минимальное значение потенциала
        """
        N = self.N
        dt = self.dt
        gamma = self.gamma

        # Текущие координаты и скорости
        x_cur = self.x0[:]
        v_cur = self.v0[:]

        # Очищаем историю
        self.grad_history = []
        self.V_history = []
        self.converged = False

        for iteration in range(self.max_iter):
            # Вычисляем градиент в текущей точке (длина N-1)
            grad = self._gradient(x_cur)
            grad_max = max(abs(g) for g in grad)

            # Запись истории (прореженная)
            if iteration % self.record_every == 0:
                self.grad_history.append(grad_max)
                self.V_history.append(self.potential_fun(x_cur))

            # Проверка сходимости
            if grad_max < self.tol:
                self.converged = True
                self.n_iter = iteration
                break

            # Проверка на взрыв
            V_cur = self.potential_fun(x_cur)
            if abs(V_cur) > 1e10 or math.isnan(V_cur):
                self.converged = False
                self.n_iter = iteration
                break

            # Шаг метода твёрдого шарика
            x_new = x_cur[:]
            v_new = [0.0] * (N - 1)

            for idx, j in enumerate(range(1, N)):
                # v_new = gamma * (v_cur - dt * grad)
                v_new[idx] = gamma * (v_cur[idx] - dt * grad[idx])
                # x_new = x_cur + dt * v_new
                x_new[j] = x_cur[j] + dt * v_new[idx]

            # Граничные условия Дирихле
            x_new[0] = 0.0
            x_new[N] = 0.0

            # Обновляем состояния для следующей итерации
            x_cur = x_new
            v_cur = v_new
        else:
            # Цикл завершился без break - достигнут максимум итераций
            self.n_iter = self.max_iter

        # Сохраняем финальный результат
        self.x_min = x_cur
        self.V_min = self.potential_fun(x_cur)

        return self.converged, self.n_iter, self.V_min

    def plot_results(self, figsize=(14, 10), save_path=None):
        """Построение графиков результатов"""
        if self.x_min is None:
            raise RuntimeError("Сначала вызовите solve()")

        N = self.N
        alpha = 1e-4
        x_vec = self.x_min
        k_nodes = list(range(N + 1))

        # Конечные разности второго порядка
        curvature = [
            (x_vec[k + 1] - 2 * x_vec[k] + x_vec[k - 1]) / alpha**2
            for k in range(1, N)
        ]

        # Теоретическая кривизна для сравнения
        theoretical_curvature = [
            -0.5 * math.sqrt(abs(k - N / 2)) for k in range(1, N)
        ]

        iters_recorded = [
            i * self.record_every for i in range(len(self.grad_history))
        ]

        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.32)

        # --- График 1: Профиль решения ---
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(k_nodes, x_vec, "b-", linewidth=1.8, label=r"$x_k$")
        ax1.axhline(0, color="k", linewidth=0.7, alpha=0.5)
        ax1.set_xlabel("k", fontsize=11)
        ax1.set_ylabel(r"$x_k$", fontsize=11)
        ax1.set_title("Профиль решения", fontsize=12)
        ax1.grid(True, alpha=0.3, linestyle="--")
        ax1.legend()

        # --- График 2: Сходимость градиента ---
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.semilogy(iters_recorded, self.grad_history, "r-", linewidth=1.2)
        ax2.axhline(
            self.tol,
            color="gray",
            linestyle="--",
            linewidth=1,
            label=f"tol = {self.tol:.0e}",
        )
        ax2.set_xlabel("Итерация", fontsize=11)
        ax2.set_ylabel(r"$\max_k |\nabla_k V|$", fontsize=11)
        ax2.set_title("Сходимость (норма градиента)", fontsize=12)
        ax2.grid(True, alpha=0.3, which="both", linestyle="--")
        ax2.legend(fontsize=9)

        # --- График 3: Убывание потенциала (ИСПРАВЛЕНО - линейный масштаб) ---
        ax3 = fig.add_subplot(gs[1, 0])

        # Находим минимальное значение для сдвига (чтобы сделать все значения положительными)
        V_min_history = min(self.V_history)
        if V_min_history < 0:
            # Сдвигаем, чтобы минимальное значение было чуть выше 0
            shift = abs(V_min_history) + 1e-10
            V_shifted = [v + shift for v in self.V_history]
            ax3.semilogy(
                iters_recorded,
                V_shifted,
                "g-",
                linewidth=1.2,
                label=f"V(x) + {shift:.2e}",
            )
            ax3.set_ylabel(r"$V(\vec{x})$ + shift", fontsize=11)
        else:
            ax3.semilogy(iters_recorded, self.V_history, "g-", linewidth=1.2)
            ax3.set_ylabel(r"$V(\vec{x})$", fontsize=11)

        ax3.set_xlabel("Итерация", fontsize=11)
        ax3.set_title("Убывание потенциала", fontsize=12)
        ax3.grid(True, alpha=0.3, which="both", linestyle="--")
        ax3.legend(fontsize=9)

        # Добавляем информацию о значениях
        V_init = self.V_history[0]
        V_final = self.V_history[-1]
        ax3.text(
            0.02,
            0.98,
            f"V_init = {V_init:.6f}\nV_final = {V_final:.6f}",
            transform=ax3.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # --- График 4: Кривизна профиля (факт vs теория) ---
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(
            range(1, N), curvature, "b-", linewidth=1.2, label="Фактическая"
        )
        ax4.plot(
            range(1, N),
            theoretical_curvature,
            "r--",
            linewidth=1.2,
            label="Теоретическая",
        )
        ax4.axhline(0, color="k", linewidth=0.7, alpha=0.5)
        ax4.set_xlabel("k", fontsize=11)
        ax4.set_ylabel(
            r"$(x_{k+1} - 2x_k + x_{k-1})\,/\,\alpha^2$", fontsize=11
        )
        ax4.set_title("Кривизна профиля", fontsize=12)
        ax4.grid(True, alpha=0.3, linestyle="--")
        ax4.legend(fontsize=9)

        grad_method = (
            "аналитический" if self.analytical_grad_fun else "численный"
        )
        status = "✅ сошёлся" if self.converged else "❌ НЕ сошёлся"
        fig.suptitle(
            f"Задание 3 | N={N}, dt={self.dt:.2e}, γ={self.gamma} | "
            f"Градиент: {grad_method} | {status} за {self.n_iter} итераций",
            fontsize=11,
        )

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"График сохранён в: {save_path}")

        plt.show()

    def print_result(self):
        """Вывод минимального значения V и профиля решения."""
        if self.x_min is None:
            raise RuntimeError("Сначала вызовите solve()")

        grad_method = (
            "аналитический" if self.analytical_grad_fun else "численный"
        )
        status = (
            "Сошёлся"
            if self.converged
            else "НЕ сошёлся (лимит итераций или взрыв)"
        )

        print("=" * 60)
        print("Задание 3 — метод твердого шарика")
        print("=" * 60)
        print(f"N            = {self.N}")
        print(f"dt           = {self.dt:.2e}")
        print(f"gamma        = {self.gamma}")
        print(f"tol          = {self.tol:.2e}")
        print(f"Градиент     : {grad_method}")
        print(f"Статус       : {status}")
        print(f"Итераций     : {self.n_iter}")
        print("-" * 60)
        print(f"V_min        = {self.V_min:.10f}")
        print("-" * 60)
        print("Профиль решения {x_k} (первые 10 и последние 10):")

        if self.N <= 20:
            for k, xk in enumerate(self.x_min):
                print(f"  x[{k:4d}] = {xk:+.10f}")
        else:
            for k in range(10):
                print(f"  x[{k:4d}] = {self.x_min[k]:+.10f}")
            print("  ...")
            for k in range(self.N - 9, self.N + 1):
                print(f"  x[{k:4d}] = {self.x_min[k]:+.10f}")
        print("=" * 60)


def find_optimal_parameters(N=100, tol=6e-10, max_iter=50000):
    """
    Автоматический подбор параметров dt и gamma
    """
    print("=" * 70)
    print("АВТОМАТИЧЕСКИЙ ПОДБОР ПАРАМЕТРОВ")
    print("=" * 70)
    print()

    # Сетки параметров
    dt_values = [
        1e-3,
        5e-4,
        1e-4,
        5e-5,
        1e-5,
        5e-6,
        1e-6,
        5e-7,
        1e-7,
        5e-8,
        1e-8,
    ]
    gamma_values = [0.99, 0.95, 0.9, 0.8, 0.5, 0.0]

    best_params = None
    best_iterations = float("inf")
    best_V = None

    results = []

    print("Тестирование параметров...")
    print("-" * 70)
    print(
        f"{'dt':<12} {'gamma':<8} {'Статус':<12} {'Итераций':<10} {'V_min':<15}"
    )
    print("-" * 70)

    for dt in dt_values:
        for gamma in gamma_values:
            solver = Task3(
                potential_fun=task3_potential,
                analytical_grad_fun=task3_gradient_analytical,
                N=N,
                dt=dt,
                gamma=gamma,
                tol=tol,
                max_iter=max_iter,
            )

            converged, n_iter, V_min = solver.solve(verbose=False)

            status = "✅ СОШЁЛСЯ" if converged else "❌ НЕТ"

            print(
                f"{dt:<12.0e} {gamma:<8.2f} {status:<12} {n_iter:<10} {V_min:<15.10f}"
            )

            if converged:
                results.append((dt, gamma, n_iter, V_min))
                if n_iter < best_iterations:
                    best_iterations = n_iter
                    best_params = (dt, gamma)
                    best_V = V_min

    print("-" * 70)
    print()

    if best_params is not None:
        print(f"✅ НАЙДЕНЫ ОПТИМАЛЬНЫЕ ПАРАМЕТРЫ:")
        print(f"   dt = {best_params[0]:.2e}")
        print(f"   gamma = {best_params[1]:.2f}")
        print(f"   Итераций до сходимости: {best_iterations}")
        print(f"   V_min = {best_V:.12f}")

        # Сортируем результаты по числу итераций
        results.sort(key=lambda x: x[2])
        print()
        print("Топ-5 лучших комбинаций:")
        print("-" * 50)
        for i, (dt, gamma, n_iter, V_min) in enumerate(results[:5], 1):
            print(f"{i}. dt={dt:.2e}, gamma={gamma:.2f}: {n_iter} итераций")
    else:
        print("❌ НЕ НАЙДЕНО параметров, при которых метод сходится")
        print("   Попробуйте увеличить max_iter или изменить сетку параметров")
        best_params = (None, None)

    return best_params


if __name__ == "__main__":
    # Сначала находим оптимальные параметры
    optimal_dt, optimal_gamma = find_optimal_parameters(N=100, max_iter=50000)

    if optimal_dt is not None:
        print()
        print("=" * 70)
        print("ЗАПУСК С ОПТИМАЛЬНЫМИ ПАРАМЕТРАМИ")
        print("=" * 70)
        print()

        # Запускаем с оптимальными параметрами
        solver = Task3(
            potential_fun=task3_potential,
            analytical_grad_fun=task3_gradient_analytical,
            N=100,
            dt=optimal_dt,
            gamma=optimal_gamma,
            tol=6e-10,
            max_iter=50000,
        )

        solver.solve(verbose=True)
        solver.print_result()
        solver.plot_results()
