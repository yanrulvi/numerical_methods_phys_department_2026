# type: ignore
import numpy as np
from typing import Callable, List, Tuple, Optional
import matplotlib.pyplot as plt


class Task2:
    """
    Решение системы нелинейных уравнений методом Ньютона.
    """

    def __init__(
        self,
        functions: List[Callable],
        n: int,
        jacobian_step: float = 1e-8,
        tol: float = 1e-12,
        max_iter: int = 100,
        use_analytic_jacobian: bool = False,
    ):
        """
        Инициализация класса

        Параметры:
        -----------
        functions : List[Callable]
            Список функций F_i(x1, x2, ..., xn)
        n : int
            Размерность системы (количество переменных)
        jacobian_step : float
            Шаг для численного вычисления производных
        tol : float
            Точность (норма вектора смещения)
        max_iter : int
            Максимальное число итераций
        use_analytic_jacobian : bool
            Использовать аналитический якобиан (если переопределён метод jacobian)
        """
        self.functions = functions
        self.n = n
        self.jacobian_step = jacobian_step
        self.tol = tol
        self.max_iter = max_iter
        self.use_analytic_jacobian = use_analytic_jacobian

        self.solution = None
        self.iterations = 0
        self.history = {"x": [], "F": [], "delta": []}
        self.converged = False

    def F(self, x: np.ndarray) -> np.ndarray:
        """Вычисление вектора значений функций в точке x"""
        return np.array([f(*x) for f in self.functions], dtype=float)

    def jacobian_numerical(self, x: np.ndarray) -> np.ndarray:
        """
        Численное вычисление матрицы Якоби методом конечных разностей

        J[i,j] = dF_i/dx_j =~ (F_i(x + h*e_j) - F_i(x)) / h
        """
        J = np.zeros((self.n, self.n))
        Fx = self.F(x)

        for j in range(self.n):
            # Создаём вектор с возмущением по j-й координате
            x_plus_h = x.copy()
            x_plus_h[j] += self.jacobian_step

            Fxh = self.F(x_plus_h)

            # Вычисляем производные для всех функций
            J[:, j] = (Fxh - Fx) / self.jacobian_step

        return J

    def jacobian_analytic(self, x: np.ndarray) -> np.ndarray:
        """
        Аналитический якобиан (должен быть переопределён в наследниках)
        По умолчанию использует численное дифференцирование
        """
        if hasattr(self, "_analytic_jacobian"):
            return self._analytic_jacobian(x)
        else:
            return self.jacobian_numerical(x)

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """Вычисление матрицы Якоби (автоматический выбор метода)"""
        if self.use_analytic_jacobian:
            return self.jacobian_analytic(x)
        else:
            return self.jacobian_numerical(x)

    def solve(self, x0: np.ndarray, verbose: bool = True) -> np.ndarray:
        """
        Решение системы методом Ньютона

        Параметры:
        -----------
        x0 : np.ndarray
            Начальное приближение
        verbose : bool
            Выводить ли информацию о процессе решения

        Возвращает:
        -----------
        np.ndarray : Найденное решение
        """
        x = x0.copy().astype(float)
        self.history = {"x": [x.copy()], "F": [self.F(x)], "delta": [np.inf]}

        if verbose:
            print("=" * 80)
            print("РЕШЕНИЕ СИСТЕМЫ НЕЛИНЕЙНЫХ УРАВНЕНИЙ МЕТОДОМ НЬЮТОНА")
            print("=" * 80)
            print(f"\nРазмерность системы: {self.n}")
            print(f"Начальное приближение: {x0}")
            print(f"Точность: ||dx|| < {self.tol}")
            print(
                f"Метод вычисления Якоби: {'аналитический' if self.use_analytic_jacobian else 'численный'}"
            )
            print("\n" + "-" * 80)
            print(
                f"{'Итер':<6} {'Норма dx':<15} {'Норма F(x)':<15} {'Решение':<50}"
            )
            print("-" * 80)

        for iteration in range(self.max_iter):
            # Вычисляем значения функций и матрицу Якоби
            F_val = self.F(x)
            J_val = self.jacobian(x)

            # Решаем систему J·dx = -F
            try:
                delta = np.linalg.solve(J_val, -F_val)
            except np.linalg.LinAlgError:
                if verbose:
                    print(
                        f"\n⚠ Ошибка: матрица Якоби вырождена на итерации {iteration}"
                    )
                self.iterations = iteration
                return x

            # Обновляем решение
            delta_norm = np.linalg.norm(delta)
            if delta_norm > 0.1:
                delta = delta * 0.1 / delta_norm
                delta_norm = 0.1
            x_new = x + delta
            
            F_norm = np.linalg.norm(F_val)

            # Сохраняем историю
            self.history["x"].append(x_new.copy())
            self.history["F"].append(self.F(x_new))
            self.history["delta"].append(delta_norm)

            if verbose and (iteration % 1 == 0 or delta_norm < 1e-6):
                x_str = " ".join([f"{xi:12.8f}" for xi in x_new[:3]])
                if self.n > 3:
                    x_str += " ..."
                print(
                    f"{iteration:<6} {delta_norm:<15.6e} {F_norm:<15.6e} {x_str}"
                )

            # Проверка сходимости
            if delta_norm < self.tol:
                if verbose:
                    print(
                        f"\n✓ Сходимость достигнута за {iteration + 1} итераций"
                    )
                self.solution = x_new
                self.iterations = iteration + 1
                self.converged = True
                return x_new

            x = x_new

        if verbose:
            print(
                f"\n⚠ Достигнуто максимальное число итераций ({self.max_iter})"
            )
        self.solution = x
        self.iterations = self.max_iter
        return x

    def verify_solution(
        self, x: np.ndarray = None
    ) -> Tuple[np.ndarray, float]:
        """
        Проверка решения подстановкой в исходные уравнения

        Возвращает:
        -----------
        tuple: (вектор невязок, норма невязки)
        """
        if x is None:
            if self.solution is None:
                raise ValueError(
                    "Решение ещё не найдено. Сначала вызовите solve()"
                )
            x = self.solution

        residuals = self.F(x)
        norm_residual = np.linalg.norm(residuals)
        return residuals, norm_residual

    def print_solution(self):
        """Вывод найденного решения"""
        if self.solution is None:
            print("Решение ещё не найдено. Сначала вызовите solve()")
            return

        print("\n" + "=" * 80)
        print("РЕЗУЛЬТАТЫ РЕШЕНИЯ")
        print("=" * 80)
        print(f"\nКоличество итераций: {self.iterations}")
        print(
            f"Сходимость: {'достигнута ✓' if self.converged else 'не достигнута ⚠'}"
        )

        print("\nНайденное решение:")
        for i, xi in enumerate(self.solution, 1):
            print(f"  x{i} = {xi:.15f}")

        residuals, norm_residual = self.verify_solution()
        print("\nПроверка подстановкой (значения функций в найденной точке):")
        for i, r in enumerate(residuals, 1):
            print(f"  F{i}(x) = {r:.12e}")

        print(f"\nНорма вектора невязки: {norm_residual:.12e}")

        if norm_residual < 1e-10:
            print("\n✓ Решение удовлетворяет системе с высокой точностью")
        elif norm_residual < 1e-6:
            print(
                "\n⚠ Решение удовлетворяет системе с удовлетворительной точностью"
            )
        else:
            print(
                "\n✗ Решение НЕ удовлетворяет системе! Возможно, требуется больше итераций."
            )

    def plot_convergence(self, save_path: Optional[str] = None):
        """Построение графика сходимости"""
        if not self.history["delta"]:
            print("Нет данных для построения графика")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # График нормы смещения
        iterations = range(len(self.history["delta"]))
        ax1.semilogy(
            iterations,
            self.history["delta"],
            "b-o",
            markersize=3,
            linewidth=1.5,
        )
        ax1.axhline(
            y=self.tol,
            color="r",
            linestyle="--",
            label=f"Точность ({self.tol})",
        )
        ax1.set_xlabel("Номер итерации")
        ax1.set_ylabel("Норма вектора смещения ||dx||")
        ax1.set_title("Сходимость метода Ньютона (норма смещения)")
        ax1.grid(True, alpha=0.3, linestyle="--")
        ax1.legend()

        # График нормы невязки
        F_norms = [np.linalg.norm(f) for f in self.history["F"]]
        ax2.semilogy(iterations, F_norms, "g-o", markersize=3, linewidth=1.5)
        ax2.set_xlabel("Номер итерации")
        ax2.set_ylabel("Норма вектора невязки ||F(x)||")
        ax2.set_title("Сходимость метода Ньютона (норма невязки)")
        ax2.grid(True, alpha=0.3, linestyle="--")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"График сохранён в: {save_path}")

        plt.show()


def solve_system_from_task():
    """Решение системы из задания 2"""
    print("\n" + "=" * 80)
    print("ЗАДАНИЕ 2: Решение системы нелинейных уравнений")
    print("=" * 80)
    
    # Определяем функции системы
    def f1(x1, x2, x3):
        return 3 * x1 - np.cos(x2 * x3) - 0.5
    
    def f2(x1, x2, x3):
        return x1**2 - 81 * (x2 + 0.1)**2 + np.sin(x3) + 1.06
    
    def f3(x1, x2, x3):
        return np.exp(-x1 * x2) + 20 * x3 + (10 * np.pi) / 3 - 1
    
    functions = [f1, f2, f3]
    
    # Создаём решатель с численным якобианом
    solver = Task2(
        functions=functions,
        n=3,
        tol=1e-12,
        max_iter=100,
        use_analytic_jacobian=False,
    )
    
    # Начальная точка - начало координат
    x0 = np.array([0.0, 0.0, 0.0])
    
    # Решаем систему
    solution = solver.solve(x0, verbose=True)
    
    # Выводим результаты
    solver.print_solution()
    
    # Строим график сходимости
    solver.plot_convergence()
    
    return solver


def solve_system_with_analytic_jacobian():
    """Решение системы с аналитическим якобианом (для сравнения)"""
    print("\n" + "=" * 80)
    print("ДОПОЛНИТЕЛЬНО: Решение с аналитическим якобианом")
    print("=" * 80)
    
    class AnalyticSolver(Task2):
        def _analytic_jacobian(self, x: np.ndarray) -> np.ndarray:
            """Аналитический якобиан для системы из задания"""
            x1, x2, x3 = x
            J = np.zeros((3, 3))
            
            # Производные первого уравнения
            J[0, 0] = 3
            J[0, 1] = x3 * np.sin(x2 * x3)
            J[0, 2] = x2 * np.sin(x2 * x3)
            
            # Производные второго уравнения
            J[1, 0] = 2 * x1
            J[1, 1] = -162 * (x2 + 0.1)
            J[1, 2] = np.cos(x3)
            
            # Производные третьего уравнения
            J[2, 0] = -x2 * np.exp(-x1 * x2)
            J[2, 1] = -x1 * np.exp(-x1 * x2)
            J[2, 2] = 20
            
            return J
    
    def f1(x1, x2, x3):
        return 3 * x1 - np.cos(x2 * x3) - 0.5
    
    def f2(x1, x2, x3):
        return x1**2 - 81 * (x2 + 0.1)**2 + np.sin(x3) + 1.06
    
    def f3(x1, x2, x3):
        return np.exp(-x1 * x2) + 20 * x3 + (10 * np.pi) / 3 - 1
    
    functions = [f1, f2, f3]
    
    solver = AnalyticSolver(
        functions=functions,
        n=3,
        tol=1e-12,
        max_iter=100,
        use_analytic_jacobian=True,
    )
    
    x0 = np.array([0.0, 0.0, 0.0])
    solution = solver.solve(x0, verbose=True)
    solver.print_solution()
    
    return solver


if __name__ == "__main__":
    # Решаем основную задачу
    solver_numerical = solve_system_from_task()
    
    # Для сравнения решаем с аналитическим якобианом
    print("\n" + "=" * 80)
    print("СРАВНЕНИЕ МЕТОДОВ")
    print("=" * 80)
    solver_analytic = solve_system_with_analytic_jacobian()
    
    # Сравниваем результаты
    print("\n" + "=" * 80)
    print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
    print("=" * 80)
    print("\nЧисленный якобиан:")
    print(f"  x = [{solver_numerical.solution[0]:.12f}, {solver_numerical.solution[1]:.12f}, {solver_numerical.solution[2]:.12f}]")
    print(f"  Итераций: {solver_numerical.iterations}")
    
    print("\nАналитический якобиан:")
    print(f"  x = [{solver_analytic.solution[0]:.12f}, {solver_analytic.solution[1]:.12f}, {solver_analytic.solution[2]:.12f}]")
    print(f"  Итераций: {solver_analytic.iterations}")
    
    diff = np.linalg.norm(solver_numerical.solution - solver_analytic.solution)
    print(f"\nРазница между решениями: {diff:.12e}")