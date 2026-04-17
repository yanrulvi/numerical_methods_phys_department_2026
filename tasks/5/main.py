# type: ignore
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional


class Task5:
    """
    Класс для фиттирования данных функцией y(x) = a - b * ln(x + c)
    с использованием нелинейного метода наименьших квадратов
    """

    def __init__(self, filename: str):
        """
        Инициализация задачи.

        Parameters:
        -----------
        filename : str
            Имя файла с данными
        """
        self.filename = filename
        self.x_data = None
        self.y_data = None
        self.n = 0

        # Параметры модели
        self.a = None
        self.b = None
        self.c = None

        # Ошибки параметров
        self.a_err = None
        self.b_err = None
        self.c_err = None

        # Статистики
        self.r_squared = None
        self.adj_r_squared = None
        self.sigma_squared = None
        self.cov_matrix = None

        # Данные для статистик (сохраняются после фиттирования)
        self.J_final = None
        self.r_final = None

        self._load_data()

    def _load_data(self) -> None:
        """
        Загрузка данных из файла.
        """
        try:
            data = np.loadtxt(self.filename)
            self.x_data = data[:, 0]
            self.y_data = data[:, 1]
            self.n = len(self.x_data)
            print(f"Данные загружены: {self.n} точек")
            print(f"x ∈ [{self.x_data.min():.3f}, {self.x_data.max():.3f}]")
            print(f"y ∈ [{self.y_data.min():.3f}, {self.y_data.max():.3f}]")
        except FileNotFoundError:
            print(f"Файл {self.filename} не найден. Генерирую тестовые данные...")
            self._generate_test_data()

    def _generate_test_data(self) -> None:
        """
        Генерация тестовых данных для демонстрации.
        """
        np.random.seed(42)
        self.x_data = np.linspace(0.1, 10, 50)
        true_a, true_b, true_c = 5.0, 2.0, 0.5
        self.y_data = true_a - true_b * np.log(self.x_data + true_c)
        noise = np.random.normal(0, 0.05, len(self.x_data))
        self.y_data += noise
        self.n = len(self.x_data)
        print(f"Сгенерировано {self.n} тестовых точек")
        print(f"Истинные параметры: a={true_a}, b={true_b}, c={true_c}")

    def _model(self, x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        """
        Модельная функция: y = a - b * ln(x + c)
        """
        return a - b * np.log(x + c)

    def fit(
        self,
        initial_guess: Optional[Tuple[float, float, float]] = None,
        tol: float = 1e-10,
        max_iter: int = 100,
        Dmax: float = 0.1,
        verbose: bool = True
    ) -> None:
        """
        Выполнение фиттирования

        Parameters:
        -----------
        initial_guess : Tuple[float, float, float], optional
            Начальное приближение (a, b, c)
        tol : float
            Порог сходимости по норме шага
        max_iter : int
            Максимальное число итераций
        Dmax : float
            Максимально допустимая норма шага (демпфирование)
        verbose : bool
            Печатать ли информацию о ходе итераций
        """
        if initial_guess is None:
            a0 = np.mean(self.y_data)
            b0 = 1.0
            c0 = 0.1
            initial_guess = (a0, b0, c0)

        p = np.array(initial_guess, dtype=float)  # [a, b, c]
        N = self.n

        if verbose:
            print("\n" + "=" * 70)
            print("ВЫПОЛНЕНИЕ ФИТТИРОВАНИЯ")
            print("=" * 70)
            print(f"Начальное приближение: a={p[0]:.4f}, b={p[1]:.4f}, c={p[2]:.4f}")
            print(f"Параметры: tol={tol:.1e}, max_iter={max_iter}, Dmax={Dmax}")
            print("-" * 70)

        for iteration in range(max_iter):
            a, b, c = p
            #Заполнение матрицы Якоби J и вектора невязок r
            J = np.zeros((N, 3))
            r = np.zeros(N)

            for i in range(N):
                x = self.x_data[i]
                ln_xc = np.log(x + c)

                # Производные
                J[i, 0] = 1.0                      # df/da
                J[i, 1] = -ln_xc                   # df/db
                J[i, 2] = -b / (x + c)             # df/dc

                # Невязка: f(x) - y
                r[i] = a - b * ln_xc - self.y_data[i]

            # Нормальные уравнения
            A = J.T @ J     
            b_vec = -J.T @ r

            # Решение системы A * dp = b_vec
            try:
                dp = np.linalg.solve(A, b_vec)
            except np.linalg.LinAlgError:
                # Регуляризация при плохой обусловленности
                lambda_reg = 1e-6
                A_reg = A + lambda_reg * np.eye(3)
                dp = np.linalg.solve(A_reg, b_vec)

            norm_dp = np.linalg.norm(dp)
            ssr = np.sum(r ** 2)

            if verbose and iteration % 10 == 0:
                print(f"Итер {iteration:3d}: dp_norm = {norm_dp:.2e}, SSR = {ssr:.6f}")

            # Проверка сходимости
            if norm_dp < tol:
                if verbose:
                    print(f"✅ Сошлось за {iteration + 1} итераций")
                break

            # Демпфирование шага
            if norm_dp > Dmax:
                dp *= Dmax / norm_dp
                if verbose:
                    print(f"   Шаг демпфирован: {norm_dp:.2e} -> {Dmax:.2e}")

            # Обновление параметров
            p += dp

        else:
            if verbose:
                print(f"Достигнут максимум итераций ({max_iter})")

        # Сохраняем оптимальные параметры
        self.a, self.b, self.c = p

        # Сохраняем финальные J и r для вычисления статистик
        self._compute_final_J_and_r()

        if verbose:
            print(f"\nФинальные параметры: a={self.a:.8f}, b={self.b:.8f}, c={self.c:.8f}")
            print(f"Финальная SSR = {np.sum(self.r_final ** 2):.8f}")

        # Вычисляем статистики
        self._compute_statistics()

    def _compute_final_J_and_r(self) -> None:
        """
        Вычисление финальной матрицы Якоби и вектора невязок
        для оптимальных параметров.
        """
        N = self.n
        self.J_final = np.zeros((N, 3))
        self.r_final = np.zeros(N)

        for i in range(N):
            x = self.x_data[i]
            ln_xc = np.log(x + self.c)

            self.J_final[i, 0] = 1.0
            self.J_final[i, 1] = -ln_xc
            self.J_final[i, 2] = -self.b / (x + self.c)

            self.r_final[i] = self.a - self.b * ln_xc - self.y_data[i]

    def _compute_statistics(self) -> None:
        """
        Вычисление статистических характеристик по формулам из лекции:
        - стандартные отклонения параметров
        - скорректированный R^2
        """
        N = self.n

        # Остаточная сумма квадратов
        RSS = np.sum(self.r_final ** 2)

        # Общая сумма квадратов
        y_mean = np.mean(self.y_data)
        TSS = np.sum((self.y_data - y_mean) ** 2)

        # R^2
        self.r_squared = 1.0 - RSS / TSS

        # Скорректированный R^2
        self.adj_r_squared = 1.0 - (RSS / (N - 3)) / (TSS / (N - 1))

        # Остаточная дисперсия
        self.sigma_squared = RSS / (N - 3)

        # Ковариационная матрица: C = (J^T J)^{-1} * σ^2
        JTJ = self.J_final.T @ self.J_final
        try:
            JTJ_inv = np.linalg.inv(JTJ)
        except np.linalg.LinAlgError:
            JTJ_inv = np.linalg.pinv(JTJ)

        self.cov_matrix = JTJ_inv * self.sigma_squared

        # Стандартные отклонения параметров
        self.a_err = np.sqrt(self.cov_matrix[0, 0])
        self.b_err = np.sqrt(self.cov_matrix[1, 1])
        self.c_err = np.sqrt(self.cov_matrix[2, 2])

    def get_parameters_with_errors(self) -> dict:
        """
        Возвращает параметры с их стандартными отклонениями.
        """
        return {
            "a": (self.a, self.a_err),
            "b": (self.b, self.b_err),
            "c": (self.c, self.c_err),
            "adj_r_squared": self.adj_r_squared,
        }

    def plot_results(self, save_path: Optional[str] = None) -> None:
        """
        Построение графика с исходными данными и аппроксимирующей кривой.
        """
        x_smooth = np.linspace(
            max(0.001, self.x_data.min() - 0.1),
            self.x_data.max() + 0.1,
            200
        )
        y_smooth = self._model(x_smooth, self.a, self.b, self.c)
        y_pred = self._model(self.x_data, self.a, self.b, self.c)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Основной график
        ax1.scatter(self.x_data, self.y_data, alpha=0.7, s=30,
                   label="Данные", color="blue")
        ax1.plot(x_smooth, y_smooth, "r-", linewidth=2,
                label=f"y = {self.a:.4f} - {self.b:.4f}·ln(x + {self.c:.4f})")
        ax1.set_xlabel("x", fontsize=12)
        ax1.set_ylabel("y", fontsize=12)
        ax1.set_title("Аппроксимация данных", fontsize=14, fontweight="bold")
        ax1.legend(loc="best")
        ax1.grid(True, alpha=0.3)

        # График остатков
        residuals = self.y_data - y_pred
        ax2.scatter(self.x_data, residuals, alpha=0.7, s=30, color="green")
        ax2.axhline(y=0, color="red", linestyle="--", linewidth=1.5)
        ax2.set_xlabel("x", fontsize=12)
        ax2.set_ylabel("Остатки", fontsize=12)
        ax2.set_title("График остатков", fontsize=14, fontweight="bold")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"\nГрафик сохранен в {save_path}")

        plt.show()

    def print_summary(self) -> None:
        """
        Вывод полной сводки результатов.
        """
        print("\n" + "=" * 70)
        print("РЕЗУЛЬТАТЫ ФИТТИРОВАНИЯ")
        print("=" * 70)

        print("\nМодель: y(x) = a - b·ln(x + c)")
        print(f"Количество точек: {self.n}")
        print(f"Число параметров: 3")

        print("\n" + "-" * 70)
        print("ПАРАМЕТРЫ МОДЕЛИ:")
        print("-" * 70)
        print(f"  a = {self.a:.8f} ± {self.a_err:.8f}")
        print(f"  b = {self.b:.8f} ± {self.b_err:.8f}")
        print(f"  c = {self.c:.8f} ± {self.c_err:.8f}")

        print("\n" + "-" * 70)
        print("СТАТИСТИЧЕСКИЕ ПОКАЗАТЕЛИ:")
        print("-" * 70)
        print(f"  R² = {self.r_squared:.8f}")
        print(f"  Скорректированный R² (adjR²) = {self.adj_r_squared:.8f}")
        print(f"  Остаточная дисперсия σ² = {self.sigma_squared:.8f}")

        print("\n" + "=" * 70)


def main():
    # Создание экземпляра задачи
    task = Task5(filename="tasks/5/data5.txt")

    # Выполнение фиттирования
    task.fit(
        initial_guess=None,  # автоматическое
        tol=1e-10,
        max_iter=100,
        Dmax=0.1,
        verbose=True
    )

    # Вывод сводки результатов
    task.print_summary()

    # Получение параметров в виде словаря
    params = task.get_parameters_with_errors()

    print("\n" + "=" * 70)
    print("ИТОГОВЫЙ ОТВЕТ:")
    print("=" * 70)
    print(f"a = {params['a'][0]:.8f} ± {params['a'][1]:.8f}")
    print(f"b = {params['b'][0]:.8f} ± {params['b'][1]:.8f}")
    print(f"c = {params['c'][0]:.8f} ± {params['c'][1]:.8f}")
    print(f"adjR² = {params['adj_r_squared']:.8f}")
    print("=" * 70)

    # Построение графиков
    task.plot_results(save_path="tasks/5/task5_result.png")

    return task


if __name__ == "__main__":
    task = main()