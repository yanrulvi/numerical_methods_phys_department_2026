# type: ignore
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import os


class Task4:
    """
    Экстраполяция значения функции в точке x=11 методом Эйткена
    с использованием M самых правых точек из набора данных.
    """

    def __init__(
        self,
        filename: str,
        x_target: float,
        true_value: Optional[float],
    ):
        """
        Инициализация задачи.

        Parameters:
        -----------
        filename : str
            Имя файла с данными
        x_target : float
            Точка экстраполяции
        true_value : float, optional
            Точное значение функции в точке x_target
        """
        self.filename = filename
        self.x_target = x_target
        self.true_value = true_value
        self.x_nodes = []
        self.y_nodes = []
        self._load_data()

    def _load_data(self) -> None:
        """
        Загрузка данных из файла.
        """
        try:
            data = np.loadtxt(self.filename)

            self.x_nodes = data[:, 0]
            self.y_nodes = data[:, 1]
            print(
                f"Данные загружены: {len(self.x_nodes)} точек, x ∈ [{self.x_nodes[0]}, {self.x_nodes[-1]}]"
            )
        except FileNotFoundError:
            print(f"Файл {self.filename} не найден")
            self._generate_test_data()

    def _generate_test_data(self) -> None:
        """
        Генерация тестовых данных для демонстрации работы,
        если файл не найден.
        """
        self.x_nodes = np.arange(1.0, 11.0, 1.0)
        # Тестовая функция: f(x) = sin(x)/x + 0.2*cos(2x)
        self.y_nodes = np.sin(self.x_nodes) / self.x_nodes + 0.2 * np.cos(
            2 * self.x_nodes
        )
        # Пересчитываем точное значение для тестовой функции
        self.true_value = np.sin(self.x_target) / self.x_target + 0.2 * np.cos(
            2 * self.x_target
        )
        print(f"Сгенерировано {len(self.x_nodes)} тестовых точек")

    def _aitken_extrapolation(
        self, x_nodes: np.ndarray, y_nodes: np.ndarray
    ) -> float:
        """
        Вычисление значения интерполяционного полинома в точке x_target
        методом Эйткена.

        Parameters:
        -----------
        x_nodes : np.ndarray
            Узлы интерполяции
        y_nodes : np.ndarray
            Значения функции в узлах

        Returns:
        --------
        float
            Значение полинома в точке x_target
        """
        n = len(x_nodes)
        # Начальный уровень: P_i = y_i
        current_level = y_nodes.copy()

        # Построение треугольной таблицы Эйткена
        for level in range(1, n):
            next_level = []
            for i in range(n - level):
                xi = x_nodes[i]
                xj = x_nodes[i + level]
                # Формула Эйткена:
                # P_{i,i+level} = [(xj - x)*P_i + (x - xi)*P_{i+1}] / (xj - xi)
                numerator = (xj - self.x_target) * current_level[i] + (
                    self.x_target - xi
                ) * current_level[i + 1]
                denominator = xj - xi
                next_level.append(numerator / denominator)
            current_level = next_level

        return current_level[0]

    def compute_extrapolation(self, M: int) -> Tuple[float, float]:
        """
        Вычисление экстраполяции для заданного M.

        Parameters:
        -----------
        M : int
            Количество самых правых точек для интерполяции

        Returns:
        --------
        Tuple[float, float]
            (экстраполированное значение, ошибка)
        """
        if M < 2:
            raise ValueError("M должно быть не менее 2")
        if M > len(self.x_nodes):
            print(
                f"Предупреждение: M={M} больше числа точек ({len(self.x_nodes)}), "
                f"использую все точки"
            )
            M = len(self.x_nodes)

        # Берем M самых правых точек
        x_subset = self.x_nodes[-M:]
        y_subset = self.y_nodes[-M:]

        # Экстраполяция
        extrapolated_value = self._aitken_extrapolation(x_subset, y_subset)

        # Ошибка
        error = abs(extrapolated_value - self.true_value)

        return extrapolated_value, error

    def analyze(
        self, M_min: int = 4, M_max: Optional[int] = None
    ) -> Tuple[List[int], List[float]]:
        """
        Анализ зависимости ошибки от M.

        Parameters:
        -----------
        M_min : int
            Минимальное значение M
        M_max : Optional[int]
            Максимальное значение M (по умолчанию - количество точек)

        Returns:
        --------
        Tuple[List[int], List[float]]
            (список M, список ошибок)
        """
        if M_max is None:
            M_max = len(self.x_nodes)

        M_values = list(range(M_min, M_max + 1))
        errors = []

        print("\n" + "=" * 70)
        print(
            f"{'M':>4} | {'Экстраполированное значение':>25} | {'Ошибка':>12}"
        )
        print("=" * 70)

        for M in M_values:
            try:
                extrap_val, error = self.compute_extrapolation(M)
                errors.append(error)
                print(f"{M:4d} | {extrap_val:25.15f} | {error:12.4e}")
            except Exception as e:
                print(f"{M:4d} | {'Ошибка':>25} | {str(e):12}")
                errors.append(np.nan)

        print("=" * 70)
        return M_values, errors

    def plot_results(
        self,
        M_values: List[int],
        errors: List[float],
        save_path: Optional[str] = None,
    ) -> None:
        """
        Построение графика зависимости ошибки от M.

        Parameters:
        -----------
        M_values : List[int]
            Список значений M
        errors : List[float]
            Список соответствующих ошибок
        save_path : Optional[str]
            Путь для сохранения графика (если указан)
        """
        # Убираем NaN значения
        valid_idx = ~np.isnan(errors)
        M_valid = np.array(M_values)[valid_idx]
        errors_valid = np.array(errors)[valid_idx]

        plt.figure(figsize=(12, 7))

        # Основной график (логарифмический масштаб по Y)
        plt.semilogy(
            M_valid,
            errors_valid,
            "bo-",
            linewidth=2,
            markersize=6,
            label="|Ошибка экстраполяции|",
        )

        # Добавляем линию минимальной ошибки
        min_error_idx = np.argmin(errors_valid)
        min_M = M_valid[min_error_idx]
        min_error = errors_valid[min_error_idx]
        plt.axvline(
            x=min_M,
            color="r",
            linestyle="--",
            alpha=0.7,
            label=f"Минимум при M={min_M}",
        )

        # Оформление
        plt.xlabel(
            "M (количество правых точек)", fontsize=14, fontweight="bold"
        )
        plt.ylabel(
            "|f_extrapolated(11) - f_true(11)|", fontsize=14, fontweight="bold"
        )
        plt.title(
            "Зависимость ошибки экстраполяции от количества точек M\n"
            f"Метод Эйткена, x_target = {self.x_target}",
            fontsize=16,
            fontweight="bold",
        )
        plt.grid(True, which="both", linestyle="--", alpha=0.6)
        plt.legend(loc="best", fontsize=12)

        # Добавляем аннотацию с минимальной ошибкой
        plt.annotate(
            f"Минимальная ошибка: {min_error:.2e}\nпри M = {min_M}",
            xy=(min_M, min_error),
            xytext=(min_M + 2, min_error * 2),
            arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"\nГрафик сохранен в {save_path}")

        plt.show()

    def print_summary(self, M_values: List[int], errors: List[float]) -> None:
        """
        Вывод статистической сводки по результатам.

        Parameters:
        -----------
        M_values : List[int]
            Список значений M
        errors : List[float]
            Список соответствующих ошибок
        """
        valid_idx = ~np.isnan(errors)
        M_valid = np.array(M_values)[valid_idx]
        errors_valid = np.array(errors)[valid_idx]

        if len(errors_valid) == 0:
            print("Нет валидных данных для анализа")
            return

        min_error_idx = np.argmin(errors_valid)
        max_error_idx = np.argmax(errors_valid)

        print("\n" + "=" * 70)
        print("СВОДКА")
        print("=" * 70)
        print(f"Точка экстраполяции: x = {self.x_target}")
        print(f"Точное значение f({self.x_target}) = {self.true_value:.15f}")
        print(f"Количество доступных точек: {len(self.x_nodes)}")
        print(f"Диапазон M: [{M_valid[0]}, {M_valid[-1]}]")
        print("-" * 70)
        print(
            f"Минимальная ошибка: {errors_valid[min_error_idx]:.4e} при M = {M_valid[min_error_idx]}"
        )
        print(
            f"Максимальная ошибка: {errors_valid[max_error_idx]:.4e} при M = {M_valid[max_error_idx]}"
        )
        print("=" * 70)


def main():
    """
    Основная функция для запуска решения задачи.
    """
    # Создание экземпляра задачи
    print(os.getcwd())
    task = Task4(
        filename="tasks/4/data4.txt",
        x_target=11.0,
        true_value=0.99591322856153597,
    )

    # Анализ зависимости ошибки от M
    M_min = 4
    M_max = 30
    M_values, errors = task.analyze(M_min=M_min, M_max=M_max)

    # Построение графика
    task.plot_results(M_values, errors, save_path="tasks/4/task4_result.png")

    # Вывод статистической сводки
    task.print_summary(M_values, errors)


if __name__ == "__main__":
    main()
