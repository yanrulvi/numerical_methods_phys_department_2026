import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import t
from typing import Tuple, Optional
import warnings


class Task5:
    """
    Класс для фиттирования данных функцией y(x) = a - b * ln(x + c)
    с использованием метода наименьших квадратов.
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
        self.n = 0  # количество точек

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
        self.sigma_squared = None  # остаточная дисперсия
        self.cov_matrix = None  # ковариационная матрица

        self._load_data()

    def _load_data(self) -> None:
        """
        Загрузка данных из файла.
        Предполагается формат: два столбца (x, y)
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
            print(
                f"Файл {self.filename} не найден. Генерирую тестовые данные..."
            )
            self._generate_test_data()

    def _generate_test_data(self) -> None:
        """
        Генерация тестовых данных для демонстрации работы.
        """
        np.random.seed(42)
        self.x_data = np.linspace(0.1, 10, 50)
        true_a, true_b, true_c = 5.0, 2.0, 0.5
        self.y_data = true_a - true_b * np.log(self.x_data + true_c)
        # Добавляем шум
        noise = np.random.normal(0, 0.05, len(self.x_data))
        self.y_data += noise
        self.n = len(self.x_data)
        print(f"Сгенерировано {self.n} тестовых точек")
        print(f"Истинные параметры: a={true_a}, b={true_b}, c={true_c}")

    def _model(
        self, x: np.ndarray, a: float, b: float, c: float
    ) -> np.ndarray:
        """
        Модельная функция: y = a - b * ln(x + c)

        Parameters:
        -----------
        x : np.ndarray
            Аргумент
        a, b, c : float
            Параметры модели

        Returns:
        --------
        np.ndarray
            Значения функции
        """
        return a - b * np.log(x + c)

    def _residuals(
        self, params: np.ndarray, x: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """
        Вычисление невязок.

        Parameters:
        -----------
        params : np.ndarray
            Параметры [a, b, c]
        x, y : np.ndarray
            Данные

        Returns:
        --------
        np.ndarray
            Невязки
        """
        a, b, c = params
        return y - self._model(x, a, b, c)

    def _objective(
        self, params: np.ndarray, x: np.ndarray, y: np.ndarray
    ) -> float:
        """
        Целевая функция: сумма квадратов невязок.

        Parameters:
        -----------
        params : np.ndarray
            Параметры [a, b, c]
        x, y : np.ndarray
            Данные

        Returns:
        --------
        float
            Сумма квадратов невязок
        """
        residuals = self._residuals(params, x, y)
        return np.sum(residuals**2)

    def fit(
        self, initial_guess: Optional[Tuple[float, float, float]] = None
    ) -> None:
        """
        Выполнение фиттирования методом наименьших квадратов.

        Parameters:
        -----------
        initial_guess : Tuple[float, float, float], optional
            Начальное приближение для параметров (a, b, c)
        """
        if initial_guess is None:
            # Эвристическое начальное приближение
            a0 = np.mean(self.y_data)
            b0 = 1.0
            c0 = 0.1
            initial_guess = (a0, b0, c0)

        print("\n" + "=" * 70)
        print("ВЫПОЛНЕНИЕ ФИТТИРОВАНИЯ")
        print("=" * 70)
        print(
            f"Начальное приближение: a={initial_guess[0]:.4f}, "
            f"b={initial_guess[1]:.4f}, c={initial_guess[2]:.4f}"
        )

        # Минимизация суммы квадратов невязок
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                self._objective,
                initial_guess,
                args=(self.x_data, self.y_data),
                method="L-BFGS-B",
                bounds=[(None, None), (None, None), (0.001, None)],
            )

        if not result.success:
            print(f"Предупреждение: оптимизация не сошлась: {result.message}")

        self.a, self.b, self.c = result.x

        print("\nОптимальные параметры:")
        print(f"  a = {self.a:.8f}")
        print(f"  b = {self.b:.8f}")
        print(f"  c = {self.c:.8f}")
        print(f"Целевая функция (SSR) = {result.fun:.6f}")

        # Вычисление статистик
        self._compute_statistics()

    def _compute_statistics(self) -> None:
        """
        Вычисление статистических характеристик:
        - стандартные отклонения параметров
        - R² и скорректированный R²
        - ковариационная матрица
        """
        # Вычисляем предсказанные значения
        y_pred = self._model(self.x_data, self.a, self.b, self.c)

        # Остатки
        residuals = self.y_data - y_pred
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((self.y_data - np.mean(self.y_data)) ** 2)

        # R²
        self.r_squared = 1 - ss_res / ss_tot

        # Скорректированный R²
        self.adj_r_squared = 1 - (1 - self.r_squared) * (self.n - 1) / (
            self.n - 3 - 1
        )

        # Остаточная дисперсия
        self.sigma_squared = ss_res / (self.n - 3)

        # Вычисление ковариационной матрицы через численное дифференцирование
        J = np.zeros((self.n, 3))  # матрица Якоби

        for i in range(self.n):
            x = self.x_data[i]
            # Производные по параметрам
            J[i, 0] = 1  # da
            J[i, 1] = -np.log(x + self.c)  # db
            J[i, 2] = -self.b / (x + self.c)  # dc

        # Ковариационная матрица: cov = σ² * (J^T J)^(-1)
        try:
            JTJ = J.T @ J
            self.cov_matrix = self.sigma_squared * np.linalg.inv(JTJ)

            # Стандартные отклонения параметров
            self.a_err = np.sqrt(self.cov_matrix[0, 0])
            self.b_err = np.sqrt(self.cov_matrix[1, 1])
            self.c_err = np.sqrt(self.cov_matrix[2, 2])
        except np.linalg.LinAlgError:
            print(
                "Предупреждение: не удалось вычислить ковариационную матрицу"
            )
            self.a_err = self.b_err = self.c_err = np.nan
            self.cov_matrix = np.full((3, 3), np.nan)

    def get_parameters_with_errors(self) -> dict:
        """
        Возвращает параметры с их стандартными отклонениями.

        Returns:
        --------
        dict
            Словарь с параметрами и ошибками
        """
        return {
            "a": (self.a, self.a_err),
            "b": (self.b, self.b_err),
            "c": (self.c, self.c_err),
            "r_squared": self.r_squared,
            "adj_r_squared": self.adj_r_squared,
            "sigma_squared": self.sigma_squared,
        }

    def confidence_intervals(self, alpha: float = 0.05) -> dict:
        """
        Вычисление доверительных интервалов для параметров.

        Parameters:
        -----------
        alpha : float
            Уровень значимости (по умолчанию 0.05 для 95% ДИ)

        Returns:
        --------
        dict
            Доверительные интервалы для параметров
        """
        dof = self.n - 3  # степени свободы
        t_critical = t.ppf(1 - alpha / 2, dof)

        intervals = {}
        params = ["a", "b", "c"]
        errors = [self.a_err, self.b_err, self.c_err]
        values = [self.a, self.b, self.c]

        for name, val, err in zip(params, values, errors):
            margin = t_critical * err
            intervals[name] = (val - margin, val + margin)

        return intervals

    def plot_results(self, save_path: Optional[str] = None) -> None:
        """
        Построение графика с исходными данными и аппроксимирующей кривой.

        Parameters:
        -----------
        save_path : Optional[str]
            Путь для сохранения графика
        """
        # Создаем гладкую кривую для отображения модели
        x_smooth = np.linspace(
            self.x_data.min() - 0.1, self.x_data.max() + 0.1, 200
        )
        y_smooth = self._model(x_smooth, self.a, self.b, self.c)

        # Предсказанные значения для исходных точек
        y_pred = self._model(self.x_data, self.a, self.b, self.c)

        plt.figure(figsize=(12, 8))

        # Основной график
        plt.subplot(2, 1, 1)
        plt.scatter(
            self.x_data,
            self.y_data,
            alpha=0.7,
            s=30,
            label="Экспериментальные данные",
            color="blue",
        )
        plt.plot(
            x_smooth,
            y_smooth,
            "r-",
            linewidth=2,
            label=f"Аппроксимация: y = {self.a:.4f} - {self.b:.4f}·ln(x + {self.c:.4f})",
        )
        plt.xlabel("x", fontsize=12)
        plt.ylabel("y", fontsize=12)
        plt.title(
            "Аппроксимация данных функцией y(x) = a - b·ln(x + c)",
            fontsize=14,
            fontweight="bold",
        )
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)

        # График остатков
        plt.subplot(2, 1, 2)
        residuals = self.y_data - y_pred
        plt.scatter(self.x_data, residuals, alpha=0.7, s=30, color="green")
        plt.axhline(y=0, color="red", linestyle="--", linewidth=1.5)
        plt.xlabel("x", fontsize=12)
        plt.ylabel("Остатки (y - y_pred)", fontsize=12)
        plt.title("График остатков", fontsize=14, fontweight="bold")
        plt.grid(True, alpha=0.3)

        # Добавляем информацию об остатках
        residual_stats = f"Ср. остаток: {np.mean(residuals):.4e}\n"
        residual_stats += f"Стд. остаток: {np.std(residuals):.4e}"
        plt.text(
            0.05,
            0.95,
            residual_stats,
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

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
        print("Число параметров: 3")
        print(f"Степени свободы: {self.n - 3}")

        print("\n" + "-" * 70)
        print("ПАРАМЕТРЫ МОДЕЛИ:")
        print("-" * 70)
        print(
            f"  a = {self.a:.8f} ± {self.a_err:.8f}  (отн. погр.: {self.a_err / abs(self.a) * 100:.2f}%)"
        )
        print(
            f"  b = {self.b:.8f} ± {self.b_err:.8f}  (отн. погр.: {self.b_err / abs(self.b) * 100:.2f}%)"
        )
        print(
            f"  c = {self.c:.8f} ± {self.c_err:.8f}  (отн. погр.: {self.c_err / abs(self.c) * 100:.2f}%)"
        )

        # Доверительные интервалы (95%)
        ci = self.confidence_intervals(alpha=0.05)
        print("\n" + "-" * 70)
        print("95% ДОВЕРИТЕЛЬНЫЕ ИНТЕРВАЛЫ:")
        print("-" * 70)
        print(f"  a ∈ [{ci['a'][0]:.8f}, {ci['a'][1]:.8f}]")
        print(f"  b ∈ [{ci['b'][0]:.8f}, {ci['b'][1]:.8f}]")
        print(f"  c ∈ [{ci['c'][0]:.8f}, {ci['c'][1]:.8f}]")

        print("\n" + "-" * 70)
        print("СТАТИСТИЧЕСКИЕ ПОКАЗАТЕЛИ:")
        print("-" * 70)
        print(f"  R² = {self.r_squared:.8f}")
        print(f"  Скорректированный R² (adjR²) = {self.adj_r_squared:.8f}")
        print(f"  Остаточная дисперсия σ² = {self.sigma_squared:.8f}")
        print(
            f"  Стандартное отклонение остатков = {np.sqrt(self.sigma_squared):.8f}"
        )

        # Корреляционная матрица параметров
        if self.cov_matrix is not None and not np.any(
            np.isnan(self.cov_matrix)
        ):
            corr_matrix = np.zeros((3, 3))
            for i in range(3):
                for j in range(3):
                    corr_matrix[i, j] = self.cov_matrix[i, j] / np.sqrt(
                        self.cov_matrix[i, i] * self.cov_matrix[j, j]
                    )

            print("\n" + "-" * 70)
            print("КОРРЕЛЯЦИОННАЯ МАТРИЦА ПАРАМЕТРОВ:")
            print("-" * 70)
            print("       a       b       c")
            for i, name in enumerate(["a", "b", "c"]):
                print(
                    f"{name}: [{corr_matrix[i, 0]:7.4f} {corr_matrix[i, 1]:7.4f} {corr_matrix[i, 2]:7.4f}]"
                )

        print("\n" + "=" * 70)

        # Оценка качества фитта
        print("\nОЦЕНКА КАЧЕСТВА ФИТТИРОВАНИЯ:")
        print("-" * 70)
        if self.adj_r_squared > 0.95:
            quality = "Отлично"
        elif self.adj_r_squared > 0.9:
            quality = "Хорошо"
        elif self.adj_r_squared > 0.8:
            quality = "Удовлетворительно"
        else:
            quality = "Требуется улучшение"

        print(f"  adjR² = {self.adj_r_squared:.4f} → {quality}")

        if self.adj_r_squared < 0.8:
            print("  Возможные проблемы:")
            print("    - Неправильно выбрана модель")
            print("    - Данные содержат выбросы")
            print("    - Нелинейность не описывается данной функцией")


def main():
    """
    Основная функция для запуска решения задачи.
    """
    # Создание экземпляра задачи
    task = Task5(filename="tasks/5/data5.txt")

    # Выполнение фиттирования
    # Можно задать начальное приближение, если нужно
    # initial_guess = (5.0, 2.0, 0.5)
    task.fit()  # или task.fit(initial_guess)

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
