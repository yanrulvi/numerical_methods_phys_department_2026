import math
import matplotlib.pyplot as plt


def task_1_function(x, N=10):
    """Вычисление f(x) = sum_{k=1}^{N} cos(kx)/k^2 * (x - 0.01k)^k"""
    res = 0
    for k in range(1, N + 1):
        term = math.cos(k * x) / (k * k) * ((x - 0.01 * k) ** k)
        res += term

    return res


class Task1:
    """
    На отрезке [−2.5;2] при помощи метода бисекции вычислите все корни 𝑓(𝑥)
    с запрошенной точностью 10^(−14).
    Запрограммируйте табулирование функции, т.е. проход отрезка слева направо
    не слишком большим шагом с целью изоляции каждого отдельного корня.
    """

    def __init__(
        self,
        a=-2.5,
        b=2.0,
        step=0.02,
        f=task_1_function,
        tol=1e-14,
        max_iter=200,
    ):
        """
        Инициализация параметров задания №1

        :param a(float): Начала рассматриваемого отрезка
        :param b(float): Конец рассматриемого отрезка
        :param step(float): Шаг табуляции
        :param f(function): Функция f(x)
        """
        self.a = a
        self.b = b
        self.step = step
        self.fun = f
        self.tol = tol
        self.max_iter = max_iter

        self.roots = None

    def bisection(self, a, b):
        """Метод бисекции для поиска корня на [a, b]"""
        f = self.fun
        fa = f(a)
        fb = f(b)

        if fa * fb > 0:
            raise ValueError("На интервале нет корня или чётное число корней")

        for _ in range(self.max_iter):
            c = (a + b) / 2
            fc = f(c)

            if abs(fc) < self.tol or (b - a) / 2 < self.tol:
                return c

            if fa * fc < 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc
        return (a + b) / 2

    def find_roots(self):
        """Функция поиска корней на интервале с табуляцией"""
        f = self.fun

        roots = []

        if abs(self.fun(self.a)) < self.tol:
            roots.append(self.a)
        if abs(self.fun(self.b)) < self.tol:
            roots.append(self.b)

        x_left = self.a
        f_left = f(x_left)

        x_right = self.a + self.step
        while x_right <= self.b:
            f_right = f(x_right)

            if f_left * f_right < 0:
                try:
                    root = self.bisection(x_left, x_right)
                    if not roots or abs(root - roots[-1]) > self.step * 0.8:
                        roots.append(root)
                except ValueError:
                    pass

            x_left = x_right
            f_left = f_right
            x_right += self.step

        self.roots = roots
        return roots

    def plot_function(self, show_roots=True, save_path=None, figsize=(12, 6)):
        """
        Построение графика функции на отрезке [a, b] с шагом self.step.

        :param show_roots(bool): Отображать ли найденные корни на графике
        :param save_path(str): Путь для сохранения графика (если None — не сохранять)
        :param figsize(tuple): Размер фигуры (ширина, высота)
        """
        # Генерируем точки для графика
        x_values = []
        y_values = []

        x = self.a
        while x <= self.b + self.step / 2:
            x_values.append(x)
            y_values.append(self.fun(x))
            x += self.step

        # Создаём график
        plt.figure(figsize=figsize)

        # Рисуем функцию
        plt.plot(x_values, y_values, "b-", linewidth=1.5, label="f(x)")

        # Линия нуля
        plt.axhline(y=0, color="k", linestyle="-", linewidth=0.8, alpha=0.7)

        # Вертикальная линия в x=0
        plt.axvline(
            x=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5
        )

        if show_roots:
            if self.roots is None:
                self.find_roots()
            y_roots = [self.fun(r) for r in self.roots]
            plt.scatter(
                self.roots,
                y_roots,
                color="red",
                s=50,
                zorder=5,
                marker="o",
                label=f"Корни ({len(self.roots)} шт.)",
            )

            # Добавляем подписи значений корней
            for root in self.roots:
                plt.text(
                    root,
                    self.fun(root) + 0.05 * (max(y_values) - min(y_values)),
                    f"{root:.4f}",
                    fontsize=8,
                    ha="center",
                )

        # Настройка оформления
        plt.xlabel("x", fontsize=12)
        plt.ylabel("f(x)", fontsize=12)
        plt.title(
            f"График функции f(x) на отрезке [{self.a}, {self.b}]", fontsize=14
        )
        plt.grid(True, alpha=0.3, linestyle="--")
        plt.legend(loc="best")

        # Добавляем информацию о количестве корней
        if show_roots:
            plt.text(
                0.02,
                0.98,
                f"Найдено корней: {len(self.roots)}",
                transform=plt.gca().transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"График сохранён в: {save_path}")

        plt.show()

    def print_roots(self):
        """Выводит найденные корни в консоль"""
        if self.roots is None:
            self.find_roots()

        if self.roots is None:
            print("Корни не найдены.")
            return

        print(f"Найдено корней: {len(self.roots)}")
        print("-" * 50)
        for i, r in enumerate(self.roots, 1):
            print(f"Корень {i:2d}: x = {r:.15f}, f(x) = {self.fun(r):.2e}")
        print("-" * 50)


if __name__ == "__main__":
    solver = Task1(step=0.005)
    solver.print_roots()
    solver.plot_function()
