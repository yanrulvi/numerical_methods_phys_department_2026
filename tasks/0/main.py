import matplotlib.pyplot as plt

def f(x):
    return 1 / (x * x)

I_exact = 0.5

k_values = list(range(2, 9))
x_values = []   # сюда будем класть -k
errors = []

a = 1.0
b = 2.0

for k in k_values:
    N = 10 ** k
    h = (b - a) / N

    S = f(a) + f(b)

    sum_odd = 0.0
    sum_even = 0.0

    for i in range(1, N):
        x = a + i * h
        if i % 2 == 0:
            sum_even += f(x)
        else:
            sum_odd += f(x)

    I_N = (h / 3) * (S + 4 * sum_odd + 2 * sum_even)

    error = abs(I_N - I_exact)

    errors.append(error)
    x_values.append(-k)

    print(f"k = {k}, N = {N}, I_N = {I_N}, error = {error}")

# график зависимости от -k
plt.plot(x_values, errors, marker='o')
plt.yscale('log')
plt.xlabel("-k (N = 10^k)")
plt.ylabel("|I_N - 0.5|")
plt.title("Ошибка метода Симпсона от -k")
plt.grid(True)
plt.show()