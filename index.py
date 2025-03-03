import numpy as np
import pandas as pd
from scipy.optimize import linprog

# Вхідні дані: коефіцієнти цільової функції
c = np.array([-2, -1, 1, 1])  # Максимізація, тому беремо -C

# Коефіцієнти обмежень (ліва частина)
A_eq = np.array([[2, -1, 3, 4]])  # Рівняння
b_eq = np.array([10])  # Вільні члени рівняння

A_ub = np.array([
    [1, 1, 1, -1],  # Нерівність (≤)
    [1, 2, 2, 4]    # Нерівність (≥)
])
b_ub = np.array([5, 12])  # Вільні члени нерівностейa

# Всі змінні невід’ємні
x_bounds = [(0, None) for _ in range(len(c))]

# Формуємо вихідну симплекс-таблицю
A_mod = np.hstack([
    np.vstack([A_eq, A_ub]), 
    np.eye(3)  # Додаємо штучні змінні (s1, s2 для нерівностей + a1 для рівняння)
])
c_mod = np.hstack([c, [0, 0, 0]])  # Розширена функція мети

# Формуємо таблицю
table = np.hstack([A_mod, np.concatenate([b_eq, b_ub]).reshape(-1, 1)])
z_row = np.hstack([c_mod, [0]])  # Останній елемент – значення функції мети
table = np.vstack([table, z_row])

# Оформлення у вигляді таблиці
columns = ["x1", "x2", "x3", "x4", "s1", "s2", "a1", "Вільний член"]
index = ["a1", "s1", "s2", "Z"]
df = pd.DataFrame(table, columns=columns, index=index)

# Виводимо вихідну симплекс-таблицю
print("Вихідна симплекс-таблиця:")
print(df)

# Розв'язок задачі методом симплексу
res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=x_bounds, method="highs")

# Вивід результатів
print("\nОптимальне значення змінних:", res.x)
print("Оптимальне значення цільової функції:", -res.fun)  # Повертаємо знак, бо лінпрог мінімізує
