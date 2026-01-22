import numpy as np
from scipy.stats import norm

# Решаем задачу множественного тестирования
# Хотим выбрать лучшую выборку из k выборок по некоторой целевой метрике
# Пусть гипотеза H_i - iя выборка лучшая, i = 1, ..., k
# Пусть гипотеза H_0 - нет лучшей выборки

# Разработали метод, основанный на предельном распределении
# Он позволяет придерживаться заложенных вероятностей ошибок 1го и 2го рода

# Функции минимального размера выборки и критерия являются основными
# Функция квантиля предельного распределения случайной величины минимума используется в основных функциях

def quantile_of_marginal_distribution(k, gamma, var=[1,1,1], equal_var=True):
    """Функция квантиля предельного распределения минимума

    Parameters
    ----------
    k : int
        Количество выборок, целое число больше 2
    gamma : float
        Уровень квантиля предельного распределения минимума
    var : list of float, optional
        Список дисперсий выборок одинаковой длины. Количество выборок больше 2
    equal_var : bool, optional
        Равенство дисперсий

    Returns
    -------
    float, если equal_var=True
        Квантиль предельного распределения минимума уровня gamma
    list of float, если equal_var=False
        Набор квантилей предельного распределения минимума для каждого j уровня gamma
    """
    iter_size = 20000                        # Количество итераций теста
    # N = 80                                 # Количество генераций квантилей, из которых возвращается максимум

    if equal_var == True:
        j = 0 # в силу симметрии j по гипотезе H_0 возьмём j = 0 (первая выборка)
        t_j = []
        total = norm.rvs(size=[iter_size, k])
        for l in range(iter_size):
            Z = total[l]
            t = np.inf
            for i in range(k):
                if i != j:
                    t_ji = (Z[j] - Z[i]) / np.sqrt(2)
                    if t_ji < t:
                        t = t_ji
            t_j += [t]
        return np.quantile(t_j, gamma)
    else:
        c_ = []
        for j in range(k):
            t_j = []
            total = norm.rvs(size=[iter_size, k])
            for l in range(iter_size):
                Z = total[l]
                t = np.inf
                for i in range(k):
                    if i != j:
                        t_ji = Z[j] / np.sqrt(1 + var[i] / var[j]) - Z[i] / np.sqrt(1 + var[j] / var[i])
                        if t_ji < t:
                            t = t_ji
                t_j += [t]
            c_ += [np.quantile(t_j, gamma)]
        return c_

def test_on_marginal_distribution(X, alpha=0.05, equal_var=True, c=None):
    """Функция критерия, основанного на предельном распределении минимума

    Parameters
    ----------
    X : list of lists
        Список выборок одинаковой длины. Количество выборок больше 2
    alpha : float, optional
        Уровень значимости, число от 0 до 1
    equal_var : bool, optional
        Равенство дисперсий
    c : optional
            float, если equal_var=True
                Квантиль предельного распределения минимума уровня 1-alpha/len(X)
            list of float, если equal_var=False
                Набор квантилей предельного распределения минимума для каждого j уровня 1-alpha/len(X)
    
    Returns
    -------
    int
        Число от 0 до k - номер принятой гипотезы
    """
    k = len(X)                                                   # Число выборок
    n = len(X[0])                                                # Размер выборки
    
    mean = []
    var = []
    for m in range(k):
        mean += [np.mean(X[m])]
        var += [np.var(X[m]) * n / (n-1)] 

    if equal_var == True:
        if c == None:
            c = quantile_of_marginal_distribution(k=k, gamma=1-alpha/k) # квантиль предельного распределения
        for j in range(k):
            t_j = np.inf
            for i in range(k):
                if i != j:
                    t_ji = np.sqrt(n) * (mean[j] - mean[i]) / np.sqrt(var[j] + var[i])
                    if t_ji < t_j:
                        t_j = t_ji
            if t_j > c:
                return j + 1
        return 0
    else:
        if c == None:
            c = quantile_of_marginal_distribution(k=k, gamma=1-alpha/k, var=var, equal_var=False) # набор квантилей предельного распределения
        for j in range(k):
            t_j = np.inf
            for i in range(k):
                if i != j:
                    t_ji = np.sqrt(n) * (mean[j] - mean[i]) / np.sqrt(var[j] + var[i])
                    if t_ji < t_j:
                        t_j = t_ji
            if t_j > c[j]:
                return j + 1
        return 0

def min_sample_size(k, d, var, alpha=0.05, beta=0.2, equal_var=True, c_1=None, c_2=None, N=None):
    """Функция для подсчёта минимального размера выборки

    Parameters
    ----------
    k : int
        Количество выборок, целое число больше 2
    d : float
        Minimum Detectable Effect, положительное число
    var : list of float, если equal_var=False
          float, если equal_var=True
        Оценка дисперсии выборок одинаковой длины при H(0). Количество выборок больше 2
    alpha : float, optional
        Уровень значимости, число от 0 до 1
    beta : float, optional
        1 - мощность, число от 0 до 1
    equal_var : bool, optional
        Равенство дисперсий
    c_1 : optional
            float, если equal_var=True
                Квантиль предельного распределения минимума уровня 1-alpha/len(X)
            list of float, если equal_var=False
                Набор квантилей предельного распределения минимума для каждого j уровня 1-alpha/len(X)
    c_2 : optional
            float, если equal_var=True
                Квантиль предельного распределения минимума уровня beta
            None, если equal_var=False
    N : int, optional
        Нижняя граница для размера (для более быстрой работы программы)

    # В нашем случае все выборки будут одного размера

    Returns
    -------
    int
        Число n - размер одной выборки
    """
    if equal_var == True:
        if c_1 == None:
            c_1 = quantile_of_marginal_distribution(k=k, gamma=1-alpha/k) # квантиль предельного распределения 1-alpha/k
        
        if c_2 == None:
            c_2 = quantile_of_marginal_distribution(k=k, gamma=beta) # квантиль предельного распределения beta
        
        return int(2 * var * ((c_1 - c_2) / d)**2) + 1
    else:
        iter_size = 3000 # Количество итераций
        if c_1 == None:
            c_1 = quantile_of_marginal_distribution(k=k, gamma=1-alpha/k, var=var, equal_var=False) # набор квантилей предельного распределения
        N_ = [] # для размеров выборки
        for j in range(k):
            if N == None:
                n = 0
            else:
                n = N
            power = 0 # мощность
            while power < 1 - beta:
                n += 100
                power = 0
                total = norm.rvs(size=[iter_size, k])
                for l in range(iter_size):
                    Z = total[l]
                    t = np.inf
                    for i in range(k):
                        if i != j:
                            t_ji = Z[j] / np.sqrt(1 + var[i] / var[j]) - Z[i] / np.sqrt(1 + var[j] / var[i]) + d * np.sqrt(n / (var[j] + var[i]))
                            if t_ji < t:
                                t = t_ji
                    if t > c_1[j]:
                        power += 1
                power = power / iter_size 
            N_ += [n]
        return np.max(N_)

# Применение метода

# Пример 1
# Рассмотрим множественный тест для выявления выборки с лучшей конверсии (распределение Бернулли)
# В этом случае при гипотезе H_0, равенстве конверсий на всех выборках, дисперсии равны, 
# поскольку дисперсия зависит только от конверсии

from scipy.stats import bernoulli

k = 10                 # число выборок
d = 0.05               # MDE
p = 0.3                # предполагаемая конверсия (параметр распределения Бернулли)
alpha = 0.05           # уровень значимости
beta = 0.2             # 1 - мощность 

# считаем минимальный размер выборки
n = min_sample_size(k=k, d=d, var=p * (1 - p), alpha=alpha, beta=beta, equal_var=True)
print(f'Размер выборки = {n}')

# Попробуем сгенерировать выборки и посмотреть результат тестирования
N = 5
# все выборки имеют одинаковую конверсию
print('\nВсе выборки имеют одинаковую конверсию')
for _ in range(N):
    X = bernoulli.rvs(p, size=[k, n])
    hyp = test_on_marginal_distribution(X, alpha=alpha)
    print(f'\tПринята гипотеза H({hyp})')

# десятая выборка имеет большую на MDE конверсию
print('\nДесятая выборка имеет большую на MDE конверсию')
for _ in range(N):
    X = []
    for i in range(k-1):
        X += [bernoulli.rvs(p, size=n)]
    X += [bernoulli.rvs(p + d, size=n)]
    hyp = test_on_marginal_distribution(X, alpha=alpha)
    print(f'\tПринята гипотеза H({hyp})')


# Пример 2
# Рассмотрим множественный тест для выявления выборки с лучшим доходом на клиента (конверсия * цена)
# В этом случае при гипотезе H_0, равенстве ARPU на всех выборках, дисперсии не равны, 
# поскольку при гипотезе H_0 при разных ценах получаем разные конверсии

k = 5                                     # число выборок
d = 2.5                                   # MDE
# в среднем ARPU = 15 рублей
Price = [100, 150, 150, 200, 250]         # цены тарифов
conv = [0.15, 0.1, 0.1, 0.075, 0.06]      # конверсии тарифов
alpha = 0.05                              # уровень значимости
beta = 0.2                                # 1 - мощность 
var = []                                  # дисперсии тарифов
for i in range(k):
    var += [Price[i]**2 * conv[i] * (1 - conv[i])]

# считаем минимальный размер выборки
n = min_sample_size(k=k, d=d, var=var, alpha=alpha, beta=beta, equal_var=False)
print(f'Размер выборки = {n}')

# Попробуем сгенерировать выборки и посмотреть результат тестирования
N = 5
# все выборки имеют одинаковый ARPU
print('\nВсе выборки имеют одинаковый ARPU')
for _ in range(N):
    X = []
    for i in range(k):
        X += [Price[i] * bernoulli.rvs(conv[i], size=n)]
    hyp = test_on_marginal_distribution(X, alpha=alpha)
    print(f'\tПринята гипотеза H({hyp})')

# десятая выборка имеет больший на MDE ARPU
print('\nДесятая выборка имеет больший на MDE ARPU')
for _ in range(N):
    X = []
    for i in range(k-1):
        X += [Price[i] * bernoulli.rvs(conv[i], size=n)]
    X += [Price[k-1] * bernoulli.rvs(conv[k-1] + d / Price[k-1], size=n)]
    hyp = test_on_marginal_distribution(X, alpha=alpha)
    print(f'\tПринята гипотеза H({hyp})') 
