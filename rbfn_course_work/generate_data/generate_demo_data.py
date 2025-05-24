## Генерация демонстрационных данных для RBF-сети
import numpy as np

def generate_demo_data(n_samples=100, noise=0.1, func_type='sin'):
    """
    Генерирует демонстрационные данные для RBF-сети

    Параметры:
    - n_samples: количество точек данных
    - noise: уровень шума
    - func_type: тип функции ('sin', 'exp', 'quadratic', 'complex')

    Возвращает:
    - X: входные данные
    - y: целевые значения
    """
    if func_type == 'sin':
        X = np.linspace(-5, 5, n_samples).reshape(-1, 1)
        y = np.sin(X.flatten()) + noise * np.random.randn(n_samples)
    elif func_type == 'exp':
        X = np.linspace(-2, 2, n_samples).reshape(-1, 1)
        y = np.exp(-X.flatten()**2) + noise * np.random.randn(n_samples)
    elif func_type == 'quadratic':
        # Квадратичная функция: y = ax² + bx + c
        X = np.linspace(-3, 3, n_samples).reshape(-1, 1)
        y = 0.5 * X.flatten()**2 - 1.5 * X.flatten() + 2 + noise * np.random.randn(n_samples)
    elif func_type == '2d':
        # 2D данные для регрессии
        from sklearn.datasets import make_friedman2
        X, y = make_friedman2(n_samples=n_samples, noise=noise, random_state=42)
        # Ограничиваем до первых двух признаков для наглядности
        X = X[:, :2]
    elif func_type == 'classification':
        # 2D данные для классификации
        from sklearn.datasets import make_circles
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=42)
    else:  # complex
        X = np.linspace(-5, 5, n_samples).reshape(-1, 1)
        y = np.sin(X.flatten()) * np.exp(-0.1 * X.flatten()**2) + noise * np.random.randn(n_samples)

    return X, y