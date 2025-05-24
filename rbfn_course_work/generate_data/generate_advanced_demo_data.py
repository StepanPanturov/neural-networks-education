"""## Генерация демонстрационных данных для RBF-сети (advanced)"""
import numpy as np

def generate_advanced_demo_data(n_samples=200, noise=0.1, func_type='multimodal'):
    """
    Генерирует более сложные демонстрационные данные для RBF-сети

    Параметры:
    - n_samples: количество точек данных
    - noise: уровень шума
    - func_type: тип функции

    Типы функций:
    - 'multimodal': многомодальная функция
    - 'discontinuous': разрывная функция
    - 'multidimensional': многомерная функция (3D)
    - 'noisy_peaks': функция с шумными всплесками
    - 'complex_3d': сложная 3D поверхность
    """
    if func_type == 'multimodal':
        X = np.linspace(-8, 8, n_samples).reshape(-1, 1)
        y = np.sin(X.flatten()) + 0.5 * np.sin(3 * X.flatten()) + noise * np.random.randn(n_samples)

    elif func_type == 'discontinuous':
        X = np.linspace(-5, 5, n_samples).reshape(-1, 1)
        y = np.zeros(n_samples)
        mask1 = X.flatten() < -2
        mask2 = (X.flatten() >= -2) & (X.flatten() < 1)
        mask3 = X.flatten() >= 1
        y[mask1] = -1 + noise * np.random.randn(np.sum(mask1))
        y[mask2] = np.sin(X.flatten()[mask2] * 2) + noise * np.random.randn(np.sum(mask2))
        y[mask3] = 1 + 0.5 * np.sin(X.flatten()[mask3]) + noise * np.random.randn(np.sum(mask3))

    elif func_type == 'multidimensional':
        # 3D данные для регрессии
        from sklearn.datasets import make_friedman3
        X, y = make_friedman3(n_samples=n_samples, noise=noise, random_state=42)
        # Ограничиваем до первых трех признаков для наглядности
        X = X[:, :3]

    elif func_type == 'noisy_peaks':
        X = np.linspace(-5, 5, n_samples).reshape(-1, 1)
        base = np.sin(X.flatten()) * np.exp(-0.1 * X.flatten()**2)
        # Добавляем случайные всплески
        peaks = np.zeros(n_samples)
        for _ in range(10):  # 10 случайных всплесков
            pos = np.random.randint(0, n_samples)
            width = np.random.randint(5, 20)
            height = np.random.uniform(1, 3)
            # Ограничиваем диапазон индексов
            start = max(0, pos - width // 2)
            end = min(n_samples, pos + width // 2)
            peaks[start:end] = height * np.exp(-0.5 * ((np.arange(start, end) - pos) / (width / 5))**2)
        y = base + peaks + noise * np.random.randn(n_samples)

    elif func_type == 'complex_3d':
        # Создаем сетку точек в 2D пространстве
        x = np.linspace(-3, 3, int(np.sqrt(n_samples)))
        y = np.linspace(-3, 3, int(np.sqrt(n_samples)))
        xx, yy = np.meshgrid(x, y)
        X = np.column_stack([xx.ravel(), yy.ravel()])

        # Сложная функция от двух переменных
        z = np.sin(np.sqrt(xx**2 + yy**2)) + 0.1 * xx * yy + np.exp(-0.1 * (xx**2 + yy**2)) * np.cos(xx * yy)
        z = z.ravel() + noise * np.random.randn(X.shape[0])

        return X, z

    else:  # default - return simple noisy function
        X = np.linspace(-5, 5, n_samples).reshape(-1, 1)
        y = 0.5 * X.flatten()**2 + np.sin(X.flatten() * 3) + noise * np.random.randn(n_samples)

    return X, y