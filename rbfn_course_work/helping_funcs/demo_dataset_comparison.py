"""### Демонстрация сравнения результатов на разных наборах данных"""
from generate_data.generate_demo_data import generate_demo_data
import matplotlib.pyplot as plt
import numpy as np
from rbfn_class import RBFNetwork

def demo_dataset_comparison():
    """
    Демонстрация сравнения результатов на разных наборах данных
    """
    print("Демонстрация сравнения результатов на разных наборах данных")

    # Генерируем разные наборы данных
    datasets = {
        'sin': generate_demo_data(n_samples=200, noise=0.1, func_type='sin'),
        'exp': generate_demo_data(n_samples=200, noise=0.1, func_type='exp'),
        'complex': generate_demo_data(n_samples=200, noise=0.1, func_type='complex')
    }

    # Параметры модели
    n_centers = 7
    sigma = 0.5

    # Создаем подграфики
    fig, axes = plt.subplots(len(datasets), 1, figsize=(12, 4 * len(datasets)))

    for i, (name, (X, y)) in enumerate(datasets.items()):
        # Разделяем на обучающую и тестовую выборки
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Создаем и обучаем модель
        model = RBFNetwork(n_centers=n_centers, sigma=sigma)
        model.fit(X_train, y_train)

        # Предсказания
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Вычисляем ошибки
        train_mse = np.mean((y_train_pred - y_train) ** 2)
        test_mse = np.mean((y_test_pred - y_test) ** 2)

        # Сортируем для корректного отображения линий
        train_sort_idx = np.argsort(X_train.flatten())
        test_sort_idx = np.argsort(X_test.flatten())

        # Строим график
        ax = axes[i]
        ax.scatter(X_train, y_train, alpha=0.5, label='Обучающие данные')
        ax.scatter(X_test, y_test, alpha=0.5, label='Тестовые данные')
        ax.plot(X_train[train_sort_idx], y_train_pred[train_sort_idx], 'r-',
                linewidth=2, label='Обучающие предсказания')
        ax.plot(X_test[test_sort_idx], y_test_pred[test_sort_idx], 'g--',
                linewidth=2, label='Тестовые предсказания')

        ax.set_title(f'Набор данных: {name} | Train MSE: {train_mse:.4f} | Test MSE: {test_mse:.4f}')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()