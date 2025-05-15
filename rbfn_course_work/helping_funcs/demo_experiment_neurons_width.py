"""### Демонстрация экспериментов с разным количеством нейронов и шириной функций"""
import matplotlib.pyplot as plt
import numpy as np
from rbfn_class import RBFNetwork
from generate_data.generate_demo_data import generate_demo_data
from visualization.visualize_experiment_results import visualize_experiment_results
from experiment_with_rbfn import experiment_with_rbf

def demo_experiment_neurons_width():
    """
    Демонстрация экспериментов с разным количеством нейронов и шириной функций
    """
    print("Демонстрация экспериментов с разным количеством нейронов и шириной функций")

    # Генерируем данные
    X, y = generate_demo_data(n_samples=200, noise=0.2, func_type='complex')

    # Разделяем на обучающую и тестовую выборки
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Задаем параметры для экспериментов
    n_centers_list = [3, 5, 7, 10, 15, 20]
    sigma_list = [0.1, 0.5, 1.0, 2.0]

    # Проводим эксперименты
    results = experiment_with_rbf(X_train, y_train, n_centers_list, sigma_list, X_test, y_test)

    # Визуализируем результаты
    visualize_experiment_results(results, test_data_available=True)

    # Выводим таблицу результатов
    print("\nРезультаты экспериментов:")
    print(results)

    # Находим лучшую комбинацию параметров
    best_idx = results['test_mse'].idxmin()
    best_params = results.iloc[best_idx]

    print(f"\nЛучшие параметры: n_centers={best_params['n_centers']}, sigma={best_params['sigma']}")
    print(f"Ошибка на обучающей выборке: {best_params['train_mse']:.6f}")
    print(f"Ошибка на тестовой выборке: {best_params['test_mse']:.6f}")

    # Визуализируем лучшую модель
    print("\nВизуализация лучшей модели:")
    best_model = RBFNetwork(n_centers=int(best_params['n_centers']), sigma=best_params['sigma'])
    best_model.fit(X_train, y_train)

    # Предсказания на тестовой выборке
    y_pred = best_model.predict(X_test)

    plt.figure(figsize=(10, 6))
    plt.scatter(X_train, y_train, alpha=0.5, label='Обучающие данные')
    plt.scatter(X_test, y_test, alpha=0.5, label='Тестовые данные')

    # Сортируем для корректного отображения линии
    sort_idx = np.argsort(X_test.flatten())
    plt.plot(X_test[sort_idx], y_pred[sort_idx], 'r-', linewidth=2, label='Предсказание')

    plt.title(f'Лучшая модель: n_centers={best_params["n_centers"]}, sigma={best_params["sigma"]}')
    plt.legend()
    plt.grid(True)
    plt.show()
