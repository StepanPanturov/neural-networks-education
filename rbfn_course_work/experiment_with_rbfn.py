## Функция для проведения экспериментов
import numpy as np
from rbfn_class import RBFNetwork

def experiment_with_rbf(X, y, n_centers_list, sigma_list, X_test=None, y_test=None):
    """
    Проводит эксперименты с разными параметрами RBF-сети

    Параметры:
    - X, y: обучающие данные
    - n_centers_list: список количества центров для тестирования
    - sigma_list: список значений сигмы для тестирования
    - X_test, y_test: тестовые данные (необязательно)

    Возвращает:
    - DataFrame с результатами экспериментов
    """
    import pandas as pd

    results = []

    for n_centers in n_centers_list:
        for sigma in sigma_list:
            # Создаем и обучаем модель
            model = RBFNetwork(n_centers=n_centers, sigma=sigma)
            model.fit(X, y)

            # Вычисляем ошибки
            train_pred = model.predict(X)
            train_mse = np.mean((train_pred - y) ** 2)

            result = {
                'n_centers': n_centers,
                'sigma': sigma,
                'train_mse': train_mse
            }

            # Если есть тестовые данные
            if X_test is not None and y_test is not None:
                test_pred = model.predict(X_test)
                test_mse = np.mean((test_pred - y_test) ** 2)
                result['test_mse'] = test_mse

            results.append(result)

    # Создаем DataFrame с результатами
    results_df = pd.DataFrame(results)
    return results_df