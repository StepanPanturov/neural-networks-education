"""### Расширенные эксперименты:"""
from generate_data.generate_advanced_demo_data import generate_advanced_demo_data
from experiment_with_rbfn import experiment_with_rbf
from rbfn_class import RBFNetwork
from visualization.visualize_metrics import visualize_metrics
from helping_funcs.plot_dataset_results import plot_dataset_results

def demo_advanced_experiments():
    """
    Демонстрация расширенных экспериментов с RBF-сетями
    """
    print("=== Демонстрация расширенных экспериментов с RBF-сетями ===")

    # Генерируем более сложные наборы данных
    advanced_datasets = {
        'Многомодальная': generate_advanced_demo_data(n_samples=300, noise=0.1, func_type='multimodal'),
        'Разрывная': generate_advanced_demo_data(n_samples=300, noise=0.1, func_type='discontinuous'),
        'С шумными всплесками': generate_advanced_demo_data(n_samples=300, noise=0.1, func_type='noisy_peaks')
    }

    # Параметры для экспериментов
    n_centers_list = [5, 10, 15, 20, 30, 50]
    sigma_list = [0.1, 0.5, 1.0, 2.0]

    # Для каждого набора данных проводим эксперименты
    all_results = {}
    best_params = {}
    best_metrics = {}

    for dataset_name, (X, y) in advanced_datasets.items():
        print(f"\n--- Эксперименты с набором данных: {dataset_name} ---")

        # Разделяем на обучающую и тестовую выборки
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Проводим эксперименты
        results = experiment_with_rbf(X_train, y_train, n_centers_list, sigma_list, X_test, y_test)
        all_results[dataset_name] = results

        # Находим лучшую комбинацию параметров
        best_idx = results['test_mse'].idxmin()
        best_params[dataset_name] = results.iloc[best_idx]

        print(f"Лучшие параметры: n_centers={best_params[dataset_name]['n_centers']}, sigma={best_params[dataset_name]['sigma']}")
        print(f"Ошибка на обучающей выборке: {best_params[dataset_name]['train_mse']:.6f}")
        print(f"Ошибка на тестовой выборке: {best_params[dataset_name]['test_mse']:.6f}")

        # Создаем лучшую модель и вычисляем расширенные метрики
        best_model = RBFNetwork(
            n_centers=int(best_params[dataset_name]['n_centers']),
            sigma=best_params[dataset_name]['sigma']
        )
        best_model.fit(X_train, y_train)

        # Предсказания
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)

        # Вычисляем расширенные метрики
        train_metrics = RBFNetwork.calculate_metrics(y_train, y_train_pred)
        test_metrics = RBFNetwork.calculate_metrics(y_test, y_test_pred)
        best_metrics[dataset_name] = {'train': train_metrics, 'test': test_metrics}

        # Выводим таблицу метрик
        print("\nМетрики качества на обучающей выборке:")
        for metric, value in train_metrics.items():
            print(f"{metric}: {value:.6f}")

        print("\nМетрики качества на тестовой выборке:")
        for metric, value in test_metrics.items():
            print(f"{metric}: {value:.6f}")

        # Визуализируем метрики
        visualize_metrics(test_metrics, title=f"Метрики качества для {dataset_name} (тестовая выборка)")

        # Визуализируем результаты на графике
        plot_dataset_results(X_train, y_train, X_test, y_test, y_train_pred, y_test_pred,
                           title=f"Результаты для {dataset_name}")

    return all_results, best_params, best_metrics