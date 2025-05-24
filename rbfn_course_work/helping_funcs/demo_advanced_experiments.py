"""### Расширенные эксперименты:"""
from generate_data.generate_advanced_demo_data import generate_advanced_demo_data
from experiment_with_rbfn import experiment_with_rbf
from rbfn_class import RBFNetwork
from visualization.visualize_metrics import visualize_metrics
from helping_funcs.plot_dataset_results import plot_dataset_results
import numpy as np

def demo_advanced_experiments():
    """
    Демонстрация расширенных экспериментов с RBF-сетями
    """
    print("=== РАСШИРЕННЫЕ ЭКСПЕРИМЕНТЫ С RBF-СЕТЯМИ ===")
    print()
    print("ЦЕЛЬ ЭКСПЕРИМЕНТА:")
    print("Исследование возможностей радиально-базисных нейронных сетей")
    print("при работе со сложными функциями, которые трудно аппроксимировать.")
    print()
    print("ПОЧЕМУ ИМЕННО ЭТИ ТИПЫ ФУНКЦИЙ?")
    print("• Многомодальная функция - содержит несколько локальных максимумов/минимумов")
    print("  (проверяет способность сети улавливать сложные зависимости)")
    print("• Разрывная функция - имеет скачки и разрывы")
    print("  (проверяет способность сети работать с нестандартными данными)")
    print("• Функция с шумными всплесками - содержит случайные аномалии")
    print("  (проверяет устойчивость сети к выбросам и шуму)")
    print()
    print("ПАРАМЕТРЫ ЭКСПЕРИМЕНТА:")
    print("• Количество центров RBF: 5, 10, 15, 20, 30, 50")
    print("  (от простых до сложных моделей)")
    print("• Ширина RBF функций (σ): 0.1, 0.5, 1.0, 2.0")
    print("  (от узких до широких функций)")
    print("• Размер выборки: 300 точек (70% обучение, 30% тест)")
    print("  (достаточно для надежной оценки)")
    print()
    print("Начинаем эксперименты...")
    print("=" * 60)

    # Генерируем более сложные наборы данных
    print("\n ГЕНЕРАЦИЯ ТЕСТОВЫХ ДАННЫХ...")
    advanced_datasets = {
        'Многомодальная': generate_advanced_demo_data(n_samples=300, noise=0.1, func_type='multimodal'),
        'Разрывная': generate_advanced_demo_data(n_samples=300, noise=0.1, func_type='discontinuous'),
        'С шумными всплесками': generate_advanced_demo_data(n_samples=300, noise=0.1, func_type='noisy_peaks')
    }

    # Параметры для экспериментов
    n_centers_list = [5, 10, 15, 20, 30, 50]
    sigma_list = [0.1, 0.5, 1.0, 2.0]

    print(f"Сгенерированы {len(advanced_datasets)} типа сложных функций")
    print(f"Будет протестировано {len(n_centers_list) * len(sigma_list)} комбинаций параметров для каждой функции")

    # Для каждого набора данных проводим эксперименты
    all_results = {}
    best_params = {}
    best_metrics = {}

    for idx, (dataset_name, (X, y)) in enumerate(advanced_datasets.items(), 1):
        print(f"\n{'='*60}")
        print(f"ЭКСПЕРИМЕНТ {idx}/3: {dataset_name.upper()} ФУНКЦИЯ")
        print(f"{'='*60}")

        # Пояснение для каждого типа функции
        if dataset_name == 'Многомодальная':
            print("   ОПИСАНИЕ: Функция sin(x) + 0.5*sin(3x) с несколькими пиками и впадинами.")
            print("   Сложность: RBF-сети должны разместить центры так, чтобы покрыть все локальные особенности.")
            print("   Ожидание: Потребуется больше центров для точной аппроксимации.")
        elif dataset_name == 'Разрывная':
            print("   ОПИСАНИЕ: Кусочная функция с резкими скачками в точках x=-2 и x=1.")
            print("   Сложность: Разрывы создают проблемы для гладких функций аппроксимации.")
            print("   Ожидание: Узкие RBF (малое σ) могут дать лучший результат.")
        elif dataset_name == 'С шумными всплесками':
            print("   ОПИСАНИЕ: Базовая функция с добавленными случайными пиками.")
            print("   Сложность: Модель должна отличать истинную закономерность от шума.")
            print("   Ожидание: Средние значения параметров могут показать лучший баланс.")

        # Разделяем на обучающую и тестовую выборки
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        print(f"\n Размеры выборок:")
        print(f"   • Обучающая выборка: {len(X_train)} точек")
        print(f"   • Тестовая выборка: {len(X_test)} точек")
        print(f"   • Диапазон значений y: [{np.min(y):.3f}, {np.max(y):.3f}]")

        # Проводим эксперименты
        print(f"\n ПОИСК ОПТИМАЛЬНЫХ ПАРАМЕТРОВ...")
        print("   Тестируем все комбинации параметров методом перебора...")

        results = experiment_with_rbf(X_train, y_train, n_centers_list, sigma_list, X_test, y_test)
        all_results[dataset_name] = results

        # Находим лучшую комбинацию параметров
        best_idx = results['test_mse'].idxmin()
        best_params[dataset_name] = results.iloc[best_idx]

        print(f"\n РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ:")
        print(f"   • Лучшие параметры: n_centers={int(best_params[dataset_name]['n_centers'])}, σ={best_params[dataset_name]['sigma']}")
        print(f"   • MSE на обучающей выборке: {best_params[dataset_name]['train_mse']:.6f}")
        print(f"   • MSE на тестовой выборке: {best_params[dataset_name]['test_mse']:.6f}")

        # Анализ результатов
        test_mse = best_params[dataset_name]['test_mse']
        train_mse = best_params[dataset_name]['train_mse']
        overfitting_ratio = test_mse / train_mse

        print(f"\n АНАЛИЗ КАЧЕСТВА:")
        if overfitting_ratio < 1.2:
            print(f"      Модель хорошо обобщает (отношение MSE тест/обучение: {overfitting_ratio:.2f})")
        elif overfitting_ratio < 2.0:
            print(f"      Небольшое переобучение (отношение MSE тест/обучение: {overfitting_ratio:.2f})")
        else:
            print(f"     Значительное переобучение (отношение MSE тест/обучение: {overfitting_ratio:.2f})")

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

        # Выводим таблицу метрик с пояснениями
        print(f"\n  ДЕТАЛЬНЫЕ МЕТРИКИ КАЧЕСТВА:")
        print(f"\n   Обучающая выборка:")
        for metric, value in train_metrics.items():
            print(f"   • {metric.upper()}: {value:.6f}")

        print(f"\n   Тестовая выборка:")
        for metric, value in test_metrics.items():
            print(f"   • {metric.upper()}: {value:.6f}")

        # Интерпретация метрик
        print(f"\n  ИНТЕРПРЕТАЦИЯ РЕЗУЛЬТАТОВ:")
        r2_score = test_metrics.get('r2', 0)
        if r2_score > 0.9:
            print(f"     Отличное качество аппроксимации (R² = {r2_score:.3f})")
        elif r2_score > 0.8:
            print(f"     Хорошее качество аппроксимации (R² = {r2_score:.3f})")
        elif r2_score > 0.6:
            print(f"      Удовлетворительное качество (R² = {r2_score:.3f})")
        else:
            print(f"    Низкое качество аппроксимации (R² = {r2_score:.3f})")

        # Визуализируем метрики
        print(f"\n  Построение графиков метрик...")
        visualize_metrics(test_metrics, title=f"Метрики качества для {dataset_name} (тестовая выборка)")

        # Визуализируем результаты на графике
        print(f"  Построение графика аппроксимации...")
        plot_dataset_results(X_train, y_train, X_test, y_test, y_train_pred, y_test_pred,
                           title=f"Результаты аппроксимации: {dataset_name}")

    # Общий анализ результатов
    print(f"\n{'='*60}")
    print(" ОБЩИЙ АНАЛИЗ РЕЗУЛЬТАТОВ ЭКСПЕРИМЕНТОВ")
    print(f"{'='*60}")

    print("\n СВОДНАЯ ТАБЛИЦА ЛУЧШИХ ПАРАМЕТРОВ:")
    for dataset_name in best_params.keys():
        params = best_params[dataset_name]
        print(f"   {dataset_name:20} | n_centers: {int(params['n_centers']):2d} | σ: {params['sigma']:4.1f} | MSE: {params['test_mse']:.6f}")

    print("\n ВЫВОДЫ:")
    print("   • RBF-сети показывают разную эффективность на разных типах функций")
    print("   • Количество центров влияет на способность модели улавливать сложные зависимости")
    print("   • Ширина функций (σ) определяет степень локализации влияния каждого центра")
    print("   • Важен баланс между точностью аппроксимации и способностью к обобщению")

    print(f"\n Эксперименты завершены! Результаты сохранены для анализа.")

    return all_results, best_params, best_metrics