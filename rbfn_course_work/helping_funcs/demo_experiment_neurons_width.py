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
    print("=" * 70)
    print("ЭКСПЕРИМЕНТЫ С КОЛИЧЕСТВОМ НЕЙРОНОВ И ШИРИНОЙ ФУНКЦИЙ")
    print("=" * 70)

    print("\n ЧТО МЫ ИЗУЧАЕМ:")
    print("   • Как количество нейронов влияет на качество аппроксимации")
    print("   • Как ширина радиальных функций (параметр σ) влияет на обучение")
    print("   • Поиск оптимального баланса между точностью и обобщением")

    print("\n УСЛОВИЯ ЭКСПЕРИМЕНТА:")
    print("   • Функция: сложная нелинейная функция с шумом")
    print("   • Размер выборки: 200 точек (140 для обучения, 60 для тестирования)")
    print("   • Уровень шума: 20% для имитации реальных данных")

    print("\n ПАРАМЕТРЫ ДЛЯ ИССЛЕДОВАНИЯ:")
    print("   • Количество центров (нейронов): 3, 5, 7, 10, 15, 20")
    print("     - Малое количество: может быть недостаточно для сложных функций")
    print("     - Большое количество: может привести к переобучению")
    print("   • Ширина функций (σ): 0.1, 0.5, 1.0, 2.0")
    print("     - Малая σ: узкие функции, локальное влияние")
    print("     - Большая σ: широкие функции, глобальное влияние")

    print("\n Начинаем эксперимент...")
    print("-" * 50)

    # Генерируем данные
    print("Генерация демонстрационных данных...")
    X, y = generate_demo_data(n_samples=200, noise=0.2, func_type='complex')

    # Разделяем на обучающую и тестовую выборки
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print(f"✓ Данные разделены: {len(X_train)} для обучения, {len(X_test)} для тестирования")

    # Задаем параметры для экспериментов
    n_centers_list = [3, 5, 7, 10, 15, 20]
    sigma_list = [0.1, 0.5, 1.0, 2.0]

    print(f"✓ Будет протестировано {len(n_centers_list) * len(sigma_list)} комбинаций параметров")

    # Проводим эксперименты
    print("\nПроведение экспериментов (это может занять некоторое время)...")
    results = experiment_with_rbf(X_train, y_train, n_centers_list, sigma_list, X_test, y_test)

    print("\n РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТОВ")
    print("=" * 50)

    # Визуализируем результаты
    visualize_experiment_results(results, test_data_available=True)

    # Выводим краткую сводку результатов
    print("\n СВОДКА РЕЗУЛЬТАТОВ:")
    print(f"   • Всего протестировано комбинаций: {len(results)}")
    print(f"   • Минимальная ошибка на тесте: {results['test_mse'].min():.6f}")
    print(f"   • Максимальная ошибка на тесте: {results['test_mse'].max():.6f}")
    print(f"   • Средняя ошибка на тесте: {results['test_mse'].mean():.6f}")

    # Находим лучшую комбинацию параметров
    best_idx = results['test_mse'].idxmin()
    best_params = results.iloc[best_idx]

    print(f"\n ЛУЧШИЕ ПАРАМЕТРЫ:")
    print(f"   • Количество центров: {int(best_params['n_centers'])}")
    print(f"   • Ширина функций (σ): {best_params['sigma']}")
    print(f"   • Ошибка на обучающей выборке: {best_params['train_mse']:.6f}")
    print(f"   • Ошибка на тестовой выборке: {best_params['test_mse']:.6f}")

    # Анализ переобучения
    overfit_ratio = best_params['test_mse'] / best_params['train_mse']
    if overfit_ratio > 2.0:
        print("Модель может переобучаться (тестовая ошибка >> обучающей)")
    elif overfit_ratio < 1.2:
        print("Модель хорошо обобщает (низкое переобучение)")
    else:
        print("Умеренное переобучение (нормально для сложных данных)")

    # Визуализируем лучшую модель
    print(f"\n ВИЗУАЛИЗАЦИЯ ЛУЧШЕЙ МОДЕЛИ")
    print("-" * 40)

    best_model = RBFNetwork(n_centers=int(best_params['n_centers']), sigma=best_params['sigma'])
    best_model.fit(X_train, y_train)

    # Предсказания на тестовой выборке
    y_pred = best_model.predict(X_test)

    plt.figure(figsize=(12, 8))

    # Основной график
    plt.subplot(2, 1, 1)
    plt.scatter(X_train, y_train, alpha=0.6, color='blue', s=30, label='Обучающие данные')
    plt.scatter(X_test, y_test, alpha=0.6, color='green', s=30, label='Тестовые данные')

    # Сортируем для корректного отображения линии
    sort_idx = np.argsort(X_test.flatten())
    plt.plot(X_test[sort_idx], y_pred[sort_idx], 'r-', linewidth=2, label='Предсказание RBF')

    plt.title(f'Лучшая модель: {int(best_params["n_centers"])} центров, σ={best_params["sigma"]}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # График центров RBF функций
    plt.subplot(2, 1, 2)
    x_range = np.linspace(X.min(), X.max(), 300)
    rbf_outputs = []

    for i, center in enumerate(best_model.centers):
        rbf_output = np.exp(-((x_range - center) ** 2) / (2 * best_params['sigma'] ** 2))
        rbf_outputs.append(rbf_output)
        plt.plot(x_range, rbf_output, alpha=0.7, label=f'RBF {i+1}')

    plt.title('Радиальные базисные функции (центры и их влияние)')
    plt.xlabel('X')
    plt.ylabel('Активация RBF')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()

    print("\n ВЫВОДЫ И РЕКОМЕНДАЦИИ:")
    print("-" * 40)

    # Анализ найденных параметров
    best_n_centers = int(best_params['n_centers'])
    best_sigma = best_params['sigma']

    if best_n_centers <= 5:
        print("   • Малое количество центров оказалось оптимальным")
        print("     - Простая модель с хорошим обобщением")
        print("     - Подходит для не очень сложных функций")
    elif best_n_centers <= 10:
        print("   • Умеренное количество центров показало лучший результат")
        print("     - Баланс между сложностью и обобщением")
        print("     - Оптимально для большинства задач")
    else:
        print("   • Большое количество центров потребовалось для точности")
        print("     - Сложная функция требует больше параметров")
        print("     - Следите за переобучением!")

    if best_sigma <= 0.5:
        print("   • Узкие RBF функции (малая σ) работают лучше")
        print("     - Локальное влияние каждого центра")
        print("     - Подходит для данных с резкими изменениями")
    else:
        print("   • Широкие RBF функции (большая σ) предпочтительнее")
        print("     - Глобальное влияние центров")
        print("     - Подходит для гладких функций")

    print(f"\n Полная таблица результатов:")
    print(results.round(6))

    print("\n Эксперимент завершен!")
    print("   Вы изучили влияние ключевых гиперпараметров RBF сетей")
    print("   и нашли оптимальную конфигурацию для данной задачи.")
