"""### Демонстрация сравнения результатов на разных наборах данных"""
from generate_data.generate_demo_data import generate_demo_data
import matplotlib.pyplot as plt
import numpy as np
from rbfn_class import RBFNetwork

def demo_dataset_comparison():
    """
    Демонстрация сравнения результатов на разных наборах данных
    """
    print("=" * 70)
    print("ДЕМОНСТРАЦИЯ: СРАВНЕНИЕ РЕЗУЛЬТАТОВ НА РАЗНЫХ НАБОРАХ ДАННЫХ")
    print("=" * 70)

    print("\nЦЕЛЬ ЭКСПЕРИМЕНТА:")
    print("Показать, как радиально-базисные нейронные сети справляются с аппроксимацией")
    print("различных типов функций и оценить их универсальность.")

    print("\nУСЛОВИЯ ЭКСПЕРИМЕНТА:")
    print("• Количество образцов: 200 (достаточно для обучения, но не избыточно)")
    print("• Уровень шума: 0.1 (умеренный шум, имитирующий реальные данные)")
    print("• Разделение данных: 70% обучение / 30% тестирование")
    print("• Количество RBF-центров: 7 (компромисс между точностью и переобучением)")
    print("• Ширина функций (σ): 0.5 (средняя ширина для универсальности)")

    print("\nТИПЫ ДАННЫХ ДЛЯ СРАВНЕНИЯ:")
    print("1. SIN - Периодическая функция (синус)")
    print("   Особенности: регулярные колебания, проверяет способность к интерполяции")

    print("2. EXP - Гауссова функция (e^(-x²))")
    print("   Особенности: колокообразная форма, быстрое затухание к краям")

    print("3. COMPLEX - Комбинированная функция (sin(x) * e^(-0.1x²))")
    print("   Особенности: сложная форма с модуляцией, наиболее сложная для аппроксимации")

    print("\nЧТО ПОКАЗЫВАЕТ ЭКСПЕРИМЕНТ:")
    print("• Train MSE - ошибка на обучающих данных (показывает качество подгонки)")
    print("• Test MSE - ошибка на тестовых данных (показывает способность к обобщению)")
    print("• Разность между Train и Test MSE указывает на переобучение")

    print("\nЗапуск эксперимента...")
    print("-" * 50)

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

    results_summary = {}

    for i, (name, (X, y)) in enumerate(datasets.items()):
        print(f"\nОБРАБОТКА НАБОРА ДАННЫХ: {name.upper()}")

        # Разделяем на обучающую и тестовую выборки
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Создаем и обучаем модель
        model = RBFNetwork(n_centers=n_centers, sigma=sigma)
        print(f"  Обучение RBF-сети с {n_centers} центрами...")
        model.fit(X_train, y_train)

        # Предсказания
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Вычисляем ошибки
        train_mse = np.mean((y_train_pred - y_train) ** 2)
        test_mse = np.mean((y_test_pred - y_test) ** 2)

        # Вычисляем коэффициент детерминации R²
        train_r2 = 1 - (np.sum((y_train - y_train_pred) ** 2) / np.sum((y_train - np.mean(y_train)) ** 2))
        test_r2 = 1 - (np.sum((y_test - y_test_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))

        # Сохраняем результаты
        results_summary[name] = {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'overfitting': abs(test_mse - train_mse)
        }

        print(f"  ✓ Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}")
        print(f"  ✓ Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")

        # Сортируем для корректного отображения линий
        train_sort_idx = np.argsort(X_train.flatten())
        test_sort_idx = np.argsort(X_test.flatten())

        # Строим график
        ax = axes[i]
        ax.scatter(X_train, y_train, alpha=0.6, s=30, label='Обучающие данные', color='blue')
        ax.scatter(X_test, y_test, alpha=0.6, s=30, label='Тестовые данные', color='magenta')
        ax.plot(X_train[train_sort_idx], y_train_pred[train_sort_idx], 'r-',
                linewidth=2.5, label='Предсказания (обуч.)', alpha=0.8)
        ax.plot(X_test[test_sort_idx], y_test_pred[test_sort_idx], 'g--',
                linewidth=2.5, label='Предсказания (тест)', alpha=0.8)

        ax.set_title(f'Набор "{name}" | Train MSE: {train_mse:.4f} | Test MSE: {test_mse:.4f} | R²: {test_r2:.3f}',
                    fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

    plt.tight_layout()
    plt.show()

    # Анализ результатов
    print("\n" + "=" * 70)
    print("АНАЛИЗ РЕЗУЛЬТАТОВ")
    print("=" * 70)

    # Найдем лучший и худший результаты
    best_dataset = min(results_summary.keys(), key=lambda k: results_summary[k]['test_mse'])
    worst_dataset = max(results_summary.keys(), key=lambda k: results_summary[k]['test_mse'])
    most_overfitted = max(results_summary.keys(), key=lambda k: results_summary[k]['overfitting'])

    print(f"\n СВОДКА ПО КАЧЕСТВУ АППРОКСИМАЦИИ:")
    for name, results in results_summary.items():
        status = ""
        if name == best_dataset:
            status = "  ЛУЧШИЙ РЕЗУЛЬТАТ"
        elif name == worst_dataset:
            status = "  СЛОЖНЕЙШАЯ ЗАДАЧА"

        print(f"\n{name.upper()}{status}:")
        print(f"  • Ошибка на тесте: {results['test_mse']:.4f}")
        print(f"  • Качество R²: {results['test_r2']:.3f} ({'отлично' if results['test_r2'] > 0.9 else 'хорошо' if results['test_r2'] > 0.7 else 'удовлетворительно'})")
        print(f"  • Переобучение: {results['overfitting']:.4f} ({'низкое' if results['overfitting'] < 0.01 else 'умеренное' if results['overfitting'] < 0.05 else 'высокое'})")

    print(f"\n ВЫВОДЫ:")
    print(f"• Лучше всего сеть справилась с функцией '{best_dataset}'")
    print(f"• Наибольшие трудности вызвала функция '{worst_dataset}'")
    print(f"• Наибольшее переобучение наблюдается у '{most_overfitted}'")

    print(f"\n ИНТЕРПРЕТАЦИЯ:")
    print(f"• Низкое значение MSE указывает на хорошую точность аппроксимации")
    print(f"• R² близкое к 1.0 означает отличное качество модели")
    print(f"• Большая разность Train/Test MSE указывает на переобучение")
    print(f"• RBF-сети лучше работают с гладкими функциями")

    print(f"\n ПРАКТИЧЕСКОЕ ЗНАЧЕНИЕ:")
    print(f"• Эксперимент показывает универсальность RBF-сетей")
    print(f"• Демонстрирует важность выбора параметров под тип задачи")
    print(f"• Подтверждает необходимость контроля переобучения")

    print("\n" + "=" * 70)