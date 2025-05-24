"""###  Демонстрация формирования радиальных базисных функций"""
from rbfn_class import RBFNetwork
from generate_data.generate_demo_data import generate_demo_data

def demo_rbf_formation():
    """
    Демонстрация формирования радиальных базисных функций
    с подробными объяснениями для образовательных целей
    """
    print("ДЕМОНСТРАЦИЯ: Формирование радиально-базисных функций")
    print("=" * 60)
    print("Эта демонстрация покажет, как RBF-сеть создает и использует")
    print("радиальные базисные функции для аппроксимации данных.\n")

    # === 1D ДЕМОНСТРАЦИЯ ===
    print("ЧАСТЬ 1: ОДНОМЕРНЫЙ СЛУЧАЙ (1D)")
    print("-" * 40)

    # Генерируем простые данные
    print("Генерируем тестовые данные (синусоида с шумом)...")
    X, y = generate_demo_data(n_samples=100, noise=0.1, func_type='sin')

    print(f"✓ Создано {len(X)} точек данных")
    print(f"✓ Диапазон X: от {X.min():.2f} до {X.max():.2f}")
    print(f"✓ Диапазон Y: от {y.min():.2f} до {y.max():.2f}")

    # Создаем и обучаем модель
    print("\nСоздаем RBF-сеть...")
    model = RBFNetwork(n_centers=5, sigma=0.5, rbf_type='gaussian')
    print(f"✓ Конфигурация: {model.n_centers} RBF-нейронов, sigma={model.sigma}")

    print("Обучаем сеть на данных...")
    model.fit(X, y)
    print("✓ Обучение завершено!")

    # Показываем расположение центров
    print(f"\n Центры RBF-функций расположены в точках:")
    for i, center in enumerate(model.centers):
        print(f"   RBF {i+1}: x = {center.item():.3f}")

    # Визуализируем RBF функции с обучающими данными
    print("\n Строим детальную визуализацию...")
    model.visualize_rbf_functions(X, feature_names=['X'], X_train=X, y_train=y)

    # === 2D ДЕМОНСТРАЦИЯ ===
    print("\n" + "=" * 60)
    print(" ЧАСТЬ 2: ДВУМЕРНЫЙ СЛУЧАЙ (2D)")
    print("-" * 40)

    print("Генерируем 2D данные для регрессии...")
    X_2d, y_2d = generate_demo_data(n_samples=200, noise=0.1, func_type='2d')

    print(f"✓ Создано {len(X_2d)} точек данных")
    print(f"✓ Диапазон X1: от {X_2d[:, 0].min():.2f} до {X_2d[:, 0].max():.2f}")
    print(f"✓ Диапазон X2: от {X_2d[:, 1].min():.2f} до {X_2d[:, 1].max():.2f}")
    print(f"✓ Диапазон Y: от {y_2d.min():.2f} до {y_2d.max():.2f}")

    # Создаем и обучаем 2D модель
    print("\nСоздаем 2D RBF-сеть...")
    model_2d = RBFNetwork(n_centers=9, sigma=0.5, rbf_type='gaussian')
    print(f"✓ Конфигурация: {model_2d.n_centers} RBF-нейронов, sigma={model_2d.sigma}")

    print("Обучаем сеть на 2D данных...")
    model_2d.fit(X_2d, y_2d)
    print("✓ Обучение завершено!")

    # Показываем расположение центров в 2D
    print(f"\n Центры RBF-функций в 2D пространстве:")
    for i, center in enumerate(model_2d.centers):
        print(f"   RBF {i+1}: ({center[0]:.3f}, {center[1]:.3f})")

    # Визуализируем RBF функции в 2D с обучающими данными
    print("\n Строим 2D визуализацию...")
    model_2d.visualize_rbf_functions(X_2d, feature_names=['X1', 'X2'],
                                   X_train=X_2d, y_train=y_2d)

    # === ИТОГОВЫЕ ПОЯСНЕНИЯ ===
    print("\n" + "=" * 60)
    print(" ОБРАЗОВАТЕЛЬНЫЕ ВЫВОДЫ")
    print("-" * 40)
    print(" ЧТО МЫ УЗНАЛИ:")
    print("   1. RBF-функции создают 'области влияния' вокруг своих центров")
    print("   2. Каждая функция максимально активна в своем центре")
    print("   3. Итоговый результат = сумма всех RBF × их веса")
    print("   4. В 2D случае RBF создают 'холмы' активации")
    print("   5. Сеть автоматически выбирает центры и веса при обучении")

    print("\n ПРИНЦИП РАБОТЫ:")
    print("   • Новая точка данных 'активирует' ближайшие RBF-функции")
    print("   • Чем ближе точка к центру RBF, тем сильнее активация")
    print("   • Итоговый ответ = взвешенная сумма всех активаций")
    print("   • Это позволяет аппроксимировать сложные функции!")

    print(f"\n Демонстрация завершена!")
    print(f"   Изучено {model.n_centers} RBF-функций в 1D и {model_2d.n_centers} в 2D")