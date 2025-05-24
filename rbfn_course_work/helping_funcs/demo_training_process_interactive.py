from rbfn_class import RBFNetwork
import matplotlib.pyplot as plt
from generate_data.generate_demo_data import generate_demo_data

def demo_training_process_interactive():
    """
    Интерактивная демонстрация процесса обучения с выбором параметров
    """
    print("=== Демонстрация процесса обучения RBF-сети ===")
    print()

    # Выбор параметров пользователем
    print("Настройка параметров сети:")
    print("-" * 40)

    # Выбор количества центров
    while True:
        try:
            print("Количество центров (n_centers):")
            print("  Рекомендуемые значения: 3-15")
            print("  Меньше центров = более гладкая аппроксимация")
            print("  Больше центров = более точная, но может быть переобучение")
            n_centers = int(input("Введите количество центров (по умолчанию 7): ") or "7")
            if n_centers < 1:
                print("Количество центров должно быть больше 0!")
                continue
            elif n_centers > 50:
                print("Слишком много центров! Рекомендуется не более 50.")
                continue
            break
        except ValueError:
            print("Пожалуйста, введите целое число!")

    print()

    # Выбор параметра sigma (ширина RBF функций)
    while True:
        try:
            print("Параметр sigma (ширина RBF функций):")
            print("  Рекомендуемые значения: 0.1-2.0")
            print("  Меньше sigma = более узкие функции, локальное влияние")
            print("  Больше sigma = более широкие функции, глобальное влияние")
            sigma = float(input("Введите sigma (по умолчанию 0.5): ") or "0.5")
            if sigma <= 0:
                print("Sigma должна быть больше 0!")
                continue
            elif sigma > 10:
                print("Слишком большая sigma! Рекомендуется не более 10.")
                continue
            break
        except ValueError:
            print("Пожалуйста, введите число!")

    print()

    # Выбор максимального количества эпох
    while True:
        try:
            print("Максимальное количество эпох обучения:")
            print("  Рекомендуемые значения: 50-200")
            print("  Меньше эпох = быстрее, но может не доучиться")
            print("  Больше эпох = точнее, но дольше и риск переобучения")
            max_epochs = int(input("Введите количество эпох (по умолчанию 100): ") or "100")
            if max_epochs < 1:
                print("Количество эпох должно быть больше 0!")
                continue
            elif max_epochs > 1000:
                print("Слишком много эпох! Рекомендуется не более 1000.")
                continue
            break
        except ValueError:
            print("Пожалуйста, введите целое число!")

    print()

    # Выбор интервала анимации
    while True:
        try:
            print("Интервал записи кадров анимации (animation_interval):")
            print("  Рекомендуемые значения: 1-10")
            print("  1 = каждая эпоха (более плавно, но медленнее)")
            print("  5-10 = каждая 5-10 эпоха (быстрее, менее детально)")
            animation_interval = int(input("Введите интервал (по умолчанию 5): ") or "5")
            if animation_interval < 1:
                print("Интервал должен быть больше 0!")
                continue
            elif animation_interval > max_epochs:
                print(f"Интервал не может быть больше количества эпох ({max_epochs})!")
                continue
            break
        except ValueError:
            print("Пожалуйста, введите целое число!")

    print()

    # Выбор типа данных для демонстрации
    print("Выберите тип функции для аппроксимации:")
    print("1. Синусоида (sin)")
    print("2. Экспонента (exp)")
    print("3. Квадратичная функция (quadratic)")
    print("4. Сложная функция (complex)")

    while True:
        try:
            func_choice = int(input("Выберите функцию (1-4, по умолчанию 1): ") or "1")
            if func_choice == 1:
                func_type = 'sin'
                break
            elif func_choice == 2:
                func_type = 'exp'
                break
            elif func_choice == 3:
                func_type = 'quadratic'
                break
            elif func_choice == 4:
                func_type = 'complex'
                break
            else:
                print("Выберите число от 1 до 4!")
        except ValueError:
            print("Пожалуйста, введите число!")

    print()

    # Выбор уровня шума
    while True:
        try:
            print("Уровень шума в данных:")
            print("  0.0 = без шума")
            print("  0.1 = слабый шум")
            print("  0.2 = умеренный шум")
            print("  0.3+ = сильный шум")
            noise = float(input("Введите уровень шума (по умолчанию 0.2): ") or "0.2")
            if noise < 0:
                print("Уровень шума не может быть отрицательным!")
                continue
            elif noise > 1:
                print("Слишком большой уровень шума! Рекомендуется не более 1.0")
                continue
            break
        except ValueError:
            print("Пожалуйста, введите число!")

    print()
    print("=" * 50)
    print(f"Выбранные параметры:")
    print(f"  - Количество центров: {n_centers}")
    print(f"  - Sigma: {sigma}")
    print(f"  - Максимальное количество эпох: {max_epochs}")
    print(f"  - Интервал анимации: {animation_interval}")
    print(f"  - Тип функции: {func_type}")
    print(f"  - Уровень шума: {noise}")
    print("=" * 50)
    print()

    # Подтверждение запуска
    confirm = input("Начать обучение с этими параметрами? (y/n, по умолчанию y): ").lower() or "y"
    if confirm not in ['y', 'yes', 'д', 'да']:
        print("Демонстрация отменена.")
        return

    print("\nГенерируем данные...")
    # Генерируем данные
    X, y = generate_demo_data(n_samples=100, noise=noise, func_type=func_type)

    print("Создаем и настраиваем модель...")
    # Создаем и обучаем модель с сохранением истории
    model = RBFNetwork(n_centers=n_centers, sigma=sigma, max_epochs=max_epochs,
                       animation_interval=animation_interval)

    print("Начинаем обучение (это может занять некоторое время)...")
    model.fit(X, y, animate=True)

    # Выбор - сохранять ли анимацию в файл
    print("\nОбучение завершено!")
    save_choice = input("Сохранить анимацию в файл? (y/n, по умолчанию n): ").lower() or "n"

    print("Создаем анимацию...")
    # Создаем анимацию
    ani = model.animate_training(X, y, interval=200)

    if save_choice in ['y', 'yes', 'д', 'да']:
        try:
            print("Сохраняем анимацию (может занять время)...")
            # Исправленное имя файла с правильным форматированием
            filename = f"rbf_training_c{n_centers}_s{sigma:.1f}_e{max_epochs}_i{animation_interval}.mp4"
            ani.save(filename, writer='ffmpeg')
            print(f"Анимация сохранена как: {filename}")
        except Exception as e:
            print(f"Ошибка при сохранении: {e}")
            print("Анимация не была сохранена, но будет показана.")

    # Сохраняем ссылку на анимацию чтобы избежать сборки мусора
    globals()['current_animation'] = ani

    print("Отображаем анимацию...")
    print("Закройте окно с анимацией для завершения демонстрации.")

    # Отображаем анимацию
    plt.show()

    # Показываем итоговую статистику
    print("\n" + "=" * 50)
    print("РЕЗУЛЬТАТЫ ОБУЧЕНИЯ:")
    if hasattr(model, 'training_history') and model.training_history:
        print(f"  - Количество кадров анимации: {len(model.training_history)}")

        # Собираем все MSE значения, исключая None
        mse_values = [frame.get('mse') for frame in model.training_history if frame.get('mse') is not None]

        if mse_values:
            print(f"  - Начальная ошибка (MSE): {mse_values[0]:.6f}")
            print(f"  - Финальная ошибка (MSE): {mse_values[-1]:.6f}")
            improvement = ((mse_values[0] - mse_values[-1]) / mse_values[0]) * 100
            print(f"  - Улучшение: {improvement:.2f}%")
        else:
            print("  - MSE данные недоступны")
    print("=" * 50)