from helping_funcs.demo_rbf_formation import demo_rbf_formation
from helping_funcs.demo_experiment_neurons_width import demo_experiment_neurons_width
from helping_funcs.demo_dataset_comparison import demo_dataset_comparison
from helping_funcs.demo_advanced_experiments import demo_advanced_experiments
from helping_funcs.demo_training_process import demo_training_process
from helping_funcs.demo_training_process_interactive import demo_training_process_interactive
from helping_funcs.demonstrate_rbf_types import demonstrate_rbf_types

def main():
    """
    Главная функция для запуска демонстраций
    """
    print("==== Демонстрация работы радиально-базисных нейронных сетей ====")

    # Выберите демонстрации для запуска
    print("\nДоступные демонстрации:")
    print("1. Базовая демонстрация формирования RBF функций")
    print("2. Визуализация процесса обучения")
    print("3. Визуализация процесса обучения - интерактив")
    print("4. Эксперименты с количеством нейронов и шириной функций")
    print("5. Сравнение результатов на разных наборах данных")
    print("6. Расширенные эксперименты на сложных функциях")
    print("7. Сравнение различных типов радиальных базисных функций")

    try:
        choice = int(input("\nВыберите демонстрацию (введите число от 1 до 7): "))

        if choice == 1:
            print("\n--- БАЗОВАЯ ДЕМОНСТРАЦИЯ ФОРМИРОВАНИЯ RBF ФУНКЦИЙ ---")
            print("Показывает основные принципы построения радиально-базисных функций")
            print("Демонстрирует расположение центров и форму активационных функций")
            demo_rbf_formation()
        elif choice == 2:
            print("\n--- ВИЗУАЛИЗАЦИЯ ПРОЦЕССА ОБУЧЕНИЯ ---")
            print("Показывает пошаговое обучение РБНС и изменение весов")
            print("Демонстрирует, как сеть постепенно приближает целевую функцию")
            demo_training_process()
        elif choice == 3:
            print("\n--- ИНТЕРАКТИВНАЯ ВИЗУАЛИЗАЦИЯ ОБУЧЕНИЯ ---")
            print("Позволяет управлять процессом обучения в реальном времени")
            print("Показывает динамические изменения аппроксимации на каждом шаге")
            demo_training_process_interactive()
        elif choice == 4:
            print("\n--- ЭКСПЕРИМЕНТЫ С ПАРАМЕТРАМИ СЕТИ ---")
            print("Исследует влияние количества нейронов и ширины функций на качество")
            print("Демонстрирует оптимальный выбор архитектуры РБНС")
            demo_experiment_neurons_width()
        elif choice == 5:
            print("\n--- СРАВНЕНИЕ НА РАЗНЫХ НАБОРАХ ДАННЫХ ---")
            print("Показывает работу РБНС на различных типах функций и задач")
            print("Демонстрирует универсальность и ограничения метода")
            demo_dataset_comparison()
        elif choice == 6:
            print("\n--- РАСШИРЕННЫЕ ЭКСПЕРИМЕНТЫ ---")
            print("Тестирует РБНС на сложных многомерных и нелинейных функциях")
            print("Показывает возможности аппроксимации комплексных зависимостей")
            demo_advanced_experiments()
        elif choice == 7:
            print("\n--- ТЕОРЕТИЧЕСКОЕ СРАВНЕНИЕ RBF ФУНКЦИЙ ---")
            print("Показывает математические свойства и формы различных RBF функций")
            print("Каждая функция определяет, как нейрон реагирует на расстояние от центра")
            demonstrate_rbf_types()
        else:
            print("Некорректный выбор. Пожалуйста, введите число от 1 до 7.")
    except ValueError:
        print("Пожалуйста, введите целое число от 1 до 7.")

if __name__ == "__main__":
    main()