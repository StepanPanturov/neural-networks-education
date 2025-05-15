from helping_funcs.demo_rbf_formation import demo_rbf_formation
from helping_funcs.demo_experiment_neurons_width import demo_experiment_neurons_width
from helping_funcs.demo_dataset_comparison import demo_dataset_comparison
from helping_funcs.demo_advanced_experiments import demo_advanced_experiments
from helping_funcs.demo_training_process import demo_training_process


def main():
    """
    Главная функция для запуска демонстраций
    """
    print("==== Демонстрация работы радиально-базисных нейронных сетей ====")

    print("\nДоступные демонстрации:")
    print("1. Базовая демонстрация формирования RBF функций")
    print("2. Визуализация процесса обучения")
    print("3. Эксперименты с количеством нейронов и шириной функций")
    print("4. Сравнение результатов на разных наборах данных")
    print("5. Расширенные эксперименты на сложных функциях")

    try:
        choice = int(input("\nВыберите демонстрацию (введите число от 1 до 5): "))

        if choice == 1:
            demo_rbf_formation()
        elif choice == 2:
            demo_training_process()
        elif choice == 3:
            demo_experiment_neurons_width()
        elif choice == 4:
            demo_dataset_comparison()
        elif choice == 5:
            demo_advanced_experiments()
        else:
            print("Некорректный выбор. Пожалуйста, введите число от 1 до 5.")
    except ValueError:
        print("Пожалуйста, введите целое число от 1 до 5.")


if __name__ == "__main__":
    main()