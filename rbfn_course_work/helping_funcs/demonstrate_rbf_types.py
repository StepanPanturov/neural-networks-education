import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def demonstrate_rbf_types():
    """
    Демонстрирует различные типы радиальных базисных функций
    и их влияние на форму активаций с подробными пояснениями
    """

    def apply_rbf(r, rbf_type, sigma=1.0, alpha=0.5, beta=0.5):
        """
        Применяет различные типы радиальных базисных функций
        """
        r = np.abs(r)  # Работаем с абсолютными значениями расстояния

        if rbf_type == 'linear':
            return r
        elif rbf_type == 'gaussian':
            return np.exp(-(r**2) / (2 * sigma**2))
        elif rbf_type == 'thin_plate_spline':
            # Избегаем логарифма от нуля
            r_safe = np.where(r == 0, 1e-10, r)
            return (r**2) * np.log(r_safe)
        elif rbf_type == 'logistic':
            return 1 / (1 + np.exp((r**2) / (sigma**2)))
        elif rbf_type == 'hardy_multiquadric':
            return 1 / ((r**2 + sigma**2)**alpha)
        elif rbf_type == 'multiquadric':
            return (r**2 + sigma**2)**beta
        elif rbf_type == 'dsp_kernel':
            return 1 / (1 + (r**2) / (sigma**2))
        elif rbf_type == 'proposed_quadratic':
            return np.maximum(0, 1 - (r**2) / (sigma**2))
        else:
            raise ValueError(f"Неизвестный тип RBF: {rbf_type}")

    # Диапазон для визуализации
    r = np.linspace(-3, 3, 1000)
    r_abs = np.abs(r)

    # Параметры RBF функций
    sigma = 1.0
    alpha = 0.5
    beta = 0.5

    # Определяем типы функций и их описания
    rbf_functions = [
        ('linear', 'Линейная', 'h(r) = r', 'blue'),
        ('gaussian', 'Гауссова', f'h(r) = exp(-r²/2σ²), σ={sigma}', 'orange'),
        ('thin_plate_spline', 'Thin-plate Spline', 'h(r) = r² ln r', 'green'),
        ('logistic', 'Логистическая', f'h(r) = 1/(1+exp(r²/σ²)), σ={sigma}', 'red'),
        ('hardy_multiquadric', 'Hardy Multiquadric', f'h(r) = 1/(r²+σ²)^α, α={alpha}', 'purple'),
        ('multiquadric', 'Multiquadric', f'h(r) = (r²+σ²)^β, β={beta}', 'brown'),
        ('dsp_kernel', 'DSP Kernel', f'h(r) = 1/(1+r²/σ²), σ={sigma}', 'pink'),
        ('proposed_quadratic', 'Предложенная квадратичная', f'h(r) = max(0, 1-r²/σ²), σ={sigma}', 'gray')
    ]

    # Создаем фигуру с улучшенным дизайном
    fig = plt.figure(figsize=(20, 16))

    # График 1: Основные RBF функции (нормализованный масштаб)
    ax1 = plt.subplot(2, 3, 1)

    # Выбираем наиболее важные функции для лучшей читаемости
    main_functions = [
        ('gaussian', 'Гауссова', 'orange'),
        ('logistic', 'Логистическая', 'red'),
        ('dsp_kernel', 'DSP Kernel', 'blue'),
        ('proposed_quadratic', 'Предложенная квадратичная', 'green')
    ]

    for rbf_type, name, color in main_functions:
        values = apply_rbf(r_abs, rbf_type, sigma, alpha, beta)
        plt.plot(r, values, label=name, linewidth=3, color=color)

    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.8)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3, linewidth=0.8)
    plt.grid(True, alpha=0.3)
    plt.xlabel('Расстояние от центра (r)', fontsize=11, fontweight='bold')
    plt.ylabel('Значение активации h(r)', fontsize=11, fontweight='bold')
    plt.title('Основные RBF функции\n(локализованные типы)',
              fontsize=12, fontweight='bold', pad=15)
    plt.legend(fontsize=10)
    plt.xlim(-3, 3)
    plt.ylim(-0.1, 1.1)

    # График 2: Функции с неограниченным ростом (отдельный масштаб)
    ax2 = plt.subplot(2, 3, 2)

    growing_functions = [
        ('linear', 'Линейная', 'blue'),
        ('thin_plate_spline', 'Thin-plate Spline', 'green'),
        ('multiquadric', 'Multiquadric', 'brown')
    ]

    r_small = np.linspace(-2, 2, 500)  # Меньший диапазон для лучшей видимости

    for rbf_type, name, color in growing_functions:
        values = apply_rbf(np.abs(r_small), rbf_type, sigma, alpha, beta)
        if rbf_type == 'thin_plate_spline':
            # Ограничиваем значения для лучшей визуализации
            values = np.clip(values, -10, 10)
        plt.plot(r_small, values, label=name, linewidth=3, color=color)

    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.8)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3, linewidth=0.8)
    plt.grid(True, alpha=0.3)
    plt.xlabel('Расстояние от центра (r)', fontsize=11, fontweight='bold')
    plt.ylabel('Значение активации h(r)', fontsize=11, fontweight='bold')
    plt.title('RBF функции с ростом\n(неограниченные типы)',
              fontsize=12, fontweight='bold', pad=15)
    plt.legend(fontsize=10)
    plt.xlim(-2, 2)

    # График 3: Влияние параметра sigma для Гауссовой функции
    ax3 = plt.subplot(2, 3, 3)

    sigmas = [0.3, 0.7, 1.0, 1.5, 2.0]
    colors = ['darkred', 'red', 'orange', 'blue', 'darkblue']

    for sigma_val, color in zip(sigmas, colors):
        values = apply_rbf(r_abs, 'gaussian', sigma_val, alpha, beta)
        plt.plot(r, values, label=f'σ = {sigma_val}', linewidth=2.5, color=color)

    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.8)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3, linewidth=0.8)
    plt.grid(True, alpha=0.3)
    plt.xlabel('Расстояние от центра (r)', fontsize=11, fontweight='bold')
    plt.ylabel('Значение активации', fontsize=11, fontweight='bold')
    plt.title('Влияние параметра σ\n(Гауссова функция)',
              fontsize=12, fontweight='bold', pad=15)
    plt.legend(fontsize=10)
    plt.xlim(-3, 3)
    plt.ylim(-0.05, 1.05)

    # График 4: Сравнение Hardy Multiquadric с разными параметрами
    ax4 = plt.subplot(2, 3, 4)

    alphas = [0.2, 0.5, 1.0, 1.5]
    colors = ['purple', 'darkviolet', 'mediumorchid', 'plum']

    for alpha_val, color in zip(alphas, colors):
        values = apply_rbf(r_abs, 'hardy_multiquadric', sigma, alpha_val, beta)
        plt.plot(r, values, label=f'α = {alpha_val}', linewidth=2.5, color=color)

    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.8)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3, linewidth=0.8)
    plt.grid(True, alpha=0.3)
    plt.xlabel('Расстояние от центра (r)', fontsize=11, fontweight='bold')
    plt.ylabel('Значение активации', fontsize=11, fontweight='bold')
    plt.title('Hardy Multiquadric\nс разными параметрами α',
              fontsize=12, fontweight='bold', pad=15)
    plt.legend(fontsize=10)
    plt.xlim(-3, 3)

    # График 5: Логарифмический масштаб для демонстрации убывания
    ax5 = plt.subplot(2, 3, 5)

    r_log = np.linspace(0.1, 5, 500)

    log_functions = [
        ('gaussian', 'Гауссова', 'orange'),
        ('logistic', 'Логистическая', 'red'),
        ('hardy_multiquadric', 'Hardy Multiquadric', 'purple'),
        ('dsp_kernel', 'DSP Kernel', 'blue')
    ]

    for rbf_type, name, color in log_functions:
        values = apply_rbf(r_log, rbf_type, sigma, alpha, beta)
        # Избегаем логарифма от нуля
        values_safe = np.maximum(values, 1e-10)
        plt.semilogy(r_log, values_safe, label=name, linewidth=2.5, color=color)

    plt.grid(True, alpha=0.3)
    plt.xlabel('Расстояние от центра (r)', fontsize=11, fontweight='bold')
    plt.ylabel('log(Значение активации)', fontsize=11, fontweight='bold')
    plt.title('Скорость убывания функций\n(логарифмический масштаб)',
              fontsize=12, fontweight='bold', pad=15)
    plt.legend(fontsize=10)
    plt.xlim(0.1, 5)

    # График 6: Сравнение производных (градиентов)
    ax6 = plt.subplot(2, 3, 6)

    # Вычисляем численные производные
    dr = 0.01
    r_grad = np.linspace(0.1, 3, 300)

    gradient_functions = [
        ('gaussian', 'Гауссова', 'orange'),
        ('logistic', 'Логистическая', 'red'),
        ('dsp_kernel', 'DSP Kernel', 'blue')
    ]

    for rbf_type, name, color in gradient_functions:
        values_plus = apply_rbf(r_grad + dr/2, rbf_type, sigma, alpha, beta)
        values_minus = apply_rbf(r_grad - dr/2, rbf_type, sigma, alpha, beta)
        gradient = (values_plus - values_minus) / dr
        plt.plot(r_grad, gradient, label=f'∇{name}', linewidth=2.5, color=color)

    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.8)
    plt.grid(True, alpha=0.3)
    plt.xlabel('Расстояние от центра (r)', fontsize=11, fontweight='bold')
    plt.ylabel('Производная dh/dr', fontsize=11, fontweight='bold')
    plt.title('Градиенты RBF функций\n(скорость изменения)',
              fontsize=12, fontweight='bold', pad=15)
    plt.legend(fontsize=10)
    plt.xlim(0.1, 3)

    # Улучшенное размещение подграфиков
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, hspace=0.35, wspace=0.3)
    plt.show()

    # Подробная информационная таблица
    print("\n" + "="*100)
    print("ПОДРОБНОЕ СРАВНЕНИЕ РАДИАЛЬНЫХ БАЗИСНЫХ ФУНКЦИЙ")
    print("="*100)

    print("\n1. ЛОКАЛИЗОВАННЫЕ ФУНКЦИИ (убывают с расстоянием):")
    print("-" * 60)
    print("   Гауссова: h(r) = exp(-r²/2σ²)")
    print("   • Самая популярная в машинном обучении")
    print("   • Гладкая, бесконечно дифференцируемая")
    print("   • Быстро убывает к нулю")
    print("   • Хорошо подходит для аппроксимации гладких функций")

    print("\n   Логистическая: h(r) = 1/(1+exp(r²/σ²))")
    print("   • Ограничена между 0 и 1")
    print("   • Монотонно убывающая")
    print("   • Подходит для задач классификации")
    print("   • Вычислительно стабильная")

    print("\n   DSP Kernel: h(r) = 1/(1+r²/σ²)")
    print("   • Рациональная функция")
    print("   • Вычислительно эффективная")
    print("   • Хороший компромисс между локализацией и гладкостью")

    print("\n   Предложенная квадратичная: h(r) = max(0, 1-r²/σ²)")
    print("   • Компактный носитель (равна 0 за пределами σ)")
    print("   • Обеспечивает разреженность вычислений")
    print("   • Простая для вычисления")

    print("\n   Hardy Multiquadric: h(r) = 1/(r²+σ²)^α")
    print("   • Обратная мультиквадратичная функция")
    print("   • Численно стабильная")
    print("   • Параметр α контролирует скорость убывания")

    print("\n2. НЕОГРАНИЧЕННЫЕ ФУНКЦИИ (растут с расстоянием):")
    print("-" * 60)
    print("   Линейная: h(r) = r")
    print("   • Простейшая функция")
    print("   • Не имеет локализации")
    print("   • Редко используется на практике")

    print("\n   Thin-plate Spline: h(r) = r²ln(r)")
    print("   • Классическая в теории интерполяции")
    print("   • Минимизирует энергию изгиба")
    print("   • Используется в геометрическом моделировании")

    print("\n   Multiquadric: h(r) = (r²+σ²)^β")
    print("   • Растет степенным образом")
    print("   • Может привести к плохой обусловленности матриц")
    print("   • Требует осторожного выбора параметров")

    print("\n3. ПРАКТИЧЕСКИЕ РЕКОМЕНДАЦИИ:")
    print("-" * 60)
    print("   -> Для аппроксимации гладких функций: Гауссова")
    print("   -> Для задач классификации: Логистическая")
    print("   -> Для быстрых вычислений: DSP Kernel")
    print("   -> Для разреженных вычислений: Предложенная квадратичная")
    print("   -> Для интерполяции: Thin-plate Spline")
    print("   -> Для численной стабильности: Hardy Multiquadric")

    print("\n4. ВЛИЯНИЕ ПАРАМЕТРОВ:")
    print("-" * 60)
    print("   σ (sigma) - ширина функции:")
    print("   • Малое σ -> узкие, острые функции")
    print("   • Большое σ -> широкие, пологие функции")
    print("   • Влияет на способность к обобщению")

    print("\n   α, β - степенные параметры:")
    print("   • Контролируют скорость роста/убывания")
    print("   • Влияют на гладкость аппроксимации")
    print("   • Требуют экспериментального подбора")

    print("="*100)