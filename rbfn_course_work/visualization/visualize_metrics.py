"""## Визуализация метрик"""
import matplotlib.pyplot as plt

def visualize_metrics(metrics_dict, title="Метрики качества аппроксимации"):
    """
    Визуализирует метрики качества в виде графика
    """
    # Создаем график
    fig, ax = plt.subplots(figsize=(12, 6))

    # Извлекаем метрики и их значения
    metrics = list(metrics_dict.keys())
    values = list(metrics_dict.values())

    # Строим столбчатую диаграмму
    bars = ax.bar(metrics, values, color='skyblue')

    # Добавляем значения над столбцами
    for bar in bars:
        height = bar.get_height()
        if height < 1e-10:  # Для очень маленьких значений
            ax.text(bar.get_x() + bar.get_width()/2., 0.01,
                   f'{height:.2e}', ha='center', va='bottom', rotation=45)
        else:
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02 * max(values),
                   f'{height:.4f}', ha='center', va='bottom', rotation=0)

    ax.set_title(title)
    ax.set_ylabel('Значение')

    # Поворачиваем надписи для более удобного чтения
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    return fig