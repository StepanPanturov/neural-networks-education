## Визуализация результатов
import matplotlib.pyplot as plt

def visualize_experiment_results(results_df, test_data_available=False):
    """
    Визуализирует результаты экспериментов
    """
    n_centers_list = sorted(results_df['n_centers'].unique())
    sigma_list = sorted(results_df['sigma'].unique())

    # Создаем фигуру
    fig, ax = plt.subplots(1, 2 if test_data_available else 1, figsize=(16, 8 if test_data_available else 6))

    if not test_data_available:
        ax = [ax]

    # График зависимости ошибки от количества центров
    for sigma in sigma_list:
        subset = results_df[results_df['sigma'] == sigma]
        ax[0].plot(subset['n_centers'], subset['train_mse'], 'o-', label=f'sigma={sigma}')

    ax[0].set_xlabel('Количество центров')
    ax[0].set_ylabel('MSE на обучающей выборке')
    ax[0].set_title('Зависимость ошибки от количества центров')
    ax[0].legend()
    ax[0].grid(True)

    if test_data_available:
        # График сравнения ошибок на обучающей и тестовой выборках
        for sigma in sigma_list:
            subset = results_df[results_df['sigma'] == sigma]
            ax[1].plot(subset['n_centers'], subset['train_mse'], 'o-', label=f'train, sigma={sigma}')
            ax[1].plot(subset['n_centers'], subset['test_mse'], 'x--', label=f'test, sigma={sigma}')

        ax[1].set_xlabel('Количество центров')
        ax[1].set_ylabel('MSE')
        ax[1].set_title('Сравнение обучающей и тестовой ошибок')
        ax[1].legend()
        ax[1].grid(True)

    plt.tight_layout()
    plt.show()