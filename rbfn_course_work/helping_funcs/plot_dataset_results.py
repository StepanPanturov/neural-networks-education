"""### Cравнение результатов на разных наборах данных:"""
import matplotlib.pyplot as plt
import numpy as np

def plot_dataset_results(X_train, y_train, X_test, y_test, y_train_pred, y_test_pred, title="Результаты аппроксимации"):
    """
    Визуализирует результаты аппроксимации на обучающей и тестовой выборках
    """
    plt.figure(figsize=(12, 8))

    # Если одномерные данные
    if X_train.shape[1] == 1:
        train_sort_idx = np.argsort(X_train.flatten())
        test_sort_idx = np.argsort(X_test.flatten())

        plt.scatter(X_train, y_train, alpha=0.5, label='Обучающие данные')
        plt.scatter(X_test, y_test, alpha=0.5, label='Тестовые данные')

        plt.plot(X_train[train_sort_idx], y_train_pred[train_sort_idx], 'r-',
                linewidth=2, label='Предсказания на обучающих данных')
        plt.plot(X_test[test_sort_idx], y_test_pred[test_sort_idx], 'g--',
                linewidth=2, label='Предсказания на тестовых данных')

    # Если двумерные данные
    elif X_train.shape[1] == 2:
        from mpl_toolkits.mplot3d import Axes3D

        # Создаем 3D график
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Обучающие данные
        ax.scatter(X_train[:, 0], X_train[:, 1], y_train, color='blue', alpha=0.5, label='Обучающие данные')

        # Тестовые данные
        ax.scatter(X_test[:, 0], X_test[:, 1], y_test, color='green', alpha=0.5, label='Тестовые данные')

        # Предсказания на тестовых данных
        ax.scatter(X_test[:, 0], X_test[:, 1], y_test_pred, color='red', alpha=0.5, label='Предсказания')

        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_zlabel('y')

    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()