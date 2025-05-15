import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from IPython.display import HTML
from matplotlib import animation

class RBFNetwork:
    def __init__(self, n_centers, sigma=1.0, learning_rate=0.01, max_epochs=100):
        """
        Инициализация радиально-базисной сети

        Параметры:
        - n_centers: количество центров (нейронов скрытого слоя)
        - sigma: ширина радиальных функций
        - learning_rate: скорость обучения
        - max_epochs: максимальное число эпох обучения
        """
        self.n_centers = n_centers
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.centers = None
        self.weights = None
        self.training_history = []

    def _gaussian(self, distances):
        """
        Радиальная базисная функция (гауссиана)
        """
        return np.exp(-0.5 * (distances / self.sigma) ** 2)

    def _calculate_activations(self, X):
        """
        Вычисляет активации скрытого слоя
        """
        distances = cdist(X, self.centers)
        return self._gaussian(distances)

    def fit(self, X, y, animate=False):
        """
        Обучение сети на данных

        Параметры:
        - X: входные данные, матрица размера (n_samples, n_features)
        - y: целевые значения, вектор размера (n_samples,)
        - animate: если True, сохраняет историю для анимации
        """
        # Инициализация центров с помощью K-means
        kmeans = KMeans(n_clusters=self.n_centers, random_state=42)
        kmeans.fit(X)
        self.centers = kmeans.cluster_centers_

        # Инициализация весов выходного слоя
        self.weights = np.random.randn(self.n_centers)

        # История обучения для анимации
        if animate:
            self.training_history = []
            self.training_history.append({
                'centers': self.centers.copy(),
                'weights': self.weights.copy()
            })

        # Обучение
        for epoch in range(self.max_epochs):
            # Прямой проход
            hidden_activations = self._calculate_activations(X)
            predictions = hidden_activations @ self.weights

            # Ошибка
            error = y - predictions
            mse = np.mean(error ** 2)

            # Обновление весов (градиентный спуск)
            delta_weights = self.learning_rate * hidden_activations.T @ error
            self.weights += delta_weights

            # Обновление центров (необязательно, но часто используется)
            for i in range(self.n_centers):
                # Вычисляем влияние каждого центра на общую ошибку
                activations = hidden_activations[:, i].reshape(-1, 1)
                delta_centers = self.learning_rate * np.sum(
                    (error * self.weights[i]).reshape(-1, 1) * activations *
                    (X - self.centers[i]) / (self.sigma ** 2), axis=0
                )
                self.centers[i] += delta_centers

            if animate and epoch % 5 == 0:  # Сохраняем каждую 5-ю эпоху для экономии памяти
                self.training_history.append({
                    'centers': self.centers.copy(),
                    'weights': self.weights.copy(),
                    'mse': mse
                })

            # Досрочная остановка при достижении малой ошибки
            if mse < 1e-6:
                break

        return self

    def predict(self, X):
        """
        Предсказание значений для новых данных
        """
        hidden_activations = self._calculate_activations(X)
        return hidden_activations @ self.weights

    def visualize_rbf_functions(self, X_range, feature_names=None):
        """
        Визуализирует радиальные базисные функции на заданном диапазоне X
        """
        if X_range.shape[1] > 2:
            print("Визуализация возможна только для 1D или 2D данных")
            return

        if X_range.shape[1] == 1:
            # 1D случай
            plt.figure(figsize=(10, 6))
            x = X_range.flatten()

            # Активации каждого RBF-нейрона
            for i, center in enumerate(self.centers):
                distances = np.abs(x - center.item())
                activations = self._gaussian(distances)
                plt.plot(x, activations, label=f'RBF {i+1}')
                plt.axvline(center.item(), color='r', linestyle='--', alpha=0.3)

            # Результирующая функция
            activations = self._calculate_activations(X_range)
            output = activations @ self.weights
            plt.plot(x, output, 'k--', linewidth=2, label='Выход сети')

            plt.xlabel(feature_names[0] if feature_names else 'X')
            plt.ylabel('Активация')
            plt.title('Радиальные базисные функции')
            plt.legend()
            plt.grid(True)
            plt.show()

        else:
            # 2D случай - визуализация через контурные графики
            from matplotlib import cm

            x_min, x_max = X_range[:, 0].min(), X_range[:, 0].max()
            y_min, y_max = X_range[:, 1].min(), X_range[:, 1].max()

            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                                 np.linspace(y_min, y_max, 100))

            grid_points = np.c_[xx.ravel(), yy.ravel()]

            # Создаем сетку из графиков
            fig, axes = plt.subplots(int(np.sqrt(self.n_centers)) + 1,
                                     int(np.sqrt(self.n_centers)) + 1,
                                     figsize=(15, 15))
            axes = axes.flatten()

            # Визуализация каждого RBF-нейрона
            for i, center in enumerate(self.centers):
                if i >= len(axes):
                    break

                # Вычисляем активации для данного нейрона
                distances = np.sqrt(((grid_points - center) ** 2).sum(axis=1))
                z = self._gaussian(distances).reshape(xx.shape)

                # Строим контурный график
                im = axes[i].contourf(xx, yy, z, cmap=cm.viridis)
                axes[i].scatter(center[0], center[1], color='red', s=50, marker='x')
                axes[i].set_title(f'RBF {i+1}')
                axes[i].set_xlabel(feature_names[0] if feature_names else 'X1')
                axes[i].set_ylabel(feature_names[1] if feature_names else 'X2')

            # Визуализация общего выхода сети
            activations = self._calculate_activations(grid_points)
            z_output = (activations @ self.weights).reshape(xx.shape)

            if i+1 < len(axes):
                im = axes[i+1].contourf(xx, yy, z_output, cmap=cm.viridis)
                axes[i+1].scatter(self.centers[:, 0], self.centers[:, 1],
                                 color='red', s=50, marker='x')
                axes[i+1].set_title('Общий выход сети')

            plt.tight_layout()
            plt.show()


    def animate_training(self, X, y, interval=200):
        """
        Создает анимацию процесса обучения
        """
        if not self.training_history:
            print("Нет истории обучения. Запустите fit с параметром animate=True")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        if X.shape[1] == 1:
            # 1D случай
            scatter = ax.scatter(X, y, alpha=0.6, label='Обучающие точки')
            line, = ax.plot([], [], 'r-', label='Предсказание')
            centers, = ax.plot([], [], 'kx', markersize=10, label='Центры')

            ax.set_xlim(X.min() - 0.5, X.max() + 0.5)
            ax.set_ylim(y.min() - 0.5, y.max() + 0.5)
            ax.legend()
            ax.grid(True)
            ax.set_title('Обучение RBF-сети')

            # Текст для отображения эпохи и MSE
            epoch_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

            # Создаем X для гладкой кривой предсказания
            X_smooth = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)

            def update(frame):
                # Обновляем веса и центры
                centers_pos = self.training_history[frame]['centers']
                weights = self.training_history[frame]['weights']

                # Устанавливаем текущие значения для предсказания
                self.centers = centers_pos
                self.weights = weights

                # Предсказание на гладкой кривой
                y_pred = self.predict(X_smooth)

                # Обновляем графики
                line.set_data(X_smooth.flatten(), y_pred)
                centers.set_data(centers_pos.flatten(), [y.min() - 0.3] * len(centers_pos))

                # Обновляем текст с информацией
                mse = self.training_history[frame].get("mse", "N/A")
                if mse != "N/A":
                    epoch_text.set_text(f'Эпоха: {frame*5}, MSE: {mse:.6f}')
                else:
                    epoch_text.set_text(f'Эпоха: {frame*5}, MSE: {mse}')

                return line, centers, epoch_text

        else:
            # 2D случай (упрощенный для наглядности)
            scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.6)
            centers = ax.scatter([], [], color='red', s=100, marker='x', label='Центры')

            ax.set_xlim(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
            ax.set_ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
            ax.legend()
            ax.grid(True)
            ax.set_title('Обучение RBF-сети')

            # Текст для отображения эпохи и MSE
            epoch_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

            def update(frame):
                # Обновляем центры
                centers_pos = self.training_history[frame]['centers']

                # Обновляем график
                centers.set_offsets(centers_pos)

                # Обновляем текст с информацией
                mse = self.training_history[frame].get("mse", "N/A")
                if mse != "N/A":
                    epoch_text.set_text(f'Эпоха: {frame*5}, MSE: {mse:.6f}')
                else:
                    epoch_text.set_text(f'Эпоха: {frame*5}, MSE: {mse}')

                return centers, epoch_text

        # Создаем анимацию
        ani = FuncAnimation(fig, update, frames=len(self.training_history),
                            interval=interval, blit=True)

        plt.tight_layout()
        return ani

    def calculate_metrics(y_true, y_pred):
        """
        Вычисляет различные метрики качества аппроксимации

        Параметры:
        - y_true: истинные значения
        - y_pred: предсказанные значения

        Возвращает:
        - словарь метрик
        """

        # Основные метрики
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # Нормализованная среднеквадратичная ошибка
        y_range = np.max(y_true) - np.min(y_true)
        nrmse = rmse / y_range if y_range != 0 else np.inf

        # Средняя абсолютная процентная ошибка (MAPE)
        # Избегаем деления на ноль
        non_zero = (y_true != 0)
        mape = np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100 if np.any(non_zero) else np.inf

        # Медианная абсолютная процентная ошибка (MdAPE)
        mdape = np.median(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100 if np.any(non_zero) else np.inf

        # Максимальная абсолютная ошибка
        max_error = np.max(np.abs(y_true - y_pred))

        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'NRMSE': nrmse,
            'MAPE (%)': mape,
            'MdAPE (%)': mdape,
            'Max Error': max_error
        }