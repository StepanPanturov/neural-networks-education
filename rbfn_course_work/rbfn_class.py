import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from IPython.display import HTML
from matplotlib import animation

class RBFNetwork:
    def __init__(self, n_centers, sigma=1.0, learning_rate=0.01, max_epochs=100, rbf_type='gaussian',
             alpha=0.5, beta=0.5, animation_interval=5, marker_size=80):
        """
        Инициализация радиально-базисной сети

        Параметры:
        - n_centers: количество центров (нейронов скрытого слоя)
        - sigma: ширина радиальных функций
        - learning_rate: скорость обучения
        - max_epochs: максимальное число эпох обучения
        - rbf_type: тип радиальной базисной функции ('gaussian', 'linear', 'thin_plate_spline',
                    'logistic', 'hardy_multiquadric', 'multiquadric', 'dsp_kernel', 'proposed_quadratic')
        - alpha: параметр для функций, требующих alpha (по умолчанию 0.5)
        - beta: параметр для функций, требующих beta (по умолчанию 0.5)
        - animation_interval: интервал записи кадров для анимации (каждая N-я эпоха)
        - marker_size: размер маркеров центров на графике
        """
        self.n_centers = n_centers
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.rbf_type = rbf_type
        self.alpha = alpha
        self.beta = beta
        self.animation_interval = animation_interval
        self.marker_size = marker_size
        self.centers = None
        self.weights = None
        self.training_history = []

        # Проверка допустимого типа RBF
        self._available_rbf_types = ['gaussian', 'linear', 'thin_plate_spline', 'logistic',
                                    'hardy_multiquadric', 'multiquadric', 'dsp_kernel', 'proposed_quadratic']
        if rbf_type not in self._available_rbf_types:
            raise ValueError(f"Тип RBF должен быть одним из {self._available_rbf_types}")

    def _gaussian(self, distances):
        """
        Радиальная базисная функция (гауссиана): h(r) = exp(-r²/2σ²)
        """
        return np.exp(-0.5 * (distances / self.sigma) ** 2)

    def _linear(self, distances):
        """
        Линейная функция: h(r) = r
        """
        return distances

    def _thin_plate_spline(self, distances):
        """
        Функция тонкой пластины (Thin-plate spline): h(r) = r² * ln(r)
        """
        # Избегаем логарифма от нуля
        mask = distances > 0
        result = np.zeros_like(distances)
        result[mask] = (distances[mask] ** 2) * np.log(distances[mask])
        return result

    def _logistic(self, distances):
        """
        Логистическая функция: h(r) = 1 / (1 + exp((r²-r₀²)/σ²))
        Упрощенная версия: h(r) = 1 / (1 + exp(r²/σ²))
        """
        return 1 / (1 + np.exp((distances ** 2) / (self.sigma ** 2)))

    def _hardy_multiquadric(self, distances):
        """
        Функция Харди (Hardy multiquadric): h(r) = 1 / ((r² + σ²)^α), α > 0
        """
        return 1 / ((distances ** 2 + self.sigma ** 2) ** self.alpha)

    def _multiquadric(self, distances):
        """
        Мультиквадратичная функция (Multiquadric): h(r) = (r² + σ²)^β, 0 < β < 1
        """
        return (distances ** 2 + self.sigma ** 2) ** self.beta

    def _dsp_kernel(self, distances):
        """
        DSP kernel: h(r) = 1 / (1 + r²/σ²)
        """
        return 1 / (1 + (distances ** 2) / (self.sigma ** 2))

    def _proposed_quadratic(self, distances):
        """
        Предложенная квадратичная функция: h(r) линейна на основе r²
        """
        # На основе r², линейное преобразование от r²
        # Можно реализовать различными способами, один из вариантов:
        return np.maximum(0, 1 - (distances ** 2) / (self.sigma ** 2))

    def _apply_rbf(self, distances):
        """
        Применяет выбранную радиальную базисную функцию к расстояниям
        """
        if self.rbf_type == 'gaussian':
            return self._gaussian(distances)
        elif self.rbf_type == 'linear':
            return self._linear(distances)
        elif self.rbf_type == 'thin_plate_spline':
            return self._thin_plate_spline(distances)
        elif self.rbf_type == 'logistic':
            return self._logistic(distances)
        elif self.rbf_type == 'hardy_multiquadric':
            return self._hardy_multiquadric(distances)
        elif self.rbf_type == 'multiquadric':
            return self._multiquadric(distances)
        elif self.rbf_type == 'dsp_kernel':
            return self._dsp_kernel(distances)
        elif self.rbf_type == 'proposed_quadratic':
            return self._proposed_quadratic(distances)
        else:
            return self._gaussian(distances)  # По умолчанию используем гауссиану

    def _calculate_activations(self, X):
        """
        Вычисляет активации скрытого слоя
        """
        distances = cdist(X, self.centers)
        return self._apply_rbf(distances)

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

            # Вычисляем начальную MSE для эпохи 0
            hidden_activations = self._calculate_activations(X)
            predictions = hidden_activations @ self.weights
            error = y - predictions
            initial_mse = np.mean(error ** 2)

            # Сохраняем начальное состояние (эпоха 0)
            self.training_history.append({
                'epoch': 0,
                'centers': self.centers.copy(),
                'weights': self.weights.copy(),
                'mse': initial_mse
            })

        # Обучение
        for epoch in range(1, self.max_epochs + 1):  # Начинаем с 1, так как 0 уже сохранена
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

            # Сохранение для анимации с учетом интервала
            if animate and epoch % self.animation_interval == 0:
                self.training_history.append({
                    'epoch': epoch,
                    'centers': self.centers.copy(),
                    'weights': self.weights.copy(),
                    'mse': mse
                })

            # Досрочная остановка при достижении малой ошибки
            if mse < 1e-6:
                # Если остановились досрочно, сохраняем финальное состояние
                if animate and epoch % self.animation_interval != 0:
                    self.training_history.append({
                        'epoch': epoch,
                        'centers': self.centers.copy(),
                        'weights': self.weights.copy(),
                        'mse': mse
                    })
                break

        # Если анимация включена, сохраняем финальное состояние (если еще не сохранили)
        if animate and self.max_epochs % self.animation_interval != 0:
            final_epoch = min(epoch if 'epoch' in locals() else self.max_epochs, self.max_epochs)
            # Проверяем, не сохранили ли мы уже это состояние
            if not self.training_history or self.training_history[-1]['epoch'] != final_epoch:
                # Вычисляем финальную MSE, если еще не вычислили
                hidden_activations = self._calculate_activations(X)
                predictions = hidden_activations @ self.weights
                error = y - predictions
                final_mse = np.mean(error ** 2)

                self.training_history.append({
                    'epoch': final_epoch,
                    'centers': self.centers.copy(),
                    'weights': self.weights.copy(),
                    'mse': final_mse
                })

        return self

    def predict(self, X):
        """
        Предсказание значений для новых данных
        """
        hidden_activations = self._calculate_activations(X)
        return hidden_activations @ self.weights

    def visualize_rbf_functions(self, X_range, feature_names=None, X_train=None, y_train=None):
        """
        Визуализирует радиальные базисные функции на заданном диапазоне X
        с максимальной информативностью для понимания принципа работы

        Параметры:
        - X_range: диапазон данных для визуализации
        - feature_names: названия признаков
        - X_train: обучающие данные (для отображения на графике)
        - y_train: целевые значения обучающих данных
        """
        if X_range.shape[1] > 2:
            print("Визуализация возможна только для 1D или 2D данных")
            return

        if X_range.shape[1] == 1:
            # 1D случай - подробная визуализация
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

            x = X_range.flatten()

            # График 1: Индивидуальные RBF функции
            ax1.set_title(f'Индивидуальные радиальные базисные функции (тип: {self.rbf_type})\n'
                        f'Каждая функция "реагирует" сильнее всего в своем центре', fontsize=12, pad=20)

            colors = plt.cm.Set3(np.linspace(0, 1, len(self.centers)))

            # Активации каждого RBF-нейрона
            for i, center in enumerate(self.centers):
                distances = np.abs(x - center.item())
                activations = self._apply_rbf(distances)

                # Рисуем RBF функцию
                ax1.plot(x, activations, color=colors[i], linewidth=2,
                        label=f'RBF {i+1} (центр: {center.item():.2f})')

                # Выделяем центр функции
                ax1.axvline(center.item(), color=colors[i], linestyle=':', alpha=0.7, linewidth=1)
                ax1.scatter(center.item(), 1.0, color=colors[i], s=100, marker='o',
                        edgecolor='black', linewidth=2, zorder=5)

                # Добавляем аннотацию к центру
                ax1.annotate(f'Центр {i+1}', xy=(center.item(), 1.0),
                            xytext=(center.item(), 1.1), ha='center',
                            fontsize=9, color=colors[i], weight='bold')

            # Добавляем обучающие данные, если они есть
            if X_train is not None and y_train is not None:
                # Нормализуем y_train для отображения
                y_norm = (y_train - y_train.min()) / (y_train.max() - y_train.min())
                ax1.scatter(X_train.flatten(), y_norm, color='red', alpha=0.6, s=30,
                        label='Обучающие данные (нормализованы)', zorder=4)

            ax1.set_xlabel(feature_names[0] if feature_names else 'X')
            ax1.set_ylabel('Активация (0-1)')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(-0.1, 1.2)

            # График 2: Результирующая функция
            ax2.set_title('Результирующий выход сети\n'
                        'Взвешенная сумма всех RBF функций', fontsize=12, pad=20)

            # Показываем индивидуальные вклады (с весами)
            total_weighted = np.zeros_like(x)
            for i, center in enumerate(self.centers):
                distances = np.abs(x - center.item())
                activations = self._apply_rbf(distances)
                weighted_activation = activations * self.weights[i]
                total_weighted += weighted_activation

                ax2.plot(x, weighted_activation, color=colors[i], alpha=0.5, linewidth=1,
                        linestyle='--', label=f'RBF {i+1} × вес ({self.weights[i]:.2f})')

            # Результирующая функция
            activations = self._calculate_activations(X_range)
            output = activations @ self.weights
            ax2.plot(x, output, 'black', linewidth=3, label='Итоговый выход сети', zorder=3)

            # Добавляем исходные обучающие данные
            if X_train is not None and y_train is not None:
                ax2.scatter(X_train.flatten(), y_train, color='red', alpha=0.8, s=50,
                        label='Целевые значения', zorder=4, edgecolor='darkred')

            ax2.set_xlabel(feature_names[0] if feature_names else 'X')
            ax2.set_ylabel('Выходное значение')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

            # Выводим информативный текст
            print(f"\n ОБЪЯСНЕНИЕ ГРАФИКОВ:")
            print(f" Тип RBF функции: {self.rbf_type}")
            print(f" Количество RBF нейронов: {len(self.centers)}")
            print(f" Параметр ширины (sigma): {self.sigma}")
            print(f"\n ВЕРХНИЙ ГРАФИК:")
            print(f"   • Показывает {len(self.centers)} индивидуальных RBF функций")
            print(f"   • Каждая функция имеет максимум в своем центре (отмечен точкой)")
            print(f"   • Функции 'откликаются' сильнее на данные рядом с их центрами")
            print(f"\n НИЖНИЙ ГРАФИК:")
            print(f"   • Показывает, как RBF функции комбинируются в итоговый результат")
            print(f"   • Каждая RBF функция умножается на свой вес")
            print(f"   • Черная линия - итоговый выход сети (сумма всех взвешенных RBF)")
            if X_train is not None:
                print(f"   • Красные точки - данные, на которых обучалась сеть")

        else:
            # 2D случай - улучшенная визуализация
            from matplotlib import cm
            import matplotlib.patches as patches

            x_min, x_max = X_range[:, 0].min(), X_range[:, 0].max()
            y_min, y_max = X_range[:, 1].min(), X_range[:, 1].max()

            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                                np.linspace(y_min, y_max, 100))

            grid_points = np.c_[xx.ravel(), yy.ravel()]

            # Вычисляем оптимальную сетку для графиков
            n_centers = len(self.centers)
            # Отдельная фигура для итогового выхода (делаем её шире)
            n_cols = int(np.ceil(np.sqrt(n_centers)))
            n_rows = int(np.ceil(n_centers / n_cols))

            # Создаем фигуру для индивидуальных RBF функций (увеличиваем размер)
            fig1, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
            if n_rows == 1 and n_cols == 1:
                axes = [axes]
            elif n_rows == 1 or n_cols == 1:
                axes = axes.flatten()
            else:
                axes = axes.flatten()

            # Заголовок с достаточным отступом
            fig1.suptitle(f'Радиально-базисные функции в 2D (тип: {self.rbf_type})\n'
                        f'Каждая функция создает "холм" активации вокруг своего центра',
                        fontsize=14, y=0.98)

            # Визуализация каждого RBF-нейрона
            for i, center in enumerate(self.centers):
                if i >= len(axes) - 1:  # Оставляем место для общего выхода
                    break

                # Вычисляем активации для данного нейрона
                distances = np.sqrt(((grid_points - center) ** 2).sum(axis=1))
                z = self._apply_rbf(distances).reshape(xx.shape)

                # Строим контурный график с цветовой шкалой
                im = axes[i].contourf(xx, yy, z, levels=20, cmap=cm.viridis, alpha=0.8)

                # Добавляем цветовую шкалу
                cbar = plt.colorbar(im, ax=axes[i], shrink=0.8)
                cbar.set_label('Активация', rotation=270, labelpad=15)

                # Выделяем центр функции
                axes[i].scatter(center[0], center[1], color='red', s=150, marker='X',
                            edgecolor='white', linewidth=2, zorder=5)

                # Добавляем круг, показывающий область влияния
                circle = patches.Circle((center[0], center[1]), self.sigma,
                                    fill=False, edgecolor='white', linewidth=2,
                                    linestyle='--', alpha=0.7)
                axes[i].add_patch(circle)

                # Улучшенные подписи
                axes[i].set_title(f'RBF Нейрон {i+1}\nЦентр: ({center[0]:.2f}, {center[1]:.2f})\n'
                                f'Вес: {self.weights[i]:.2f}', fontsize=10)
                axes[i].set_xlabel(feature_names[0] if feature_names else 'X1')
                axes[i].set_ylabel(feature_names[1] if feature_names else 'X2')

                # Добавляем обучающие данные, если есть
                if X_train is not None:
                    axes[i].scatter(X_train[:, 0], X_train[:, 1], color='white',
                                s=20, alpha=0.6, edgecolor='black', linewidth=0.5)

            # Визуализация общего выхода сети
            activations = self._calculate_activations(grid_points)
            z_output = (activations @ self.weights).reshape(xx.shape)

            final_idx = min(i+1, len(axes)-1)
            im_final = axes[final_idx].contourf(xx, yy, z_output, levels=20, cmap=cm.RdYlBu_r, alpha=0.8)

            # Цветовая шкала для итогового выхода (располагаем справа)
            cbar_final = plt.colorbar(im_final, ax=axes[final_idx], shrink=0.8, pad=0.2)
            cbar_final.set_label('Выходное значение', rotation=270, labelpad=15)

            # Показываем все центры на итоговом графике
            axes[final_idx].scatter(self.centers[:, 0], self.centers[:, 1],
                                color='red', s=150, marker='X',
                                edgecolor='white', linewidth=2, zorder=5)

            axes[final_idx].set_title('ИТОГОВЫЙ ВЫХОД СЕТИ\n'
                                    'Взвешенная сумма всех RBF функций', fontsize=11, weight='bold')
            axes[final_idx].set_xlabel(feature_names[0] if feature_names else 'X1')
            axes[final_idx].set_ylabel(feature_names[1] if feature_names else 'X2')

            # Добавляем обучающие данные на итоговый график
            if X_train is not None and y_train is not None:
                scatter = axes[final_idx].scatter(X_train[:, 0], X_train[:, 1],
                                                c=y_train, s=50, cmap=cm.RdYlBu_r,
                                                edgecolor='black', linewidth=1, zorder=4)
                # Отдельная цветовая шкала для данных (располагаем справа с большим отступом)
                cbar_data = plt.colorbar(scatter, ax=axes[final_idx], shrink=0.6,
                                    pad=0.05)
                cbar_data.set_label('Целевые значения', rotation=270, labelpad=15)

            # Скрываем лишние графики
            for j in range(final_idx + 1, len(axes)):
                axes[j].set_visible(False)

            plt.tight_layout()
            plt.show()

            # Информативный текст для 2D случая
            print(f"\nОБЪЯСНЕНИЕ 2D ВИЗУАЛИЗАЦИИ:")
            print(f"Тип RBF функции: {self.rbf_type}")
            print(f"Количество RBF нейронов: {len(self.centers)}")
            print(f"Параметр ширины (sigma): {self.sigma}")
            print(f"\n ИНДИВИДУАЛЬНЫЕ RBF ФУНКЦИИ:")
            print(f"   • Каждый график показывает одну RBF функцию")
            print(f"   • Красный крест (X) - центр функции")
            print(f"   • Пунктирная окружность - область основного влияния")
            print(f"   • Цвет показывает силу активации (темнее = сильнее)")
            print(f"   • Белые точки - обучающие данные")
            print(f"\n ИТОГОВЫЙ ВЫХОД:")
            print(f"   • Показывает результат работы всей сети")
            print(f"   • Комбинирует все RBF функции с их весами")
            print(f"   • Цветные точки - целевые значения данных для обучения")

    def visualize_rbf_types(self, r_range=(-2, 2), num_points=1000):
        """
        Визуализирует различные типы радиальных базисных функций

        Параметры:
        - r_range: диапазон значений r для визуализации (min, max)
        - num_points: количество точек для построения графика
        """
        # Сохраняем текущий тип RBF
        current_rbf_type = self.rbf_type

        # Создаем диапазон значений r
        r = np.linspace(r_range[0], r_range[1], num_points)
        r_squared = r ** 2

        plt.figure(figsize=(12, 10))

        # Построение графиков для различных типов RBF
        for rbf_type in self._available_rbf_types:
            self.rbf_type = rbf_type
            activations = self._apply_rbf(np.abs(r))
            plt.plot(r, activations, label=rbf_type)

        # Восстанавливаем исходный тип RBF
        self.rbf_type = current_rbf_type

        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        plt.grid(True)
        plt.xlabel('r')
        plt.ylabel('h(r)')
        plt.title('Сравнение различных типов радиальных базисных функций')
        plt.legend()
        plt.show()

        # Также построим графики для h(r²)
        plt.figure(figsize=(12, 10))

        for rbf_type in self._available_rbf_types:
            self.rbf_type = rbf_type
            activations = self._apply_rbf(np.sqrt(np.abs(r_squared)))
            plt.plot(r_squared, activations, label=rbf_type)

        # Восстанавливаем исходный тип RBF
        self.rbf_type = current_rbf_type

        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        plt.grid(True)
        plt.xlabel('r²')
        plt.ylabel('h(r²)')
        plt.title('Сравнение различных типов радиальных базисных функций (зависимость от r²)')
        plt.legend()
        plt.show()

    def animate_training(self, X, y, interval=200):
        """
        Создает анимацию процесса обучения
        """
        if not self.training_history:
            print("Нет истории обучения. Запустите fit с параметром animate=True")
            return None

        fig, ax = plt.subplots(figsize=(10, 6))

        if X.shape[1] == 1:
            # 1D случай
            scatter = ax.scatter(X, y, alpha=0.6, label='Обучающие точки', color='blue')
            line, = ax.plot([], [], 'r-', label='Предсказание', linewidth=2)
            centers, = ax.plot([], [], 'kx', markersize=10, label='Центры', markeredgewidth=2)

            ax.set_xlim(X.min() - 0.5, X.max() + 0.5)
            ax.set_ylim(y.min() - 0.5, y.max() + 0.5)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title(f'Обучение RBF-сети (тип: {self.rbf_type})')

            # Текст для отображения эпохи и MSE
            epoch_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

            # Создаем X для гладкой кривой предсказания
            X_smooth = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)

            def update(frame):
                if frame >= len(self.training_history):
                    return line, centers, epoch_text

                # Обновляем веса и центры
                frame_data = self.training_history[frame]
                centers_pos = frame_data['centers']
                weights = frame_data['weights']
                epoch = frame_data['epoch']
                mse = frame_data.get('mse', None)

                # Сохраняем текущие значения модели
                original_centers = self.centers.copy()
                original_weights = self.weights.copy()

                # Устанавливаем текущие значения для предсказания
                self.centers = centers_pos
                self.weights = weights

                try:
                    # Предсказание на гладкой кривой
                    y_pred = self.predict(X_smooth)

                    # Обновляем графики
                    line.set_data(X_smooth.flatten(), y_pred)
                    centers.set_data(centers_pos.flatten(), [y.min() - 0.3] * len(centers_pos))

                    # Обновляем текст с информацией
                    if mse is not None:
                        epoch_text.set_text(f'Эпоха: {epoch}, MSE: {mse:.6f}')
                    else:
                        epoch_text.set_text(f'Эпоха: {epoch}, MSE: вычисляется...')

                except Exception as e:
                    print(f"Ошибка в кадре {frame}: {e}")
                    # В случае ошибки показываем базовую информацию
                    epoch_text.set_text(f'Эпоха: {epoch}')
                finally:
                    # Восстанавливаем исходные значения модели
                    self.centers = original_centers
                    self.weights = original_weights

                return line, centers, epoch_text

        else:
            # 2D случай (упрощенный для наглядности)
            scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.6)
            centers_scatter = ax.scatter([], [], color='red', s=100, marker='x',
                                    label='Центры', linewidths=2)

            ax.set_xlim(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
            ax.set_ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('X1')
            ax.set_ylabel('X2')
            ax.set_title(f'Обучение RBF-сети (тип: {self.rbf_type})')

            # Текст для отображения эпохи и MSE
            epoch_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

            def update(frame):
                if frame >= len(self.training_history):
                    return centers_scatter, epoch_text

                # Получаем данные кадра
                frame_data = self.training_history[frame]
                centers_pos = frame_data['centers']
                epoch = frame_data['epoch']
                mse = frame_data.get('mse', None)

                # Обновляем график
                centers_scatter.set_offsets(centers_pos)

                # Обновляем текст с информацией
                if mse is not None:
                    epoch_text.set_text(f'Эпоха: {epoch}, MSE: {mse:.6f}')
                else:
                    epoch_text.set_text(f'Эпоха: {epoch}, MSE: вычисляется...')

                return centers_scatter, epoch_text

        # Создаем анимацию с отключенным blit для более стабильной работы
        ani = FuncAnimation(fig, update, frames=len(self.training_history),
                            interval=interval, blit=False, repeat=True)

        plt.tight_layout()

        # Убеждаемся, что анимация будет отображена
        print(f"Создана анимация с {len(self.training_history)} кадрами")

        return ani

    @staticmethod
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