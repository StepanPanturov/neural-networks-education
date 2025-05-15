"""### Демонстрация процесса обучения"""
from rbfn_class import RBFNetwork
from generate_data.generate_demo_data import generate_demo_data
import matplotlib.pyplot as plt

def demo_training_process():
    """
    Демонстрация процесса обучения
    """
    print("Демонстрация процесса обучения")

    # Генерируем данные
    X, y = generate_demo_data(n_samples=100, noise=0.2, func_type='sin')

    # Создаем и обучаем модель с сохранением истории
    model = RBFNetwork(n_centers=7, sigma=0.5, max_epochs=100)
    model.fit(X, y, animate=True)

    # Создаем анимацию
    ani = model.animate_training(X, y, interval=200)

    # Сохраняем анимацию (опционально)
    # ani.save('rbf_training.mp4', writer='ffmpeg')

    # Отображаем анимацию
    plt.show()
