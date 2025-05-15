"""###  Демонстрация формирования радиальных базисных функций"""
from rbfn_class import RBFNetwork
from generate_data.generate_demo_data import generate_demo_data

def demo_rbf_formation():
    """
    Демонстрация формирования радиальных базисных функций
    """
    print("Демонстрация формирования радиальных базисных функций")

    # Генерируем простые данные
    X, y = generate_demo_data(n_samples=100, noise=0.1, func_type='sin')

    # Создаем и обучаем модель
    model = RBFNetwork(n_centers=5, sigma=0.5)
    model.fit(X, y)

    # Визуализируем RBF функции
    model.visualize_rbf_functions(X, feature_names=['X'])

    print("Визуализация в 2D")
    # Генерируем 2D данные
    X_2d, y_2d = generate_demo_data(n_samples=200, noise=0.1, func_type='2d')

    # Создаем и обучаем модель
    model_2d = RBFNetwork(n_centers=9, sigma=0.5)
    model_2d.fit(X_2d, y_2d)

    # Визуализируем RBF функции в 2D
    model_2d.visualize_rbf_functions(X_2d, feature_names=['X1', 'X2'])