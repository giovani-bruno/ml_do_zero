from typing import List, Tuple
import math

Vector = List[float]

def mean(xs: List[float]) -> float:
    """Retorna a média dos valores"""
    return sum(xs) / len(xs)

def dot(v: Vector, w: Vector) -> float:
    """Calcula v_1 * w_1 + ... + v_n * w_n (produto escalar)"""
    assert len(v) == len(w), "vetores devem ter o mesmo tamanho"

    return sum(v_i * w_i for v_i, w_i in zip(v, w))

def sum_of_squares(v: Vector) -> float:
    """Retorna v_1 * v_1 + ... + v_n * v_n (soma dos quadrados)"""
    return dot(v, v)

def covariance(xs: List[float], ys: List[float]) -> float:
    """Calcula a covariância entre xs e ys"""
    assert len(xs) == len(ys), "xs e ys devem ter o mesmo número de elementos"

    return dot(de_mean(xs), de_mean(ys)) / (len(xs) - 1)

def correlation(xs: List[float], ys: List[float]) -> float:
    """Mede o quanto xs e ys variam juntos em torno de suas médias (correlação)"""
    stdev_x = standard_deviation(xs)
    stdev_y = standard_deviation(ys)
    if stdev_x > 0 and stdev_y > 0:
        return covariance(xs, ys) / stdev_x / stdev_y
    else:
        return 0  # Se não houver variação, a correlação é zero

def de_mean(xs: List[float]) -> List[float]:
    """Centraliza xs subtraindo sua média (para que o resultado tenha média 0)"""
    x_bar = mean(xs)
    return [x - x_bar for x in xs]

def vector_sum(vectors: List[Vector]) -> Vector:
    """Sums all corresponding elements"""
    # Check that vectors is not empty
    assert vectors, "no vectors provided!"

    # Check the vectors are all the same size
    num_elements = len(vectors[0])
    assert all(len(v) == num_elements for v in vectors), "different sizes!"

    # the i-th element of the result is the sum of every vector[i]
    return [sum(vector[i] for vector in vectors)
            for i in range(num_elements)]

def vector_mean(vectors: List[Vector]) -> Vector:
    """Computes the element-wise average"""
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))

def standard_deviation(xs: List[float]) -> float:
    """Calcula o desvio padrão, que é a raiz quadrada da variância"""
    return math.sqrt(variance(xs))

def variance(xs: List[float]) -> float:
    """Calcula a variância, que é a média dos desvios quadrados em relação à média"""
    assert len(xs) >= 2, "variância requer pelo menos dois elementos"

    n = len(xs)
    deviations = de_mean(xs)
    return sum_of_squares(deviations) / (n - 1)

def add(v: Vector, w: Vector) -> Vector:
    """Soma os elementos correspondentes de dois vetores"""
    assert len(v) == len(w), "os vetores devem ter o mesmo tamanho"

    return [v_i + w_i for v_i, w_i in zip(v, w)]

def scalar_multiply(c: float, v: Vector) -> Vector:
    """Multiplica cada elemento do vetor por um escalar c"""
    return [c * v_i for v_i in v]

def gradient_step(v: Vector, gradient: Vector, step_size: float) -> Vector:
    """
    Move 'step_size' na direção do gradiente a partir do vetor 'v'
    (usado para atualização de parâmetros em otimização)
    """
    assert len(v) == len(gradient), "os vetores devem ter o mesmo tamanho"
    step = scalar_multiply(step_size, gradient)
    return add(v, step)
