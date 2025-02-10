from typing import List, NamedTuple
from collections import Counter
import math

Vector = List[float]

class LabeledPoint(NamedTuple):
    point: Vector
    label: str

def distancia(v: Vector, w: Vector) -> float:
    """Retorna a distância Euclidiana entre dois pontos."""
    return math.sqrt(sum((v_i - w_i) ** 2 for v_i, w_i in zip(v, w)))

def voto_majoritario(rotulos: List[str]) -> str:
    """Assume que os rótulos estão ordenados do mais próximo para o mais distante."""
    contagem_votos = Counter(rotulos)
    vencedor, contagem_vencedor = contagem_votos.most_common(1)[0]
    num_vencedores = len([contagem
                       for contagem in contagem_votos.values()
                       if contagem == contagem_vencedor])

    if num_vencedores == 1:
        return vencedor # vencedor único, então retorna ele
    else:
        return voto_majoritario(rotulos[:-1]) # tenta novamente sem o mais distante

# Empate, então olha os primeiros 4 e escolhe 'b'
assert voto_majoritario(['a', 'b', 'c', 'b', 'a']) == 'b'

def knn_classify(k: int,
                pontos_rotulados: List[LabeledPoint],
                novo_ponto: Vector) -> str:

    # Ordena os pontos rotulados do mais próximo para o mais distante.
    por_distancia = sorted(pontos_rotulados,
                           key=lambda lp: distancia(lp.point, novo_ponto))

    # Encontra os rótulos dos k mais próximos
    k_rotulos_mais_proximos = [lp.label for lp in por_distancia[:k]]

    # E deixa eles votarem.
    return voto_majoritario(k_rotulos_mais_proximos)
