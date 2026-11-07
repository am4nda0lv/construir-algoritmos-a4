import numpy as np
import pandas as pd


def calcular_coeficientes(arquivo: str) -> list[float]:
    dados = pd.read_csv(arquivo)
    matriz_caracteristicas = np.column_stack(
        (
            np.ones(len(dados)),
            dados["area_util_m2"].to_numpy(),
            dados["quantidade_quartos"].to_numpy(),
        )
    )
    precos = dados["preco_venda"].to_numpy()
    coeficientes, _, _, _ = np.linalg.lstsq(
        matriz_caracteristicas, precos, rcond=None
    )
    return coeficientes.tolist()