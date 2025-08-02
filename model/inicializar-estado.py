"""inicializar_estado.py

Funciones utilitarias para inicializar un registro de qubits con un vector de
amplitudes ya normalizado y verificar la distribución de probabilidades
resultante.

Este módulo antes dependía de variables globales (`n_qubits`, `amplitudes`) y
se ejecutaba como script.  Ahora expone una función `initialize_state` que
recibe los parámetros de forma explícita y retorna tanto el circuito como las
probabilidades medidas.  Incluye un modo *CLI* sencillo para pruebas rápidas.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from typing import Tuple, Sequence

__all__ = ["initialize_state", "plot_probabilities"]


def initialize_state(amplitudes: Sequence[float]) -> Tuple[QuantumCircuit, dict[str, float]]:
    """Crea un circuito de inicialización y devuelve sus probabilidades.

    Parameters
    ----------
    amplitudes
        Vector (ya normalizado) de amplitudes complejas/real.

    Returns
    -------
    qc
        Circuito `QuantumCircuit` con la puerta ``initialize`` aplicada.
    probs_measured
        Diccionario ``{estado_binario: probabilidad}`` calculado desde el
        *statevector* simulado.
    """
    amplitudes = np.asarray(amplitudes, dtype=complex)
    if not np.isclose(np.linalg.norm(amplitudes), 1.0):
        raise ValueError("Las amplitudes deben estar normalizadas (norma 1).")

    n_qubits = int(np.log2(len(amplitudes)))
    qc = QuantumCircuit(n_qubits)
    qc.initialize(amplitudes, qc.qubits)

    state = Statevector.from_instruction(qc)
    probs_measured = state.probabilities_dict()
    return qc, probs_measured


def plot_probabilities(probs: dict[str, float]) -> None:
    """Grafica las probabilidades cuánticas medidas."""
    plt.bar(list(map(int, probs.keys())), list(probs.values()))
    plt.title("Probabilidades cuánticas medidas")
    plt.xlabel("Estado binario")
    plt.ylabel("Probabilidad")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Demo con amplitudes uniformes para n_qubits = 3
    n_qubits_demo = 3
    amps = np.ones(2 ** n_qubits_demo) / np.sqrt(2 ** n_qubits_demo)
    qc_demo, probs_demo = initialize_state(amps)
    print("Probabilidades:", probs_demo)
    plot_probabilities(probs_demo)

