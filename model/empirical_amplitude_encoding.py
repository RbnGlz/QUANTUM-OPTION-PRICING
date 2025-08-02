"""
Implementación de codificación cuántica de una distribución empírica de precios al vencimiento
usando Qiskit y codificación por amplitud.

Autor: [Tu Nombre]
Fecha: [Fecha Actual]

Referencias:
- Stamatopoulos et al., "Option Pricing using Quantum Computers", Quantum, 2019.
- Qiskit Textbook: https://qiskit.org/textbook/ch-applications/finance.html
"""

import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import Initialize
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer import AerSimulator
from qiskit.algorithms.state_fidelities import ComputeUncompute
from qiskit.utils import QuantumInstance
from qiskit_aer.noise.errors import pauli_error
import pandas as pd
import argparse
import os

# ---------------------------
# 1. Preprocesamiento de datos
# ---------------------------

def discretize_and_normalize(data, n_qubits, price_min, price_max):
    """
    Discretiza y normaliza los datos históricos de precios.
    Args:
        data (np.ndarray): Precios históricos al vencimiento.
        n_qubits (int): Número de qubits para la codificación.
        price_min (float): Límite inferior del rango de precios.
        price_max (float): Límite superior del rango de precios.
    Returns:
        probs (np.ndarray): Probabilidades normalizadas para cada bin.
        bins (np.ndarray): Límites de los bins.
    """
    n_bins = 2 ** n_qubits
    bins = np.linspace(price_min, price_max, n_bins + 1)
    hist, _ = np.histogram(data, bins=bins)
    probs = hist / np.sum(hist)
    # Evitar ceros para la codificación por amplitud
    probs = np.where(probs == 0, 1e-12, probs)
    probs /= np.sum(probs)
    return probs, bins

# ---------------------------
# 2. Codificación por amplitud
# ---------------------------

def create_amplitude_encoding_circuit(probs):
    """
    Crea un circuito cuántico que codifica la distribución empírica por amplitud.
    Args:
        probs (np.ndarray): Probabilidades normalizadas.
    Returns:
        QuantumCircuit: Circuito cuántico de inicialización.
    """
    n_qubits = int(np.log2(len(probs)))
    amplitudes = np.sqrt(probs)
    # Normalización de amplitudes
    amplitudes /= np.linalg.norm(amplitudes)
    # Inicialización
    init_gate = Initialize(amplitudes)
    qc = QuantumCircuit(n_qubits)
    qc.append(init_gate, range(n_qubits))
    qc.barrier()
    return qc

# ---------------------------
# 3. Mitigación de errores
# ---------------------------

def get_simple_noise_model():
    """
    Crea un modelo de ruido simple para simulación y mitigación.
    Returns:
        NoiseModel: Modelo de ruido de Qiskit Aer.
    """
    noise_model = NoiseModel()
    # Ejemplo: error de bit-flip con probabilidad baja
    error = pauli_error([('X', 0.001), ('I', 0.999)])
    for qubit in range(5):  # Ajustar según el hardware/simulador
        noise_model.add_all_qubit_quantum_error(error, ['id', 'u1', 'u2', 'u3', 'cx'])
    return noise_model

# ---------------------------
# 4. Verificación y comparación
# ---------------------------

def verify_state(qc, target_probs):
    """
    Verifica que el estado cuántico preparado corresponde a la distribución objetivo.
    Args:
        qc (QuantumCircuit): Circuito cuántico preparado.
        target_probs (np.ndarray): Probabilidades objetivo.
    Returns:
        fidelity (float): Fidelidad entre el estado preparado y el objetivo.
    """
    backend = Aer.get_backend('statevector_simulator')
    result = execute(qc, backend).result()
    statevector = result.get_statevector()
    # Fidelidad clásica (cuadrado del producto escalar)
    target_amplitudes = np.sqrt(target_probs)
    target_amplitudes /= np.linalg.norm(target_amplitudes)
    fidelity = np.abs(np.dot(np.conj(target_amplitudes), statevector)) ** 2
    return fidelity

def classical_histogram(data, bins):
    """
    Calcula el histograma clásico para comparación.
    """
    hist, _ = np.histogram(data, bins=bins)
    return hist / np.sum(hist)

# ---------------------------
# 5. Ejemplo de uso y pruebas
# ---------------------------

if __name__ == "__main__":
    # --- Nuevo flujo principal para cargar un dataset real ---
    parser = argparse.ArgumentParser(description="Amplitude-encode an empirical distribution of option-expiry prices.")
    parser.add_argument("--csv", type=str, required=True, help="Ruta al CSV que contiene precios al vencimiento.")
    parser.add_argument("--column", type=str, default="price", help="Nombre de la columna con los precios en el CSV.")
    parser.add_argument("--n-qubits", type=int, default=3, help="Número de qubits para la codificación (2**n bins).")
    parser.add_argument("--price-min", type=float, default=None, help="Límite inferior opcional del rango de precios.")
    parser.add_argument("--price-max", type=float, default=None, help="Límite superior opcional del rango de precios.")
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        raise FileNotFoundError(f"Archivo {args.csv} no encontrado")

    df = pd.read_csv(args.csv)
    if args.column not in df.columns:
        raise ValueError(f"Columna '{args.column}' no encontrada en el CSV. Columnas disponibles: {list(df.columns)}")

    price_data = df[args.column].dropna().astype(float).values

    # Determinar rangos de precios
    price_min = args.price_min if args.price_min is not None else float(np.min(price_data))
    price_max = args.price_max if args.price_max is not None else float(np.max(price_data))
    n_qubits = args.n_qubits

    # 1. Discretización y normalización
    probs, bins = discretize_and_normalize(price_data, n_qubits, price_min, price_max)

    # 2. Codificación cuántica
    qc = create_amplitude_encoding_circuit(probs)

    # 3. Mitigación de errores (simulación)
    noise_model = get_simple_noise_model()
    simulator = AerSimulator(noise_model=noise_model)
    result = simulator.run(qc).result()
    noisy_statevector = result.get_statevector()
    measured_probs = np.abs(noisy_statevector) ** 2

    # 4. Verificación
    fidelity = verify_state(qc, probs)
    print(f"Fidelidad del estado cuántico preparado: {fidelity:.6f}")

    # 5. Comparativa clásica
    classical_probs = classical_histogram(price_data, bins)
    print("Probabilidades clásicas:", classical_probs)
    print("Probabilidades cuánticas (simuladas):", measured_probs)

    # 6. Assert opcional de fidelidad
    if fidelity <= 0.99:
        print("Advertencia: La fidelidad puede ser insuficiente para algunas aplicaciones financieras.")