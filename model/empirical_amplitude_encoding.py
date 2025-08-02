"""
Implementación de codificación cuántica de una distribución empírica de precios al vencimiento
usando Qiskit y codificación por amplitud.

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
from qiskit_aer.noise import errors as aer_errors
from qiskit.ignis.mitigation.measurement import CompleteMeasFitter, complete_meas_cal
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import DynamicalDecoupling
from qiskit.circuit.library import XGate
import pandas as pd
import argparse
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))
from discretizacion_normalizacion import discretize_prices
from distribucion_empirica import generate_skewed_distribution

# ---------------------------
# 1. Preprocesamiento de datos
# ---------------------------

# Eliminar la función local discretize_and_normalize y su docstring

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

def get_noise_model(mitigation_type="bitflip"):
    """
    Crea un modelo de ruido configurable y técnicas de mitigación modernas.
    mitigation_type: 'bitflip', 'readout', 'dd' (dynamical decoupling), o 'none'.
    """
    if mitigation_type == "bitflip":
        noise_model = NoiseModel()
        error = pauli_error([('X', 0.001), ('I', 0.999)])
        for qubit in range(5):
            noise_model.add_all_qubit_quantum_error(error, ['id', 'u1', 'u2', 'u3', 'cx'])
        return noise_model, None, None
    elif mitigation_type == "readout":
        noise_model = NoiseModel()
        # Simular error de lectura
        readout_error = aer_errors.readout_error.ReadoutError([[0.9, 0.1], [0.2, 0.8]])
        for qubit in range(5):
            noise_model.add_readout_error(readout_error, [qubit])
        return noise_model, "readout", None
    elif mitigation_type == "dd":
        # Dynamical decoupling no requiere noise_model, sino pass manager
        dd_sequence = [XGate()]
        pass_manager = PassManager(DynamicalDecoupling(dd_sequence))
        return None, None, pass_manager
    else:
        return None, None, None

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
    parser.add_argument("--mitigation", type=str, default="bitflip", choices=["bitflip", "readout", "dd", "none"], help="Tipo de mitigación de errores cuánticos.")
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        raise FileNotFoundError(f"Archivo {args.csv} no encontrado")

    n_qubits = args.n_qubits

    # 1. Discretización y normalización
    if args.csv == "synthetic":
        # Permitir generación sintética desde CLI
        price_data = generate_skewed_distribution(n_samples=10000, spot=100, skew=-3, scale=20, seed=42)
        price_min = args.price_min if args.price_min is not None else float(np.min(price_data))
        price_max = args.price_max if args.price_max is not None else float(np.max(price_data))
    else:
        df = pd.read_csv(args.csv)
        if args.column not in df.columns:
            raise ValueError(f"Columna '{args.column}' no encontrada en el CSV. Columnas disponibles: {list(df.columns)}")
        price_data = df[args.column].dropna().astype(float).values
        price_min = args.price_min if args.price_min is not None else float(np.min(price_data))
        price_max = args.price_max if args.price_max is not None else float(np.max(price_data))

    probs, amplitudes, bins = discretize_prices(price_data, n_qubits, price_min, price_max)

    # 2. Codificación cuántica
    qc = create_amplitude_encoding_circuit(probs)

    # 3. Mitigación de errores (simulación)
    noise_model, mitigation_flag, pass_manager = get_noise_model(args.mitigation)
    if pass_manager is not None:
        simulator = AerSimulator()
        transpiled = simulator.transpile(qc, pass_manager=pass_manager)
        result = simulator.run(transpiled).result()
    else:
        simulator = AerSimulator(noise_model=noise_model)
        result = simulator.run(qc).result()
    noisy_statevector = result.get_statevector()
    measured_probs = np.abs(noisy_statevector) ** 2
    # Readout error mitigation (solo ejemplo, requiere calibración real en hardware)
    if mitigation_flag == "readout":
        meas_calibs, state_labels = complete_meas_cal(qubit_list=list(range(qc.num_qubits)), circlabel='measerrormit')
        cal_results = simulator.run(meas_calibs).result()
        meas_fitter = CompleteMeasFitter(cal_results, state_labels)
        # Aquí se podría aplicar meas_fitter.filter para corregir resultados reales
        print("[INFO] Readout error mitigation calibrada (solo ejemplo, requiere hardware real)")

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
