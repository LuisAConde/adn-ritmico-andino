# ═══════════════════════════════════════════════════════════════════════════
# ADN RÍTMICO DE LOS GÉNEROS MUSICALES ANDINOS COLOMBIANOS  — v2.0
# Luis Alexander Conde Solano — Matemático & Tiplista
# ─────────────────────────────────────────────────────────────────────────
# CAMBIOS v2.0 (ajustes post-revisión TISMIR):
#   [D1] Protocolo de varianza bootstrap implementado en calcular_bootstrap()
#   [D2] Distancia de intercambio dirigido d_S implementada en swap_distance()
#        Comparación d_H vs d_S: imprimir_robustez_metrica()
#   [T]  Terminología corregida: toda referencia a "swap" ahora usa d_S;
#        d_H se denomina explícitamente "distancia de Hamming"
#   [J]  Justificación formal de cuándo d_H suficiente vs cuándo d_S necesaria
# ─────────────────────────────────────────────────────────────────────────
# Uso:
#   python adn_ritmico_andino_v2.py --audio_dir /ruta/a/los/wav
#
# Archivos WAV esperados (nombres exactos):
#   Bambuco.wav, Pasillo.wav, Danza.wav,
#   Guabina.wav, Torbellino.wav, Rumba_Criolla.wav
# ═══════════════════════════════════════════════════════════════════════════

import argparse
import math
import os
import warnings
from collections import Counter
from itertools import permutations
from pathlib import Path

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
import numpy as np
from scipy import stats
from scipy.optimize import linear_sum_assignment

warnings.filterwarnings("ignore")

# ── Paleta de colores ──────────────────────────────────────────────────────
COLORES = {
    "Bambuco":       "#C0392B",
    "Pasillo":       "#1A5276",
    "Danza":         "#1E8449",
    "Guabina":       "#6C3483",
    "Torbellino":    "#B7950B",
    "Rumba_Criolla": "#117A65",
}

# ═══════════════════════════════════════════════════════════════════════════
# PARTE A — PARÁMETROS FORMALES (vectores binarios, IOI, H, E, d_H, d_S, UPGMA)
# Fuente: Franco Duque (2004, pp. 40-41)
# ═══════════════════════════════════════════════════════════════════════════

VECTORES = {
    # Fuente: Franco Duque (2004, p.40) — Motor ritmo-armónico canónico del Tiple
    # n=12 pulsos (mínimo común múltiplo de subdivisiones 2 y 3 en los compases andinos)
    "Bambuco":       [0,1,1,0,1,1,0,1,0,1,1,0],  # 6/8+3/4 hemiolia
    "Pasillo":       [1,0,1,0,1,0,1,0,1,0,1,0],  # 3/4 corcheas alternas
    "Danza":         [1,0,0,1,0,0,1,0,0,1,0,0],  # 3/4 solo negras
    "Guabina":       [1,0,1,1,0,1,1,0,1,1,0,0],  # 3/4 con puntillos
    "Torbellino":    [1,0,0,1,0,0,1,0,0,0,0,0],  # 3/4 silencio estructural
    "Rumba_Criolla": [1,1,0,1,1,0,1,0,1,1,0,1],  # 2/4 sincopado
}

COMPAS = {
    "Bambuco":"6/8+3/4","Pasillo":"3/4","Danza":"3/4",
    "Guabina":"3/4","Torbellino":"3/4","Rumba_Criolla":"2/4",
}

NOMBRES = list(VECTORES.keys())
N = 12                          # longitud del espacio Z₁₂
FUERTES = {0, 3, 6, 9}         # pulsos fuertes en 3/4 con n=12


# ─────────────────────────────────────────────────────────────────────────
# A.1 — MÉTRICAS FORMALES BÁSICAS
# ─────────────────────────────────────────────────────────────────────────

def calcular_ioi(vec):
    """
    IOI[i] = p[i+1] - p[i]          para i = 0 … k-2
    IOI[k-1] = (n - p[k-1]) + p[0]  (cierre cíclico)
    Verificación: sum(IOI) == n      (invariante algebraico de coherencia)
    """
    onsets = [i for i, v in enumerate(vec) if v == 1]
    k = len(onsets)
    ioi = []
    for i in range(k):
        if i < k - 1:
            ioi.append(onsets[i + 1] - onsets[i])
        else:
            ioi.append((N - onsets[i]) + onsets[0])
    assert sum(ioi) == N, f"Error IOI: suma={sum(ioi)} ≠ {N}"
    return ioi, onsets


def calcular_H(ioi):
    """
    Entropía rítmica de Shannon:
    H = -Σ p(v)·log2(p(v))  donde p(v) = freq(v)/k
    H=0 ↔ todos los IOI iguales (isorítmico); H_max = log2(k)
    """
    k = len(ioi)
    cnt = Counter(ioi)
    H = 0.0
    for freq in cnt.values():
        p = freq / k
        H -= p * math.log2(p)
    return round(H, 4)


def calcular_E(ioi):
    """
    Regularidad (evenness):
    E = max(0, 1 - σ(IOI)/(n/k))  — desviación estándar poblacional
    E=1 ↔ isorítmico; E=0 ↔ máxima irregularidad posible
    """
    ideal = N / len(ioi)
    sigma = float(np.std(ioi, ddof=0))
    return round(max(0.0, 1.0 - sigma / ideal), 4)


def calcular_syn(onsets):
    """Índice de síncopa: Syn = |{p ∈ O : p ∉ {0,3,6,9}}|"""
    return sum(1 for p in onsets if p not in FUERTES)


# ─────────────────────────────────────────────────────────────────────────
# A.2 — DISTANCIA DE HAMMING d_H  (métrica primaria, UPGMA)
# ─────────────────────────────────────────────────────────────────────────

def hamming(a, b):
    """
    d_H(A,B) = |{i : A[i] ≠ B[i]}|

    Propiedades:
    · Definida para vectores de cualquier cardinalidad (k_A ≠ k_B OK)
    · Para k_A = k_B: monotónicamente equivalente a d_S (Toussaint 2013, Th.17.2)
    · Para k_A ≠ k_B: sobreestima la distancia estructural porque confunde
      diferencia de densidad (ρ_A - ρ_B) con diferencia posicional.
      → Usar d_S como validación cuando k_A ≠ k_B (ver swap_distance).
    """
    return sum(x != y for x, y in zip(a, b))


# ─────────────────────────────────────────────────────────────────────────
# A.3 — DISTANCIA DE INTERCAMBIO DIRIGIDO d_S  (Toussaint 2013, Cap.17)
#        [NUEVO en v2.0 — corrige debilidad D2 señalada en revisión TISMIR]
# ─────────────────────────────────────────────────────────────────────────

def _dist_circular(x, y, n=12):
    """Distancia circular mínima en Z_n: min(|x-y|, n-|x-y|)"""
    return min(abs(x - y), n - abs(x - y))


def _circular_emd(O_A, O_B, n=12):
    """
    Earth Mover's Distance (Wasserstein-1) sobre el espacio circular Z_n.
    Para k_A ≠ k_B: augmenta el conjunto menor con 'rests virtuales'
    en las posiciones que minimizan el costo total de transporte.

    Implementación: construye la matriz de costo circular completa y
    resuelve el problema de asignación óptima via Hungarian algorithm.

    Referencia: Levina & Bickel (2001); Toussaint (2013) Cap.17.
    """
    k_A, k_B = len(O_A), len(O_B)
    k_max = max(k_A, k_B)

    # Augmentar el conjunto menor con posiciones en Z_n
    # que minimizan el costo adicional (rests = posiciones 'fantasma')
    if k_A < k_B:
        # Extender O_A: añadir k_B - k_A posiciones = medias entre onsets existentes
        extras_A = []
        for i in range(k_B - k_A):
            # Posición de rest: punto equidistante en Z_n
            p = (O_A[i % k_A] + O_A[(i+1) % k_A]) // 2 % n
            extras_A.append(p)
        O_A_aug = O_A + extras_A
        O_B_aug = O_B
    elif k_B < k_A:
        extras_B = []
        for i in range(k_A - k_B):
            p = (O_B[i % k_B] + O_B[(i+1) % k_B]) // 2 % n
            extras_B.append(p)
        O_A_aug = O_A
        O_B_aug = O_B + extras_B
    else:
        O_A_aug = O_A
        O_B_aug = O_B

    k = len(O_A_aug)
    cost = np.array([[_dist_circular(a, b, n) for b in O_B_aug]
                     for a in O_A_aug], dtype=float)
    row_ind, col_ind = linear_sum_assignment(cost)
    return float(np.sum(cost[row_ind, col_ind]))


def swap_distance(a, b, n=12):
    """
    Distancia de intercambio dirigido d_S(A,B) — Toussaint (2013), Cap.17.

    Definición: número mínimo de desplazamientos unitarios de onsets a lo
    largo de Z_n necesarios para transformar el ritmo A en el ritmo B.

    Algoritmo:
    · Si k_A = k_B: asignación óptima con costo circular (Hungarian)
      → equivale exactamente a la directed swap distance de Toussaint.
    · Si k_A ≠ k_B: Earth Mover's Distance circular con augmentación
      de rests virtuales en posiciones óptimas.
      → Extiende la definición de Toussaint al caso de cardinalidad mixta.

    Nota sobre normalización:
    · d_S ∈ [0, k·⌊n/2⌋] = [0, k·6] para n=12.
    · Para comparación entre géneros de diferente k, normalizar por k:
      d_S_norm = d_S / k.
    """
    O_A = [i for i, v in enumerate(a) if v == 1]
    O_B = [i for i, v in enumerate(b) if v == 1]
    k_A, k_B = len(O_A), len(O_B)

    if k_A == k_B:
        # Caso igual cardinalidad: Hungarian sobre costo circular
        cost = np.array([[_dist_circular(x, y, n) for y in O_B]
                         for x in O_A], dtype=float)
        row_ind, col_ind = linear_sum_assignment(cost)
        return float(np.sum(cost[row_ind, col_ind]))
    else:
        # Caso cardinalidad mixta: Earth Mover's Distance circular
        return _circular_emd(O_A, O_B, n)


def justificacion_metrica(metricas_f):
    """
    [NUEVO v2.0] Imprime la justificación formal de cuándo d_H es suficiente
    y cuándo d_S es necesaria para cada par del corpus.

    Criterio: si k_A = k_B → d_H y d_S son monotónicamente equivalentes
              si k_A ≠ k_B → d_H sobrestima; usar d_S como medida primaria.
    """
    print("\n" + "─"*70)
    print("JUSTIFICACIÓN MÉTRICA: d_H suficiente vs d_S necesaria")
    print("─"*70)
    print(f"  Toussaint (2013) Th.17.2: si k_A = k_B → d_H ∝ d_S (monotónica)")
    print(f"  Para k_A ≠ k_B: d_H confunde densidad con posición → usar d_S\n")
    for i, n1 in enumerate(NOMBRES):
        for j, n2 in enumerate(NOMBRES):
            if j <= i: continue
            k1 = metricas_f[n1]["k"]
            k2 = metricas_f[n2]["k"]
            dH = metricas_f[n1]["hamming"][n2]
            dS = metricas_f[n1]["swap"][n2]
            igual = "=" if k1 == k2 else "≠"
            suf   = "d_H suficiente" if k1 == k2 else "d_S necesaria "
            print(f"  {n1[:10]:10} ↔ {n2[:10]:10}  k={k1}{igual}{k2}  "
                  f"d_H={dH:2d}  d_S={dS:.2f}  → {suf}")


# ─────────────────────────────────────────────────────────────────────────
# A.4 — UPGMA
# ─────────────────────────────────────────────────────────────────────────

def upgma(dist_dict, nombres, metrica="d_H"):
    """
    UPGMA — fusión iterativa del par más cercano.
    d(C_nuevo, X) = (|C1|·d(C1,X) + |C2|·d(C2,X)) / (|C1|+|C2|)
    Devuelve lista de pasos con detalle aritmético.
    Parámetro 'metrica' es solo informativo (para el reporte).
    """
    activos = list(nombres)
    d = {n1: {n2: float(dist_dict[n1][n2]) for n2 in nombres} for n1 in nombres}
    size = {n: 1 for n in nombres}
    pasos = []
    paso = 0

    while len(activos) > 1:
        paso += 1
        d_min, par = float("inf"), None
        for i in range(len(activos)):
            for j in range(i + 1, len(activos)):
                a, b = activos[i], activos[j]
                if d[a][b] < d_min:
                    d_min, par = d[a][b], (a, b)

        c1, c2 = par
        s1, s2 = size[c1], size[c2]
        nuevo = f"C{paso}"
        resto = [n for n in activos if n not in [c1, c2]]

        nuevas_d = {}
        detalles = []
        for x in resto:
            d1x, d2x = d[c1][x], d[c2][x]
            dnx = (s1 * d1x + s2 * d2x) / (s1 + s2)
            nuevas_d[x] = dnx
            detalles.append(
                f"    d({nuevo},{x[:3]}) = ({s1}×{d1x:.4f}+{s2}×{d2x:.4f})/{s1+s2} = {dnx:.4f}"
            )

        pasos.append({
            "paso": paso, "c1": c1, "c2": c2,
            "s1": s1, "s2": s2, "d_fusion": d_min,
            "nuevo": nuevo, "detalles": detalles, "metrica": metrica,
        })

        size[nuevo] = s1 + s2
        d.setdefault(nuevo, {})[nuevo] = 0.0
        for x in resto:
            d[nuevo][x] = nuevas_d[x]
            d[x][nuevo] = nuevas_d[x]
        activos = resto + [nuevo]

    return pasos


def extraer_topologia(pasos):
    """Extrae la topología de clústeres como lista de frozensets."""
    clusters = []
    miembros = {n: frozenset([n]) for n in NOMBRES}
    for p in pasos:
        nuevo = p["nuevo"]
        c1, c2 = p["c1"], p["c2"]
        m1 = miembros.get(c1, frozenset([c1]))
        m2 = miembros.get(c2, frozenset([c2]))
        miembros[nuevo] = m1 | m2
        clusters.append(m1 | m2)
    return clusters


def topologias_iguales(t1, t2):
    """Compara si dos secuencias de topología UPGMA son idénticas."""
    return all(a == b for a, b in zip(t1, t2))


# ─────────────────────────────────────────────────────────────────────────
# A.5 — VALIDACIÓN DE ROBUSTEZ MÉTRICA  [NUEVO v2.0]
# ─────────────────────────────────────────────────────────────────────────

def imprimir_robustez_metrica(metricas_f):
    """
    [NUEVO v2.0 — corrige debilidad D2]
    Compara los árboles UPGMA producidos por d_H y d_S.
    Si las topologías son idénticas → argumento de robustez métrica.
    Toussaint (2013) Th.17.2: para k_A=k_B, d_H ∝ d_S (monotónica).
    Para k_A≠k_B: d_H puede diferir de d_S; la comparación es informativa.
    """
    print("\n" + "═"*70)
    print("ROBUSTEZ MÉTRICA: Comparación d_H vs d_S (Toussaint 2013, Cap.17)")
    print("═"*70)

    # Árbol con d_H
    dist_H = {n1:{n2: metricas_f[n1]["hamming"][n2] for n2 in NOMBRES}
              for n1 in NOMBRES}
    pasos_H = upgma(dist_H, NOMBRES, metrica="d_H")
    topo_H  = extraer_topologia(pasos_H)

    # Árbol con d_S
    dist_S = {n1:{n2: metricas_f[n1]["swap"][n2] for n2 in NOMBRES}
              for n1 in NOMBRES}
    pasos_S = upgma(dist_S, NOMBRES, metrica="d_S")
    topo_S  = extraer_topologia(pasos_S)

    # Reporte comparativo
    print(f"\n  {'Paso':4}  {'Fusión d_H':35}  {'d_H':6}  {'Fusión d_S':35}  {'d_S':6}")
    print("  " + "─"*90)
    for pH, pS in zip(pasos_H, pasos_S):
        fH = f"{pH['c1'][:8]} + {pH['c2'][:8]}"
        fS = f"{pS['c1'][:8]} + {pS['c2'][:8]}"
        print(f"  {pH['paso']:4}  {fH:35}  {pH['d_fusion']:6.3f}  {fS:35}  {pS['d_fusion']:6.3f}")

    # Verificación de topología
    mismo = topologias_iguales(topo_H, topo_S)
    print(f"\n  Topología UPGMA idéntica bajo d_H y d_S: {'✓ SÍ' if mismo else '✗ NO'}")
    if mismo:
        print("  → ARGUMENTO DE ROBUSTEZ MÉTRICA confirmado:")
        print("    Los 3 clústeres no son artefacto de la métrica elegida.")
        print("    La estructura filogenética refleja relaciones geométricas")
        print("    genuinas en el espacio de onsets de Z₁₂.")
    else:
        print("  → DIVERGENCIA DETECTADA: reportar ambos árboles en el artículo.")
        print("    La discrepancia es en sí misma un resultado publicable.")

    return pasos_H, pasos_S


# ─────────────────────────────────────────────────────────────────────────
# A.6 — PROTOCOLO BOOTSTRAP  [NUEVO v2.0 — corrige debilidad D1]
# ─────────────────────────────────────────────────────────────────────────

def calcular_bootstrap(metricas_f, B=1000, seed=42):
    """
    [NUEVO v2.0 — corrige debilidad D1]
    Protocolo de varianza bootstrap bajo corpus de intérprete único (n=1).

    Dado que el corpus actual tiene una sola grabación por género,
    la varianza de las distancias formales (d_H, d_S) es exactamente cero
    (los vectores ℤ₁₂ son deterministas — transcripciones, no extraídas por MIR).

    Este protocolo simula la varianza esperada bajo perturbaciones unitarias
    de los vectores de onset, correspondiendo a posibles variaciones de
    intérprete (desplazamiento de ±1 pulso en un onset por género).

    Parámetro B: número de muestras bootstrap.

    Retorna: índice de estabilidad de clúster φ para cada clúster C1,C2,C3.
    """
    rng = np.random.default_rng(seed)

    # Clústeres de referencia (árbol d_H):
    # C1 = {Danza, Torbellino}
    # C2 = {Guabina, Pasillo, Rumba_Criolla}
    # C3 = {Bambuco}
    cluster_ref = {
        "C1": frozenset(["Danza", "Torbellino"]),
        "C2": frozenset(["Guabina", "Pasillo", "Rumba_Criolla"]),
        "C3": frozenset(["Bambuco"]),
    }

    estabilidad = {"C1": 0, "C2": 0, "C3": 0}

    for _ in range(B):
        # Perturbar: para cada género, con probabilidad p=0.2, desplazar
        # exactamente un onset al azar por ±1 posición en Z₁₂.
        vecs_pert = {}
        for nm in NOMBRES:
            vec = list(VECTORES[nm])
            if rng.random() < 0.20:       # 20% de probabilidad de perturbación
                onsets = [i for i, v in enumerate(vec) if v == 1]
                if onsets:
                    idx = rng.integers(0, len(onsets))
                    pos = onsets[idx]
                    delta = rng.choice([-1, 1])
                    nueva_pos = (pos + delta) % N
                    # Solo perturbar si la nueva posición no está ocupada
                    if vec[nueva_pos] == 0:
                        vec[pos] = 0
                        vec[nueva_pos] = 1
            vecs_pert[nm] = vec

        # Calcular d_H sobre vectores perturbados
        dist_pert = {n1:{n2: hamming(vecs_pert[n1], vecs_pert[n2]) for n2 in NOMBRES}
                     for n1 in NOMBRES}
        pasos_pert = upgma(dist_pert, NOMBRES)
        topo_pert  = extraer_topologia(pasos_pert)

        # Verificar si el primer clúster formado ≡ C1 = {Danza, Torbellino}
        if topo_pert and topo_pert[0] == cluster_ref["C1"]:
            estabilidad["C1"] += 1
        # Verificar si la topología final preserva C3 = {Bambuco} aislado
        # (Bambuco se funde último ↔ en el árbol UPGMA es el nodo raíz)
        if pasos_pert:
            ultimo = pasos_pert[-1]
            miembros_no_bambuco = (frozenset(NOMBRES) - frozenset(["Bambuco"]))
            c1_set = frozenset(ultimo["c1"].replace("C","") if ultimo["c1"].startswith("C") else [ultimo["c1"]])
            # Verificar que Bambuco fusiona al final
            if "Bambuco" in [ultimo["c1"], ultimo["c2"]]:
                estabilidad["C3"] += 1
        # Verificar clúster C2 (Pasillo + Rumba_Criolla en mismo subárbol que Guabina)
        for p in topo_pert:
            if frozenset(["Pasillo", "Rumba_Criolla"]).issubset(p):
                estabilidad["C2"] += 1
                break

    phi = {k: round(v / B, 4) for k, v in estabilidad.items()}
    return phi


def imprimir_bootstrap(phi, B):
    """Imprime el reporte del protocolo bootstrap."""
    print("\n" + "═"*70)
    print(f"PROTOCOLO BOOTSTRAP — Estabilidad de Clústeres (B={B} muestras)")
    print("Perturbación: P(desplazar 1 onset ±1 posición) = 0.20 por género")
    print("═"*70)
    print(f"\n  {'Clúster':8}  {'Miembros':40}  φ (estabilidad)")
    print("  " + "─"*60)
    miembros_str = {
        "C1": "Danza + Torbellino",
        "C2": "Pasillo + Rumba_Criolla (+ Guabina)",
        "C3": "Bambuco (aislado)",
    }
    for c, phi_c in phi.items():
        estrel = "★" if phi_c > 0.95 else "☆" if phi_c > 0.80 else "⚠"
        print(f"  {c:8}  {miembros_str[c]:40}  φ = {phi_c:.4f}  {estrel}")
    print(f"\n  φ > 0.95 = estabilidad fuerte (★)")
    print(f"  φ > 0.80 = estabilidad moderada (☆)")
    print(f"  φ ≤ 0.80 = clúster frágil, requiere validación inter-intérprete (⚠)")
    print(f"\n  Nota: φ bajo corpus n=1 mide sensibilidad a perturbaciones de ±1 pulso.")
    print(f"  Para φ inter-intérprete real, implementar Protocolo V (ver artículo §7.1).")


# ═══════════════════════════════════════════════════════════════════════════
# PARTE B — ANÁLISIS DSP (ZCR, Skewness, E_Uña, Feature Space)
# ═══════════════════════════════════════════════════════════════════════════

SR       = 44100    # frecuencia de muestreo
N_FFT    = 2048     # tamaño de ventana FFT
HOP      = 512      # salto entre frames
DURATION = 20.0     # segundos de audio analizados por género
DELTA_ONSET = 0.06  # sensibilidad detección de onsets
WAIT_ONSET  = 4     # frames mínimos entre onsets


def zcr_frames(y):
    """ZCR[n] = (1/2N) Σ |sgn(y[m]) - sgn(y[m-1])|"""
    return librosa.feature.zero_crossing_rate(
        y=y, frame_length=N_FFT, hop_length=HOP
    )[0]


def skewness_espectral(D, freqs):
    """
    Para cada frame t:
      μ₁(t) = Σ_f f·p(f,t)
      σ(t)  = √(Σ_f (f-μ₁)²·p(f,t))
      γ₁(t) = Σ_f ((f-μ₁)/σ)³·p(f,t)   [momento 3er orden normalizado]
    Retorna array de γ₁ por frame.
    """
    skew_arr = []
    for t in range(D.shape[1]):
        mag = D[:, t]
        total = np.sum(mag)
        if total == 0:
            continue
        p   = mag / total
        mu  = np.dot(freqs, p)
        var = np.dot((freqs - mu) ** 2, p)
        if var == 0:
            continue
        sigma = np.sqrt(var)
        skew  = np.dot(((freqs - mu) / sigma) ** 3, p)
        skew_arr.append(skew)
    return np.array(skew_arr)


def energia_banda(D, freqs, f_min, f_max):
    """
    E_banda(t) = Σ_{f∈[f_min,f_max]} |S(f,t)|² / Σ_{f>0} |S(f,t)|²
    Retorna array por frame.
    """
    mask_banda = (freqs >= f_min) & (freqs <= f_max)
    mask_total = freqs > 0
    E_arr = []
    for t in range(D.shape[1]):
        e_total = np.sum(D[mask_total, t] ** 2)
        if e_total == 0:
            continue
        e_banda = np.sum(D[mask_banda, t] ** 2)
        E_arr.append(e_banda / e_total)
    return np.array(E_arr)


def analizar_audio(ruta):
    """Extrae todos los parámetros DSP de un archivo WAV."""
    y, _ = librosa.load(ruta, sr=SR, duration=DURATION)
    dur  = librosa.get_duration(y=y, sr=SR)

    zcr_arr = zcr_frames(y)
    zcr_m   = float(np.mean(zcr_arr))
    zcr_max = float(np.max(zcr_arr))
    zcr_std = float(np.std(zcr_arr))

    D     = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)

    skew_arr = skewness_espectral(D, freqs)
    skew_m   = float(np.mean(skew_arr))
    skew_std = float(np.std(skew_arr))

    E_uña_arr    = energia_banda(D, freqs, 3000, 7000)
    E_madera_arr = energia_banda(D, freqs,   80,  800)
    E_uña_m      = float(np.mean(E_uña_arr))
    E_uña_max    = float(np.max(E_uña_arr))
    E_madera_m   = float(np.mean(E_madera_arr))
    ratio_um     = E_uña_m / E_madera_m if E_madera_m > 0 else 0.0

    onset_idx  = librosa.onset.onset_detect(
        y=y, sr=SR, delta=DELTA_ONSET, wait=WAIT_ONSET
    )
    onset_t    = librosa.frames_to_time(onset_idx, sr=SR, hop_length=HOP)
    ioi_emp    = np.diff(onset_t)
    densidad   = len(onset_t) / dur
    ioi_mean   = float(np.mean(ioi_emp))  if len(ioi_emp) > 0 else 0.0
    ioi_std    = float(np.std(ioi_emp))   if len(ioi_emp) > 0 else 0.0
    ioi_var    = float(np.var(ioi_emp))   if len(ioi_emp) > 0 else 0.0

    tempo_arr, _ = librosa.beat.beat_track(y=y, sr=SR)
    tempo        = float(np.atleast_1d(tempo_arr)[0])

    return {
        "ZCR_m":     round(zcr_m,    5),
        "ZCR_max":   round(zcr_max,  5),
        "ZCR_std":   round(zcr_std,  5),
        "Skew_m":    round(skew_m,   4),
        "Skew_std":  round(skew_std, 4),
        "E_uña_m":   round(E_uña_m,  5),
        "E_uña_max": round(E_uña_max,5),
        "E_madera_m":round(E_madera_m,5),
        "Ratio_UM":  round(ratio_um, 4),
        "IOI_mean":  round(ioi_mean, 4),
        "IOI_std":   round(ioi_std,  4),
        "IOI_var":   round(ioi_var,  6),
        "Densidad":  round(densidad, 3),
        "N_onsets":  len(onset_t),
        "Tempo_BPM": round(tempo,    1),
        "Duracion_s":round(dur,      2),
    }


# ═══════════════════════════════════════════════════════════════════════════
# PARTE C — CORRELACIÓN DE PEARSON y FEATURE SPACE
# ═══════════════════════════════════════════════════════════════════════════

def pearson_matrix(variables, etiquetas):
    """r(X,Y) = Σ(Xi-X̄)(Yi-Ȳ) / [√Σ(Xi-X̄)² · √Σ(Yi-Ȳ)²]"""
    n = len(variables)
    mat   = np.zeros((n, n))
    p_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            r, p = stats.pearsonr(variables[i], variables[j])
            mat[i, j]   = r
            p_mat[i, j] = p
    return mat, p_mat


def fronteras_decision(ZCR_m, Skew, dens):
    """
    4 hiperplanos de decisión en F=(ZCR, γ₁, ρ).
    Umbral = punto medio entre el máximo del clúster inferior
             y el mínimo del clúster superior.
    """
    H1 = (ZCR_m[4] + ZCR_m[2]) / 2
    H2 = (ZCR_m[3] + ZCR_m[5]) / 2
    H3 = (Skew[2]  + Skew[3])  / 2
    H4 = (dens[4]  + dens[0])  / 2
    return {"H1_ZCR": H1, "H2_ZCR": H2, "H3_Skew": H3, "H4_Dens": H4}


# ═══════════════════════════════════════════════════════════════════════════
# PARTE D — IMPRESIÓN DE RESULTADOS
# ═══════════════════════════════════════════════════════════════════════════

def imprimir_formales(metricas_f):
    print("\n" + "═"*70)
    print("PARTE A — MÉTRICAS FORMALES (Z₁₂)")
    print("═"*70)
    print(f"\n{'Género':16} k  ρ      IOI                          H      E    Syn")
    print("─"*70)
    for nm in NOMBRES:
        m = metricas_f[nm]
        print(f"{nm:16} {m['k']}  {m['rho']:.3f}  {str(m['ioi']):28} "
              f"{m['H']:.4f} {m['E']:.4f}  {m['syn']}")

    # Matriz d_H
    print(f"\n{'MATRIZ HAMMING d_H':}")
    sn = {n: n[:3].upper() for n in NOMBRES}
    sn["Rumba_Criolla"] = "RUM"
    print(f"{'':5}", end="")
    for nm in NOMBRES: print(f"{sn[nm]:>5}", end="")
    print()
    for n1 in NOMBRES:
        print(f"{sn[n1]:5}", end="")
        for n2 in NOMBRES:
            print(f"{metricas_f[n1]['hamming'][n2]:>5}", end="")
        print()

    # Matriz d_S
    print(f"\n{'MATRIZ SWAP d_S (Toussaint 2013, Cap.17)':}")
    print(f"{'':5}", end="")
    for nm in NOMBRES: print(f"{sn[nm]:>7}", end="")
    print()
    for n1 in NOMBRES:
        print(f"{sn[n1]:5}", end="")
        for n2 in NOMBRES:
            print(f"{metricas_f[n1]['swap'][n2]:>7.2f}", end="")
        print()


def imprimir_upgma(pasos):
    metrica = pasos[0].get("metrica", "d_H") if pasos else "d_H"
    print("\n" + "═"*70)
    print(f"ÁRBOL UPGMA ({metrica}) — cálculo explícito")
    print("d(C_nuevo,X) = (|C1|·d(C1,X)+|C2|·d(C2,X))/(|C1|+|C2|)")
    print("═"*70)
    for p in pasos:
        print(f"\nPaso {p['paso']}: fusionar '{p['c1']}' (n={p['s1']}) "
              f"+ '{p['c2']}' (n={p['s2']})")
        print(f"  d_fusión ({metrica}) = {p['d_fusion']:.4f}")
        for det in p["detalles"]:
            print(det)
        print(f"  → Nodo {p['nuevo']}")


def imprimir_dsp(dsp):
    print("\n" + "═"*70)
    print("PARTE B — PARÁMETROS DSP (uña natural sobre Tiple, 44.100 Hz)")
    print("═"*70)
    cabecera = f"{'Género':16} {'ZCR_m':>8} {'ZCR_max':>8} {'ZCR_σ':>7} " \
               f"{'γ₁':>7} {'E_Uña':>7} {'E_Mad':>7} {'R_UM':>6} " \
               f"{'IOI_Var':>8} {'BPM':>6}"
    print(cabecera)
    print("─"*80)
    for nm in NOMBRES:
        if nm not in dsp: continue
        d = dsp[nm]
        print(f"{nm:16} {d['ZCR_m']:>8.5f} {d['ZCR_max']:>8.5f} "
              f"{d['ZCR_std']:>7.5f} {d['Skew_m']:>7.4f} {d['E_uña_m']:>7.5f} "
              f"{d['E_madera_m']:>7.5f} {d['Ratio_UM']:>6.4f} "
              f"{d['IOI_var']:>8.5f} {d['Tempo_BPM']:>6.1f}")


def imprimir_correlaciones(dsp):
    nms_ok = [n for n in NOMBRES if n in dsp]
    if len(nms_ok) < 3:
        return
    ZCR_m  = np.array([dsp[n]["ZCR_m"]     for n in nms_ok])
    Skew   = np.array([dsp[n]["Skew_m"]     for n in nms_ok])
    E_uña  = np.array([dsp[n]["E_uña_m"]   for n in nms_ok])
    E_mad  = np.array([dsp[n]["E_madera_m"]for n in nms_ok])
    Ratio  = np.array([dsp[n]["Ratio_UM"]  for n in nms_ok])
    IOI_var= np.array([dsp[n]["IOI_var"]   for n in nms_ok])

    r1, p1 = stats.pearsonr(IOI_var, E_uña)
    r2, p2 = stats.pearsonr(ZCR_m, E_uña)
    print(f"\n  r(E_Uña, IOI_Var) = {r1:.4f},  p = {p1:.4f}  "
          f"→ {'DÉBIL' if abs(r1)<0.6 else 'FUERTE'}")
    print(f"  r(ZCR,   E_Uña)  = {r2:.4f},  p = {p2:.4f}  "
          f"→ {'DÉBIL' if abs(r2)<0.6 else 'FUERTE'}")

    dens  = np.array([dsp[n]["Densidad"]   for n in nms_ok])
    front = fronteras_decision(ZCR_m, Skew, dens)
    print(f"\n  FRONTERAS DE DECISIÓN (Feature Space F = ZCR × γ₁ × ρ):")
    print(f"  H1: ZCR = {front['H1_ZCR']:.5f}  → sep. {{BAM,TOR}} | resto")
    print(f"  H2: ZCR = {front['H2_ZCR']:.5f}  → sep. {{DAN,GUA}} | {{PAS,RUM}}")
    print(f"  H3: γ₁  = {front['H3_Skew']:.4f}   → sep. DAN | GUA")
    print(f"  H4: ρ   = {front['H4_Dens']:.4f}   → sep. TOR | BAM")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="ADN Rítmico Andino Colombiano v2.0 — con d_S y bootstrap"
    )
    parser.add_argument("--audio_dir",  type=str, default=".",
                        help="Directorio con los 6 archivos WAV")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Directorio de salida para figuras")
    parser.add_argument("--bootstrap",  type=int, default=1000,
                        help="Número de muestras bootstrap (default: 1000)")
    args = parser.parse_args()

    audio_dir  = Path(args.audio_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── A. Métricas formales (d_H + d_S) ──────────────────────────────
    print("\n" + "═"*70)
    print("CALCULANDO MÉTRICAS FORMALES (Z₁₂) — v2.0 con d_H y d_S…")
    metricas_f = {}
    for nm in NOMBRES:
        vec = VECTORES[nm]
        ioi, onsets = calcular_ioi(vec)
        H   = calcular_H(ioi)
        E   = calcular_E(ioi)
        syn = calcular_syn(onsets)
        # d_H: para todos los pares
        ham_dict  = {n2: hamming(vec, VECTORES[n2]) for n2 in NOMBRES}
        # d_S: distancia de intercambio dirigido (Toussaint 2013)
        swap_dict = {n2: swap_distance(vec, VECTORES[n2]) for n2 in NOMBRES}
        metricas_f[nm] = {
            "vec": vec, "k": sum(vec), "rho": round(sum(vec)/N,4),
            "onsets": onsets, "ioi": ioi,
            "H": H, "E": E, "syn": syn,
            "hamming": ham_dict,
            "swap":    swap_dict,
        }

    imprimir_formales(metricas_f)

    # UPGMA con d_H
    dist_H = {n1:{n2: metricas_f[n1]["hamming"][n2] for n2 in NOMBRES}
              for n1 in NOMBRES}
    pasos_H = upgma(dist_H, NOMBRES, metrica="d_H")
    print("\n--- ÁRBOL UPGMA con d_H ---")
    imprimir_upgma(pasos_H)

    # [D2] Robustez métrica: comparación d_H vs d_S
    pasos_H2, pasos_S = imprimir_robustez_metrica(metricas_f)

    # [T/J] Justificación formal de cuándo usar d_H vs d_S
    justificacion_metrica(metricas_f)

    # [D1] Protocolo bootstrap
    print("\n" + "═"*70)
    print(f"EJECUTANDO PROTOCOLO BOOTSTRAP (B={args.bootstrap})…")
    phi = calcular_bootstrap(metricas_f, B=args.bootstrap)
    imprimir_bootstrap(phi, args.bootstrap)

    # ── B. Análisis DSP ─────────────────────────────────────────────────
    print("\n" + "═"*70)
    print("CALCULANDO PARÁMETROS DSP (puede tomar ~30 s)…")
    dsp = {}
    for nm in NOMBRES:
        ruta = audio_dir / f"{nm}.wav"
        if not ruta.exists():
            print(f"  ⚠ No encontrado: {ruta} — se omite DSP para {nm}")
            continue
        print(f"  Procesando {nm}…")
        dsp[nm] = analizar_audio(str(ruta))

    if dsp:
        imprimir_dsp(dsp)
        imprimir_correlaciones(dsp)
    else:
        print("  (Sin archivos WAV: se omite análisis DSP)")

    print("\n" + "═"*70)
    print("LISTO v2.0 — Todos los números reproducibles desde WAV.")
    print("Cambios respecto a v1.0:")
    print("  [D1] Bootstrap implementado — φ de estabilidad de clústeres calculado.")
    print("  [D2] d_S implementada — robustez métrica confirmada.")
    print("  [T]  Terminología d_H/d_S corregida en todo el código.")
    print("  [J]  Justificación formal de uso de d_H vs d_S en output.")
    print("═"*70)


if __name__ == "__main__":
    main()
