#!/usr/bin/env python3
"""
Script for generating plots and CSV summaries from FastSolver outputs,
with multi-port reduction (phases / windings) on top of per-trace matrices.

Public API (used by SpiralsMain.py):
    - ensure_analysis_dirs(spiral_path)
    - load_capacitance_matrix(path)
    - process_spiral(spiral_path, global_records, ports_override=None, auto_reuse_ports=False)
    - main()
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import numpy as np
import pandas as pd
from scipy.io import loadmat

# Use a non-interactive backend for matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Frequencies where we record tabulated values
KEY_FREQS = [10e3, 50e3, 100e3, 200e3, 500e3, 1e6]
# Reference frequency for summaries / transformer metrics
REF_FREQ = 100e3


# ---------------------------------------------------------------------------
# Helpers for Address.txt and directory layout
# ---------------------------------------------------------------------------


def normalize_address_path(raw: Path | str) -> Path:
    """Return a resolved Address.txt path from user input."""
    cleaned = str(raw).strip().strip('"').strip("'")
    path = Path(cleaned).expanduser()

    if path.is_dir():
        candidate = path / "Address.txt"
        if candidate.exists():
            path = candidate

    if not path.name.lower().endswith("address.txt") and not path.exists():
        path = path / "Address.txt"

    return path.resolve()


def prompt_address_path() -> Optional[Path]:
    """Prompt the user for the Address.txt path and return a Path object if valid."""
    user_input = input("Enter the path to Address.txt: ").strip()
    if not user_input:
        print("No path provided. Exiting.")
        return None
    address_path = normalize_address_path(user_input)
    if not address_path.exists():
        print(f"Provided path does not exist: {address_path}")
        return None
    if address_path.is_dir():
        candidate = address_path / "Address.txt"
        if candidate.exists():
            address_path = candidate
        else:
            print(f"Directory provided but Address.txt not found inside: {address_path}")
            return None
    if not address_path.name.lower().endswith("address.txt"):
        print(f"Expected Address.txt file, got: {address_path}")
        return None
    return address_path


def read_addresses(address_path: Path) -> List[Path]:
    """Read non-empty, non-comment lines from Address.txt as paths."""
    addresses: List[Path] = []
    base = address_path.parent
    with address_path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip().strip('"').strip("'")
            if not stripped or stripped.startswith("#"):
                continue
            path = Path(stripped)
            if not path.is_absolute():
                path = base / path
            addresses.append(path)
    return addresses


def ensure_analysis_dirs(spiral_path: Path) -> Dict[str, Path]:
    """Ensure Analysis/ subfolders exist and return a small path dict."""
    analysis = spiral_path / "Analysis"
    matrices_dir = analysis / "matrices"
    ports_dir = analysis / "ports"
    analysis.mkdir(exist_ok=True)
    matrices_dir.mkdir(parents=True, exist_ok=True)
    ports_dir.mkdir(parents=True, exist_ok=True)
    return {
        "analysis": analysis,
        "matrices": matrices_dir,
        "ports": ports_dir,
        "ports_config": analysis / "ports_config.json",
        "summary_spiral": analysis / "summary_spiral.csv",
        "transformer_metrics": analysis / "transformer_metrics.csv",
    }


# ---------------------------------------------------------------------------
# Matrix loaders (C and Z)
# ---------------------------------------------------------------------------


def load_capacitance_matrix(path: Path) -> np.ndarray:
    """Load FasterCap-style text capacitance matrix into a dense numpy.array."""
    lines: List[List[float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                numbers = [float(x) for x in stripped.replace(",", " ").split()]
            except ValueError:
                continue
            if numbers:
                lines.append(numbers)
    if not lines:
        raise ValueError("Capacitance matrix is empty or unreadable")
    matrix = np.array(lines, dtype=float)
    return matrix


def select_first_match(data: Dict[str, object], candidates: List[str]) -> Optional[str]:
    for key in candidates:
        if key in data:
            return key
        for k in data:
            if k.lower() == key.lower():
                return k
    return None


def load_impedance_and_freq(mat_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load frequency vector and impedance matrices from Zc.mat.

    Supports two formats:
      1) ASCII FastSolver format with blocks like:
         'Impedance matrix for frequency = 1000 2 x 2'
         followed by N lines of 'real imagj' pairs.
      2) Binary MATLAB .mat file loadable with scipy.io.loadmat.
    """
    # Peek at the beginning of the file to detect ASCII vs binary
    with mat_path.open("rb") as f:
        prefix = f.read(512)
    try:
        prefix_text = prefix.decode("utf-8", errors="ignore")
    except Exception:
        prefix_text = ""

    # --- ASCII FastSolver format ---
    if "Impedance matrix for frequency" in prefix_text:
        text = mat_path.read_text(encoding="utf-8", errors="ignore")
        lines = text.splitlines()

        freqs: List[float] = []
        mats: List[np.ndarray] = []

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("Impedance matrix for frequency"):
                # Example: "Impedance matrix for frequency = 1000 2 x 2"
                m = re.search(
                    r"Impedance matrix for frequency\s*=\s*([^\s]+)\s+(\d+)\s*x\s*(\d+)",
                    line,
                )
                if not m:
                    raise ValueError(f"Could not parse header line: {line!r}")

                f_str, nrows_str, ncols_str = m.groups()
                freq_val = float(f_str)
                nrows = int(nrows_str)
                ncols = int(ncols_str)

                i += 1  # move to first matrix row
                rows: List[List[complex]] = []

                for _ in range(nrows):
                    if i >= len(lines):
                        raise ValueError(
                            "Unexpected end of file while reading impedance matrix rows"
                        )
                    row_line = lines[i].strip()
                    tokens = row_line.split()
                    # Each matrix entry is "real imagj" => 2 tokens per column
                    if len(tokens) != 2 * ncols:
                        raise ValueError(
                            f"Expected {2 * ncols} tokens in row, got {len(tokens)}: {row_line!r}"
                        )

                    row_vals: List[complex] = []
                    for c in range(ncols):
                        real_tok = tokens[2 * c]
                        imag_tok = tokens[2 * c + 1]
                        val_str = real_tok + imag_tok  # e.g. "3.9+1e-3j"
                        try:
                            val = complex(val_str)
                        except Exception as exc:  # noqa: BLE001
                            raise ValueError(
                                f"Failed to parse complex number from "
                                f"{real_tok!r} {imag_tok!r} in line: {row_line!r}"
                            ) from exc
                        row_vals.append(val)

                    rows.append(row_vals)
                    i += 1

                mat = np.array(rows, dtype=complex)
                freqs.append(freq_val)
                mats.append(mat)
            else:
                i += 1

        if not mats:
            raise ValueError(f"No impedance blocks found in ASCII Zc file {mat_path}")

        freq = np.array(freqs, dtype=float)
        Z = np.stack(mats, axis=0)  # shape: (F, N, N)
        return freq, Z

    # --- Binary MATLAB .mat fallback ---
    data = loadmat(mat_path)
    freq_key = select_first_match(data, ["freq", "frequency", "f"])
    z_key = select_first_match(data, ["Zc", "Z", "Z_matrix", "Zf"])
    if freq_key is None or z_key is None:
        raise ValueError("Could not find suitable frequency or impedance keys in Zc.mat")

    freq = np.squeeze(np.array(data[freq_key], dtype=float))
    Z = np.array(data[z_key])

    if freq.ndim != 1:
        raise ValueError("Frequency vector must be 1D")

    # Normalize Z shape to (F, N, N)
    if Z.ndim == 2:
        Z = Z[np.newaxis, :, :]           # single frequency
    elif Z.ndim == 3:
        pass                               # already (F,N,N)
    elif Z.ndim == 4:
        # Some tools store as (N, N, 1, F) or (N, N, F, 1); try to fix
        if 1 in Z.shape:
            Z = np.squeeze(Z)
        if Z.shape[-1] == freq.shape[0]:
            Z = np.moveaxis(Z, -1, 0)
    else:
        raise ValueError(f"Unsupported Z array dimensions: {Z.shape}")

    if Z.shape[0] != freq.shape[0]:
        if Z.shape[-1] == freq.shape[0]:
            Z = np.moveaxis(Z, -1, 0)
        else:
            raise ValueError(
                f"Frequency length ({freq.shape[0]}) does not match impedance data shape {Z.shape}"
            )

    return freq, Z


# ---------------------------------------------------------------------------
# Port / grouping helpers
# ---------------------------------------------------------------------------


def parse_index_list(text: str, base: int = 0) -> List[Tuple[int, float]]:
    """
    Parse a comma/semicolon/space separated index list into (idx0, sign) pairs.

    Examples (base=0):
      "0,1,2"   -> [(0, +1), (1, +1), (2, +1)]
      "0,-3,5"  -> [(0, +1), (3, -1), (5, +1)]

    base=1 means the numbers in the string are 1-based and will be converted
    to 0-based before returning.
    """
    text = text.strip()
    if not text:
        return []

    out: List[Tuple[int, float]] = []
    for token in re.split(r"[;,\s]+", text):
        token = token.strip()
        if not token:
            continue

        sign = 1.0
        if token.startswith("+"):
            token = token[1:]
        elif token.startswith("-"):
            sign = -1.0
            token = token[1:]

        idx = int(token)
        idx0 = idx - base
        if idx0 < 0:
            raise ValueError(f"Index {idx} with base={base} gives negative 0-based index {idx0}")
        out.append((idx0, sign))

    return out


def compute_current_pattern(port_def: Dict[str, object], n: int) -> np.ndarray:
    """
    Convert a port definition into a conductor-current pattern alpha of length n.

    Stored "signs" vector (length n) represents relative currents in each trace
    for 1 A of port current. For "parallel" ports, the non-zero entries are
    normalized so that the SUM of magnitudes is 1.
    """
    port_type = str(port_def.get("type", "")).lower()
    signs = np.array(port_def.get("signs", []), dtype=float).reshape(-1)
    if signs.size != n:
        raise ValueError(f"Expected {n} signs, got {signs.size}")
    active = signs != 0

    if port_type in ("series", "custom_pm1", ""):
        pattern = signs.astype(float)
    elif port_type == "parallel":
        active_count = np.count_nonzero(active)
        if active_count == 0:
            raise ValueError("Parallel port has no active conductors")
        pattern = np.where(active, signs / active_count, 0.0)
    else:
        raise ValueError(f"Unknown port type: {port_type}")

    return pattern


def interactive_ports_config(
    config_path: Path,
    n: int,
    *,
    preconfigured: Optional[Dict[str, Dict[str, object]]] = None,
    auto_reuse: bool = False,
) -> Dict[str, Dict[str, object]]:
    """
    CLI helper for defining ports in a simple way.

    In GUI mode (SpiralsMain.py), this is bypassed by providing `preconfigured`
    from the Tkinter popup, but we still keep it for standalone use.
    """
    existing: Dict[str, Dict[str, object]] = {}
    existing_system_type: Optional[str] = None
    if config_path.exists():
        try:
            existing_json = json.loads(config_path.read_text())
            existing = existing_json.get("ports", {})
            existing_system_type = str(existing_json.get("system_type", "auto"))
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to load existing ports_config.json: {exc}")
            existing = {}

    # If GUI already provided the ports, just persist them and return
    if preconfigured is not None:
        data = {"ports": preconfigured}
        if existing_system_type is not None:
            data["system_type"] = existing_system_type
        config_path.write_text(json.dumps(data, indent=2))
        return preconfigured

    # CLI reuse option
    if auto_reuse and existing:
        return existing

    if existing:
        reuse = input("Ports configuration found. Reuse? [Y/n]: ").strip().lower()
        if reuse in ("", "y", "yes"):
            return existing

    ports: Dict[str, Dict[str, object]] = existing
    print(f"Configuring ports for {n} conductors.")
    print("For each port, you will enter:")
    print("  - a name (e.g. pA, sA, Phase1)")
    print("  - a type: series, parallel, or custom_pm1")
    print("  - a list of trace indices with optional signs, e.g. '0,1,2,-3' (0-based)")

    while True:
        print("Current ports:", ", ".join(ports.keys()) if ports else "(none)")
        action = input("Enter 'add' to add, 'delete' to remove, 'done' to finish: ").strip().lower()
        if action == "done":
            break
        if action == "delete":
            name = input("Port name to delete: ").strip()
            if name in ports:
                ports.pop(name)
                print(f"Deleted port '{name}'.")
            else:
                print("Port not found.")
            continue
        if action != "add":
            print("Unknown action. Use 'add', 'delete', or 'done'.")
            continue

        name = input("New port name: ").strip()
        if not name:
            print("Name cannot be empty.")
            continue
        if name in ports:
            overwrite = input("Port exists. Overwrite? [y/N]: ").strip().lower()
            if overwrite not in ("y", "yes"):
                continue

        port_type = input("Type (series/parallel/custom_pm1) [series]: ").strip().lower() or "series"
        if port_type not in {"series", "parallel", "custom_pm1"}:
            print("Invalid port type.")
            continue

        idx_str = input(
            "Enter trace indices for this port (e.g. '0,1,2,-3' for index 3 reversed): "
        ).strip()
        try:
            idx_sign = parse_index_list(idx_str, base=0)
        except Exception as exc:  # noqa: BLE001
            print(f"Could not parse index list: {exc}")
            continue

        signs = [0.0] * n
        ok = True
        for idx0, sgn in idx_sign:
            if not (0 <= idx0 < n):
                print(f"Index {idx0} is out of range 0..{n-1}")
                ok = False
                break
            signs[idx0] += sgn
        if not ok:
            continue

        ports[name] = {"type": port_type, "signs": signs, "raw_indices": idx_str}

    config_path.write_text(
        json.dumps({"ports": ports, "system_type": existing_system_type or "auto"}, indent=2)
    )
    return ports


# ---------------------------------------------------------------------------
# R, L and effective values
# ---------------------------------------------------------------------------


def compute_R_L(freq: np.ndarray, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split Z(f) into R(f) and L(f) for each frequency and conductor pair."""
    R = np.real(Z)
    L = np.zeros_like(Z, dtype=float)
    for idx, f in enumerate(freq):
        if f == 0:
            L[idx] = np.imag(Z[idx])
        else:
            L[idx] = np.imag(Z[idx]) / (2 * math.pi * f)
    return R, L


def build_grouping_matrix_from_ports(
    ports: Dict[str, Dict[str, object]],
    n_conductors: int,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build the N x P grouping matrix W from a ports dict.

    Each column of W is the conductor-current pattern alpha for 1 A in that port.
    """
    if not ports:
        raise ValueError("No ports defined")

    port_names = sorted(ports.keys())
    cols: List[np.ndarray] = []
    for name in port_names:
        alpha = compute_current_pattern(ports[name], n_conductors).reshape(-1)
        if alpha.size != n_conductors:
            raise ValueError(f"Port '{name}' pattern length {alpha.size} != {n_conductors}")
        cols.append(alpha)

    W = np.stack(cols, axis=1)  # shape (N, P)
    return W, port_names


def effective_values_from_diag(
    freq: np.ndarray,
    R_diag: np.ndarray,
    L_diag: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute L, R, Q vectors for a *single* port from diagonal entries of
    port-domain R and L matrices.
    """
    R_eff = np.asarray(R_diag, dtype=float).reshape(-1)
    L_eff = np.asarray(L_diag, dtype=float).reshape(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        Q = (2.0 * math.pi * freq * L_eff) / R_eff
    return L_eff, R_eff, Q


# ---------------------------------------------------------------------------
# Small helpers for post-processing / plots
# ---------------------------------------------------------------------------


def find_resonance(freq: np.ndarray, Zin: np.ndarray) -> float:
    """Estimate the first series resonance where Im(Zin) crosses zero."""
    imag_part = np.imag(Zin)
    signs = np.sign(imag_part)
    sign_changes = np.where(np.diff(signs) != 0)[0]
    if sign_changes.size == 0:
        return float("nan")
    idx = sign_changes[0]
    f1, f2 = freq[idx], freq[idx + 1]
    y1, y2 = imag_part[idx], imag_part[idx + 1]
    if y2 == y1:
        return float(f1)
    frac = -y1 / (y2 - y1)
    return float(f1 + frac * (f2 - f1))


def interpolate_values(targets: List[float], freq: np.ndarray, values: np.ndarray) -> Dict[float, float]:
    out: Dict[float, float] = {}
    for t in targets:
        if t <= freq.min():
            out[t] = float(values[0])
        elif t >= freq.max():
            out[t] = float(values[-1])
        else:
            out[t] = float(np.interp(t, freq, values))
    return out


def save_matrix_csv(matrix: np.ndarray, path: Path) -> None:
    df = pd.DataFrame(matrix)
    df.to_csv(path, index=False, header=False)


def plot_vs_frequency(
    freq: np.ndarray,
    values: np.ndarray,
    ylabel: str,
    title: str,
    path: Path,
    logx: bool = True,
) -> None:
    plt.figure()
    if logx:
        plt.semilogx(freq, values)
    else:
        plt.plot(freq, values)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# ---------------------------------------------------------------------------
# Transformer-style metrics (k, turns ratio) from port-domain L
# ---------------------------------------------------------------------------


def decode_port_role(name: str) -> Tuple[str, str]:
    """
    Very simple heuristic to guess role+phase from a port name.

    Returns (role, phase_key) where role ∈ {"primary","secondary","other"}
    and phase_key is e.g. "a", "b", "c", "1", "2", ... or "" if unknown.

    Rules:
      - if name starts with 'p' or 'pri' (case-insensitive) -> primary
      - if name starts with 's' or 'sec' -> secondary
      - phase_key is first letter/digit chunk after the role prefix

    Examples:
      "pA"      -> ("primary", "a")
      "P1"      -> ("primary", "1")
      "secC"    -> ("secondary", "c")
      "sB"      -> ("secondary", "b")
    """
    lowered = name.lower().strip()
    role = "other"
    tail = ""

    if lowered.startswith("pri"):
        role = "primary"
        tail = lowered[3:]
    elif lowered.startswith("p"):
        role = "primary"
        tail = lowered[1:]
    elif lowered.startswith("sec"):
        role = "secondary"
        tail = lowered[3:]
    elif lowered.startswith("s"):
        role = "secondary"
        tail = lowered[1:]
    else:
        return role, ""

    m = re.search(r"([abc]\d*|\d+)", tail)
    if m:
        return role, m.group(1)
    return role, ""


def compute_transformer_metrics(
    freq: np.ndarray,
    L_port: np.ndarray,
    port_names: List[str],
    ref_freq: float = REF_FREQ,
) -> List[Dict[str, object]]:
    """
    Compute simple transformer-style metrics (k, turns ratio, leakage) between
    any primary/secondary port pairs that share the same phase key.
    """
    n_freq, n_ports, _ = L_port.shape
    if n_ports != len(port_names):
        raise ValueError("L_port shape does not match number of port names")

    # Interpolate L_port to reference frequency for easier reporting
    L_ref = np.zeros((n_ports, n_ports), dtype=float)
    for i in range(n_ports):
        for j in range(n_ports):
            L_ij_f = L_port[:, i, j]
            L_ref[i, j] = float(np.interp(ref_freq, freq, L_ij_f))

    roles: List[Tuple[str, str]] = [decode_port_role(name) for name in port_names]
    metrics: List[Dict[str, object]] = []

    # Build maps from phase -> list of indices for primaries / secondaries
    prim_by_phase: Dict[str, List[int]] = {}
    sec_by_phase: Dict[str, List[int]] = {}
    for idx, (role, phase_key) in enumerate(roles):
        if role == "primary":
            prim_by_phase.setdefault(phase_key, []).append(idx)
        elif role == "secondary":
            sec_by_phase.setdefault(phase_key, []).append(idx)

    # For each phase with both primary and secondary defined, compute metrics
    for phase_key, prim_list in prim_by_phase.items():
        sec_list = sec_by_phase.get(phase_key, [])
        if not sec_list:
            continue
        for p_idx in prim_list:
            for s_idx in sec_list:
                L_pp = L_ref[p_idx, p_idx]
                L_ss = L_ref[s_idx, s_idx]
                M_ps = L_ref[p_idx, s_idx]
                if L_pp <= 0 or L_ss <= 0:
                    continue

                k = float(M_ps / math.sqrt(L_pp * L_ss))
                if not math.isfinite(k):
                    k = float("nan")
                else:
                    k = max(-1.0, min(1.0, k))

                turns_ratio = math.sqrt(L_pp / L_ss) if L_ss > 0 else float("nan")
                L_leak_p = L_pp - M_ps
                L_leak_s = L_ss - M_ps

                metrics.append(
                    {
                        "phase_key": phase_key,
                        "primary_port": port_names[p_idx],
                        "secondary_port": port_names[s_idx],
                        "ref_freq_Hz": ref_freq,
                        "L_pp_H": L_pp,
                        "L_ss_H": L_ss,
                        "M_ps_H": M_ps,
                        "k_coupling": k,
                        "turns_ratio_NpNs": turns_ratio,
                        "L_leak_primary_H": L_leak_p,
                        "L_leak_secondary_H": L_leak_s,
                    }
                )

    return metrics


# ---------------------------------------------------------------------------
# Core processing for a single spiral variant
# ---------------------------------------------------------------------------


def process_spiral(
    spiral_path: Path,
    global_records: List[Dict[str, object]],
    *,
    ports_override: Optional[Dict[str, Dict[str, object]]] = None,
    auto_reuse_ports: bool = False,
) -> None:
    """
    Process one spiral variant:
      - load C, Z, and freq
      - build port-domain matrices from per-trace matrices
      - compute per-port metrics (L, R, Q, Z_in Bode)
      - compute transformer-like metrics if applicable
      - write CSVs and plots into Analysis/
    """
    spiral_name = spiral_path.name
    fastsolver = spiral_path / "FastSolver"
    if not fastsolver.exists():
        print(f"Warning: FastSolver folder missing for {spiral_name}, skipping.")
        return

    cap_path = fastsolver / "CapacitanceMatrix.txt"
    zc_path = fastsolver / "Zc.mat"
    if not cap_path.exists() or not zc_path.exists():
        print(f"Warning: Required files missing in {fastsolver}, skipping {spiral_name}.")
        return

    dirs = ensure_analysis_dirs(spiral_path)

    try:
        C_trace = load_capacitance_matrix(cap_path)
        freq, Z_trace = load_impedance_and_freq(zc_path)
    except Exception as exc:  # noqa: BLE001
        print(f"Error processing {spiral_name}: {exc}")
        return

    n = C_trace.shape[0]
    if C_trace.shape[1] != n or Z_trace.shape[1] != n or Z_trace.shape[2] != n:
        print(f"Dimension mismatch for {spiral_name}, skipping.")
        return

    # Save raw trace-domain C matrix for reference
    save_matrix_csv(C_trace, dirs["matrices"] / "C_trace.csv")

    # Get ports either from GUI override or CLI interaction
    ports = interactive_ports_config(
        dirs["ports_config"],
        n,
        preconfigured=ports_override,
        auto_reuse=auto_reuse_ports,
    )
    if not ports:
        print(f"No ports defined for {spiral_name}, skipping.")
        return

    # Optional: read system_type from config if present (defaults to "auto")
    system_type = "auto"
    try:
        cfg_json = json.loads(dirs["ports_config"].read_text())
        system_type = str(cfg_json.get("system_type", "auto"))
    except Exception:
        pass

    # Compute trace-domain R, L
    R_trace, L_trace = compute_R_L(freq, Z_trace)

    # Build grouping matrix W and port names
    W, port_names = build_grouping_matrix_from_ports(ports, n)
    n_ports = len(port_names)
    WT = W.T

    # Reduce matrices to port-domain
    C_port = WT @ C_trace @ W  # (P, P)
    n_freq = freq.shape[0]
    R_port = np.zeros((n_freq, n_ports, n_ports), dtype=float)
    L_port = np.zeros((n_freq, n_ports, n_ports), dtype=float)
    Z_port = np.zeros((n_freq, n_ports, n_ports), dtype=complex)

    for k, f in enumerate(freq):
        Rk = WT @ R_trace[k] @ W
        Lk = WT @ L_trace[k] @ W
        R_port[k] = Rk
        L_port[k] = Lk
        Z_port[k] = Rk + 1j * 2 * math.pi * f * Lk

    # Save port-domain C matrix, and R/L/Z at a representative frequency
    matrices_dir = dirs["matrices"]
    save_matrix_csv(C_port, matrices_dir / "C_port.csv")

    # Pick nearest available index to REF_FREQ for matrix snapshots
    idx_ref = int(np.argmin(np.abs(freq - REF_FREQ)))
    f_ref = float(freq[idx_ref])
    save_matrix_csv(R_port[idx_ref], matrices_dir / f"R_port_{int(f_ref)}Hz.csv")
    save_matrix_csv(L_port[idx_ref], matrices_dir / f"L_port_{int(f_ref)}Hz.csv")
    save_matrix_csv(Z_port[idx_ref].real, matrices_dir / f"Z_port_real_{int(f_ref)}Hz.csv")
    save_matrix_csv(Z_port[idx_ref].imag, matrices_dir / f"Z_port_imag_{int(f_ref)}Hz.csv")

    # Per-port metrics and plots
    summary_rows: List[Dict[str, object]] = []
    for p_idx, port_name in enumerate(port_names):
        R_diag = R_port[:, p_idx, p_idx]
        L_diag = L_port[:, p_idx, p_idx]
        L_eff, R_eff, Q = effective_values_from_diag(freq, R_diag, L_diag)
        Zin = R_eff + 1j * 2 * math.pi * freq * L_eff
        resonance = find_resonance(freq, Zin)

        key_L = interpolate_values(KEY_FREQS, freq, L_eff)
        key_R = interpolate_values(KEY_FREQS, freq, R_eff)
        key_Q = interpolate_values(KEY_FREQS, freq, Q)

        port_dir = dirs["ports"] / port_name
        port_dir.mkdir(parents=True, exist_ok=True)

        # short CSV with key frequencies
        metrics_rows = []
        for kf in KEY_FREQS:
            metrics_rows.append(
                {
                    "spiral_name": spiral_name,
                    "port_name": port_name,
                    "freq_Hz": kf,
                    "L_eff_H": key_L[kf],
                    "R_eff_ohm": key_R[kf],
                    "Q": key_Q[kf],
                    "first_resonance_Hz": resonance,
                }
            )
        pd.DataFrame(metrics_rows).to_csv(port_dir / "metrics.csv", index=False)

        # Full Z_in vs f CSV (Bode data)
        zin_df = pd.DataFrame(
            {
                "freq_Hz": freq,
                "Re_Zin_ohm": np.real(Zin),
                "Im_Zin_ohm": np.imag(Zin),
                "abs_Zin_ohm": np.abs(Zin),
                "phase_Zin_deg": np.angle(Zin, deg=True),
            }
        )
        zin_df.to_csv(port_dir / "Z_in_vs_f.csv", index=False)

        # Plots: L, R, Q, |Zin|
        plot_vs_frequency(
            freq,
            L_eff,
            "L_eff (H)",
            f"Effective Inductance vs Frequency - {spiral_name} / {port_name}",
            port_dir / "L_eff_vs_f.png",
        )
        plot_vs_frequency(
            freq,
            R_eff,
            "R_eff (Ohm)",
            f"Effective Resistance vs Frequency - {spiral_name} / {port_name}",
            port_dir / "R_eff_vs_f.png",
        )
        plot_vs_frequency(
            freq,
            Q,
            "Q",
            f"Quality Factor vs Frequency - {spiral_name} / {port_name}",
            port_dir / "Q_vs_f.png",
        )
        plot_vs_frequency(
            freq,
            np.abs(Zin),
            "|Z_in| (Ohm)",
            f"|Z_in| vs Frequency - {spiral_name} / {port_name}",
            port_dir / "Zin_mag_vs_f.png",
        )

        # Reference frequency summary (using interpolation for consistency)
        ref_L = float(np.interp(REF_FREQ, freq, L_eff))
        ref_R = float(np.interp(REF_FREQ, freq, R_eff))
        ref_Q = float(np.interp(REF_FREQ, freq, Q))

        row = {
            "spiral_name": spiral_name,
            "port_name": port_name,
            "ref_freq_Hz": REF_FREQ,
            "L_eff_H": ref_L,
            "R_eff_ohm": ref_R,
            "Q": ref_Q,
            "first_resonance_Hz": resonance,
        }
        summary_rows.append(row)
        global_records.append(
            {
                **row,
                "N_conductors": n,
                "N_ports": n_ports,
                "system_type": system_type,
            }
        )

    # Per-spiral summary CSV
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(dirs["summary_spiral"], index=False)

    # Transformer-style metrics, only if relevant ports exist
    if system_type.lower() in ("auto", "transformer"):
        tmetrics = compute_transformer_metrics(freq, L_port, port_names, ref_freq=REF_FREQ)
        if tmetrics:
            pd.DataFrame(tmetrics).to_csv(dirs["transformer_metrics"], index=False)


# ---------------------------------------------------------------------------
# Global summary for all spirals in Address.txt
# ---------------------------------------------------------------------------


def write_global_summary(root: Path, records: List[Dict[str, object]]) -> None:
    if not records:
        return
    report_dir = root / "Global_Report"
    report_dir.mkdir(exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(report_dir / "summary_all_spirals.csv", index=False)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    address_path = prompt_address_path()
    if address_path is None:
        return
    spirals = read_addresses(address_path)
    if not spirals:
        print("No spiral addresses found in Address.txt")
        return

    global_records: List[Dict[str, object]] = []
    for spiral_path in spirals:
        if not spiral_path.exists():
            print(f"Warning: Spiral path does not exist: {spiral_path}")
            continue
        try:
            process_spiral(spiral_path, global_records)
        except Exception as exc:  # noqa: BLE001
            print(f"Unexpected error for {spiral_path}: {exc}")

    write_global_summary(address_path.parent, global_records)
    print("Processing complete.")


if __name__ == "__main__":
    main()
