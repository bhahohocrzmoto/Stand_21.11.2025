"""
Script for generating plots and CSV summaries from FastSolver outputs.
"""
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
import matplotlib.pyplot as plt


KEY_FREQS = [10e3, 50e3, 100e3, 200e3, 500e3, 1e6]
REF_FREQ = 100e3


def normalize_address_path(raw: Path | str) -> Path:
    """Return a resolved Address.txt path from user input.

    Accepts Windows-style paths and strips surrounding quotes. If a directory is
    provided, the ``Address.txt`` inside that directory is used when present.
    """

    cleaned = str(raw).strip().strip("\"").strip("'")
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
            stripped = line.strip().strip("\"").strip("'")
            if not stripped or stripped.startswith("#"):
                continue
            path = Path(stripped)
            if not path.is_absolute():
                path = base / path
            addresses.append(path)
    return addresses


def ensure_analysis_dirs(spiral_path: Path) -> Dict[str, Path]:
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
    }


def load_capacitance_matrix(path: Path) -> np.ndarray:
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
                        # Build something like "3.92056+9.90539e-05j"
                        val_str = real_tok + imag_tok
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
        # Single frequency: promote to (1, N, N)
        Z = Z[np.newaxis, :, :]
    elif Z.ndim == 3:
        # Already (F, N, N)
        pass
    elif Z.ndim == 4:
        # Some tools store as (N, N, 1, F) or (N, N, F, 1); try to fix
        if 1 in Z.shape:
            Z = np.squeeze(Z)
        # After squeeze, try to move last axis to front if it matches freq length
        if Z.shape[-1] == freq.shape[0]:
            Z = np.moveaxis(Z, -1, 0)
    else:
        raise ValueError(f"Unsupported Z array dimensions: {Z.shape}")

    if Z.shape[0] != freq.shape[0]:
        # Try swapping last axis
        if Z.shape[-1] == freq.shape[0]:
            Z = np.moveaxis(Z, -1, 0)
        else:
            raise ValueError(
                f"Frequency length ({freq.shape[0]}) does not match impedance data shape {Z.shape}"
            )

    return freq, Z



def compute_current_pattern(port_def: Dict[str, object], n: int) -> np.ndarray:
    port_type = str(port_def.get("type", "")).lower()
    signs = np.array(port_def.get("signs", []), dtype=float)
    if signs.size != n:
        raise ValueError(f"Expected {n} signs, got {signs.size}")
    active = signs != 0
    if port_type == "series" or port_type == "custom_pm1":
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
    existing: Dict[str, Dict[str, object]] = {}
    if config_path.exists():
        try:
            existing = json.loads(config_path.read_text())
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to load existing ports_config.json: {exc}")
            existing = {}
    if preconfigured is not None:
        config_path.write_text(json.dumps({"ports": preconfigured}, indent=2))
        return preconfigured
    if auto_reuse and existing:
        return existing.get("ports", {})
    if existing:
        reuse = input("Ports configuration found. Reuse? [Y/n]: ").strip().lower()
        if reuse in ("", "y", "yes"):
            return existing.get("ports", {})
    ports: Dict[str, Dict[str, object]] = existing.get("ports", {})
    print(f"Configuring ports for {n} conductors.")
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
        port_type = input("Type (series/parallel/custom_pm1): ").strip().lower()
        if port_type not in {"series", "parallel", "custom_pm1"}:
            print("Invalid port type.")
            continue
        signs_str = input(f"Enter {n} signs (+1 or -1) separated by space: ").strip()
        try:
            signs = [float(x) for x in signs_str.replace(",", " ").split()]
        except ValueError:
            print("Invalid signs list.")
            continue
        if len(signs) != n:
            print(f"Expected {n} signs, got {len(signs)}")
            continue
        ports[name] = {"type": port_type, "signs": signs}
    config_path.write_text(json.dumps({"ports": ports}, indent=2))
    return ports


def compute_R_L(freq: np.ndarray, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    R = np.real(Z)
    L = np.zeros_like(Z, dtype=float)
    for idx, f in enumerate(freq):
        if f == 0:
            L[idx] = np.imag(Z[idx])
        else:
            L[idx] = np.imag(Z[idx]) / (2 * math.pi * f)
    return R, L


def effective_values(
    freq: np.ndarray,
    R: np.ndarray,
    L: np.ndarray,
    pattern: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute effective L, R, Q for a given current pattern.

    Parameters
    ----------
    freq : (F,) array
        Frequency vector [Hz].
    R : (F, N, N) array
        Resistance matrices [ohm].
    L : (F, N, N) array
        Inductance matrices [H].
    pattern : (N,) or (N,1) array
        Current pattern alpha: conductor currents = alpha * I_port.

    Returns
    -------
    L_eff : (F,) array
        Effective inductance vs frequency [H].
    R_eff : (F,) array
        Effective series resistance vs frequency [ohm].
    Q : (F,) array
        Quality factor vs frequency (dimensionless).
    """
    # Make sure we have a 1D vector alpha
    alpha = np.asarray(pattern, dtype=float).reshape(-1)

    # Correct contraction: for each frequency f,
    # R_eff[f] = alpha^T R[f] alpha, same for L.
    R_eff = np.einsum("i,fij,j->f", alpha, R, alpha)
    L_eff = np.einsum("i,fij,j->f", alpha, L, alpha)

    with np.errstate(divide="ignore", invalid="ignore"):
        Q = (2.0 * math.pi * freq * L_eff) / R_eff

    return L_eff, R_eff, Q



def find_resonance(freq: np.ndarray, Zin: np.ndarray) -> float:
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


def plot_vs_frequency(freq: np.ndarray, values: np.ndarray, ylabel: str, title: str, path: Path, logx: bool = True) -> None:
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


def process_spiral(
    spiral_path: Path,
    global_records: List[Dict[str, object]],
    *,
    ports_override: Optional[Dict[str, Dict[str, object]]] = None,
    auto_reuse_ports: bool = False,
) -> None:
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
        C = load_capacitance_matrix(cap_path)
        freq, Z = load_impedance_and_freq(zc_path)
    except Exception as exc:  # noqa: BLE001
        print(f"Error processing {spiral_name}: {exc}")
        return
    n = C.shape[0]
    if C.shape[1] != n or Z.shape[1] != n or Z.shape[2] != n:
        print(f"Dimension mismatch for {spiral_name}, skipping.")
        return
    save_matrix_csv(C, dirs["matrices"] / "C_matrix.csv")
    ports = interactive_ports_config(
        dirs["ports_config"],
        n,
        preconfigured=ports_override,
        auto_reuse=auto_reuse_ports,
    )
    if not ports:
        print(f"No ports defined for {spiral_name}, skipping.")
        return
    R, L = compute_R_L(freq, Z)
    summary_rows: List[Dict[str, object]] = []
    for port_name, port_def in ports.items():
        try:
            pattern = compute_current_pattern(port_def, n)
        except Exception as exc:  # noqa: BLE001
            print(f"Invalid port '{port_name}' for {spiral_name}: {exc}")
            continue
        L_eff, R_eff, Q = effective_values(freq, R, L, pattern)
        Zin = R_eff + 1j * 2 * math.pi * freq * L_eff
        resonance = find_resonance(freq, Zin)
        key_L = interpolate_values(KEY_FREQS, freq, L_eff)
        key_R = interpolate_values(KEY_FREQS, freq, R_eff)
        key_Q = interpolate_values(KEY_FREQS, freq, Q)
        port_dir = dirs["ports"] / port_name
        port_dir.mkdir(parents=True, exist_ok=True)
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
        plot_vs_frequency(freq, L_eff, "L_eff (H)", f"Effective Inductance vs Frequency - {spiral_name} / {port_name}", port_dir / "L_eff_vs_f.png")
        plot_vs_frequency(freq, R_eff, "R_eff (Ohm)", f"Effective Resistance vs Frequency - {spiral_name} / {port_name}", port_dir / "R_eff_vs_f.png")
        plot_vs_frequency(freq, Q, "Q", f"Quality Factor vs Frequency - {spiral_name} / {port_name}", port_dir / "Q_vs_f.png")
        plot_vs_frequency(freq, np.abs(Zin), "|Z_in| (Ohm)", f"|Z_in| vs Frequency - {spiral_name} / {port_name}", port_dir / "Zin_mag_vs_f.png")
        ref_L = float(np.interp(REF_FREQ, freq, L_eff))
        ref_R = float(np.interp(REF_FREQ, freq, R_eff))
        ref_Q = float(np.interp(REF_FREQ, freq, Q))
        summary_rows.append(
            {
                "spiral_name": spiral_name,
                "port_name": port_name,
                "ref_freq_Hz": REF_FREQ,
                "L_eff_H": ref_L,
                "R_eff_ohm": ref_R,
                "Q": ref_Q,
                "first_resonance_Hz": resonance,
            }
        )
        global_records.append(
            {
                "spiral_name": spiral_name,
                "port_name": port_name,
                "N_conductors": n,
                "ref_freq_Hz": REF_FREQ,
                "L_eff_H": ref_L,
                "R_eff_ohm": ref_R,
                "Q": ref_Q,
                "first_resonance_Hz": resonance,
            }
        )
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(dirs["summary_spiral"], index=False)


def write_global_summary(root: Path, records: List[Dict[str, object]]) -> None:
    if not records:
        return
    report_dir = root / "Global_Report"
    report_dir.mkdir(exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(report_dir / "summary_all_spirals.csv", index=False)


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
