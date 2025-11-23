#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spiral_Batch_Variants_UI.py
===========================
A tiny Tkinter GUI that batches *spiral variants* by sweeping user-specified ranges
of:
  • K_arms  (integer)
  • N_turns (float, fractional allowed)

For each (K, N) combination, it *calls your existing generator* (from
Spiral_Drawer_updated.py) to:
    1) build the multi-arm geometry and
    2) write ONLY "Wire_Sections.txt"
…saving each result in its own subfolder under a user-chosen "mother" folder.

IMPORTANT:
- We DO NOT create DXF files here (as requested).
- We DO NOT modify your original script. We just import and call its public APIs.

USAGE:
- Put this file next to "Spiral_Drawer_updated.py", then run it.
  If it's not in the same folder, the app will prompt you to locate it.

NOTES:
- The rest of the fixed inputs (Dout, W, S, … header fields) can be kept at safe
  defaults or tweaked in the "Advanced (fixed params)" section.
- Folder naming pattern is configurable; default: "K{K}_N{N}" (with a configurable
  decimal precision for N). Dots in folder names are allowed on all major OSes.

Author: Lumi (for Bahadir)
"""

import importlib
import importlib.util
import math
import os
import sys
from dataclasses import dataclass, field
from decimal import Decimal, getcontext
from pathlib import Path
from typing import List, Tuple, Optional

# --- GUI ---
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog


@dataclass
class LayerConfig:
    name: str
    dirs: List[str]
    apply_all: bool = True
    combo_filter: Optional[set[Tuple[int, float]]] = field(default=None)

    def ensure_length(self, M: int):
        M = max(0, int(M))
        dirs = list(self.dirs)
        if len(dirs) < M:
            dirs.extend(["CCW"] * (M - len(dirs)))
        self.dirs = dirs[:M]

    def copy(self) -> "LayerConfig":
        return LayerConfig(
            name=self.name,
            dirs=list(self.dirs),
            apply_all=self.apply_all,
            combo_filter=set(self.combo_filter) if self.combo_filter else None,
        )

# ------------------------------------------------------------------------------
# 1) Utilities for robust floating sweeps and safe folder names
# ------------------------------------------------------------------------------

def frange(start: float, stop: float, step: float) -> List[float]:
    """
    Inclusive float range with minimal cumulative error, using Decimal under the hood.
    Example: frange(0.5, 2.0, 0.25) -> [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    """
    if step <= 0:
        raise ValueError("N_turns step must be > 0")

    # Use Decimal to control precision and avoid drift
    getcontext().prec = 28  # plenty for GUI inputs
    d_start = Decimal(str(start))
    d_stop  = Decimal(str(stop))
    d_step  = Decimal(str(step))

    vals: List[float] = []
    v = d_start
    # Include stop (± tiny tolerance)
    while v <= d_stop + Decimal("1e-12"):
        vals.append(float(v))
        v += d_step
    return vals


def sanitize_token(token: str) -> str:
    """Remove characters that would make folder names noisy."""
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in token)
    return cleaned.strip("_") or "cfg"


def layer_label(m_layers: int, dirs: List[str], cfg_name: str | None = None) -> str:
    base = f"L{m_layers}_" + "_".join(dirs or ["CCW"])
    if cfg_name:
        base += f"_{sanitize_token(cfg_name)}"
    return base


def safe_combo_folder_name(K: int, N: float, fmt: str, m_layers: int, dirs: List[str], cfg_name: str | None = None) -> str:
    """
    Builds a subfolder name from K, N, and layer metadata.
    - fmt is a Python format for N (e.g., ".3f" means 3 decimals).
    - Folder names now also encode layer count + chirality (and optional config label)
      so that different layer direction sets produce unique outputs.
    """
    n_str = format(N, fmt)
    return f"K{K}_N{n_str}_{layer_label(m_layers, dirs, cfg_name)}"


# NEW HELPER: write Address.txt in the mother folder
def write_address_file(mother: Path, subfolders: List[Path], filename: str = "Address.txt") -> None:
    """
    Create (or overwrite) a text file in the *mother* folder that contains the
    full paths of all generated subfolders, one path per line.

    Parameters
    ----------
    mother : Path
        The "mother" folder selected in the GUI. Address.txt is created here.

    subfolders : List[Path]
        List of subfolder paths created during the batch run
        (e.g. mother / "K1_N1.00", mother / "K1_N1.50", ...).

    filename : str, default "Address.txt"
        Name of the address file. Spelled as requested.
    """
    # If no subfolders were generated (e.g. everything skipped), do nothing.
    if not subfolders:
        return

    # Path to the Address.txt file
    addr_path = mother / filename

    # Write absolute paths, one per line.
    # Using resolve() ensures we get full paths like:
    #   D:\...\Try_16.11.2025_15.38\Spirals\K1_N1.00
    with addr_path.open("w", encoding="utf-8") as f:
        for folder in subfolders:
            f.write(str(folder.resolve()) + os.linesep)


def verify_address_file(mother: Path, expected: List[Path], filename: str = "Address.txt") -> Tuple[bool, str]:
    """Confirm Address.txt exists and matches the generated folders."""
    addr_path = mother / filename
    if not addr_path.is_file():
        return False, "Address.txt was not written."
    lines = [Path(line.strip()) for line in addr_path.read_text().splitlines() if line.strip()]
    expected_resolved = [p.resolve() for p in expected]
    if len(lines) != len(expected_resolved):
        return False, "Address.txt entry count does not match generated folders."
    missing = [p for p in lines if not p.exists()]
    if missing:
        return False, f"Missing folders listed in Address.txt: {', '.join(str(m) for m in missing)}"
    mismatch = set(lines) ^ set(expected_resolved)
    if mismatch:
        return False, "Address.txt entries differ from generated folders."
    return True, "All generated folders are present in Address.txt."


# ------------------------------------------------------------------------------
# 2) Dynamic import helper — load Spiral_Drawer_updated.py cleanly if needed
# ------------------------------------------------------------------------------

def import_spiral_module(preferred_name="Spiral_Drawer_updated"):
    """
    Try to import the user's existing spiral generator as a module:
      - First: normal import if the module is on sys.path (e.g., same folder)
      - Else: pop a file chooser and import from that path
    Returns the imported module object, or raises an Exception on failure.
    """
    try:
        return importlib.import_module(preferred_name)
    except Exception:
        # Not found on sys.path — ask the user to locate it
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo(
            "Locate module",
            "Please select your 'Spiral_Drawer_updated.py' so I can import it."
        )
        py_path = filedialog.askopenfilename(
            title="Select Spiral_Drawer_updated.py",
            filetypes=[("Python file", "*.py"), ("All files", "*.*")]
        )
        if not py_path:
            raise FileNotFoundError("Module not selected. Aborting.")

        spec = importlib.util.spec_from_file_location(preferred_name, py_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load spec from: {py_path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[preferred_name] = mod
        spec.loader.exec_module(mod)
        return mod


# ------------------------------------------------------------------------------
# 3) GUI App
# ------------------------------------------------------------------------------

class BatchApp(tk.Tk):
    """
    Simple Tkinter GUI to collect ranges and fixed params, then batch-generate
    variants by calling the user's existing functions.
    """
    def __init__(self, SDU_module):
        super().__init__()
        self.title("Spiral Batch Variants — TXT only")
        self.geometry("780x620")
        self.minsize(740, 580)

        # Keep a handle to the imported module (your original code)
        self.SDU = SDU_module

        # ---------------------------
        # Default values (safe, modest)
        # ---------------------------
        # Ranges:
        self.var_K_min   = tk.StringVar(value="1")
        self.var_K_max   = tk.StringVar(value="3")
        self.var_K_step  = tk.StringVar(value="1")
        self.var_N_min   = tk.StringVar(value="1.0")
        self.var_N_max   = tk.StringVar(value="3.0")
        self.var_N_step  = tk.StringVar(value="0.5")
        self.var_decfmt  = tk.StringVar(value=".2f")   # N name precision

        # Output mother folder + naming pattern
        self.var_mother  = tk.StringVar(value=str(Path.cwd() / "Spiral_Variants"))
        self.var_pattern = tk.StringVar(value="K{K}_N{N}")  # NOTE: used for display only
        # We actually format with safe_combo_folder_name(..., fmt=var_decfmt)

        # Fixed geometry (mm) — keep modest defaults; user can adjust if desired
        self.var_Dout = tk.StringVar(value="50.0")
        self.var_W    = tk.StringVar(value="0.25")
        self.var_S    = tk.StringVar(value="0.25")
        self.var_M    = tk.StringVar(value="1")
        self.var_dz   = tk.StringVar(value="")  # blank → default (W+S)
        self.var_dz_list = tk.StringVar(value="")  # optional per-gap overrides
        self.var_base = tk.StringVar(value="0.0")
        self.var_twist= tk.StringVar(value="0.0")
        self.var_layer_dir_summary = tk.StringVar(value="All layers: CCW")

        # Sampling density — take the constant from the original module if present
        default_pts = getattr(self.SDU, "PTS_PER_TURN", 50)
        self.var_pts  = tk.StringVar(value=str(default_pts))

        # Header for Wire_Sections.txt (cm) — pull defaults from original if present
        def_or = str(getattr(self.SDU, "DEF_VOL_RES_CM", 0.010))
        def_cr = str(getattr(self.SDU, "DEF_COIL_RES_CM", 0.005))
        def_mg = str(getattr(self.SDU, "DEF_MARGIN_CM",   1.000))
        self.var_vol_res  = tk.StringVar(value=def_or)
        self.var_coil_res = tk.StringVar(value=def_cr)
        self.var_margin   = tk.StringVar(value=def_mg)

        # Internal state
        self._building = False
        self._layer_dirs: List[str] = []
        self._extra_layer_configs: List[LayerConfig] = []
        self.var_M.trace_add("write", self._on_layers_changed)
        self._on_layers_changed()

        # UI layout
        self._build_ui()

    # ---------------- UI construction ----------------

    def _build_ui(self):
        # Top frame for ranges + output
        top = ttk.LabelFrame(self, text="Parameter sweeps & output")
        top.pack(side="top", fill="x", padx=10, pady=8)

        # Row 1: Mother folder
        r1 = ttk.Frame(top); r1.pack(fill="x", padx=8, pady=4)
        ttk.Label(r1, text="Mother output folder:").pack(side="left")
        ttk.Entry(r1, textvariable=self.var_mother, width=65).pack(side="left", padx=6)
        ttk.Button(r1, text="Browse…", command=self.on_browse_folder).pack(side="left")

        # Row 2: K range
        r2 = ttk.Frame(top); r2.pack(fill="x", padx=8, pady=4)
        ttk.Label(r2, text="K_arms range (min / max / step):").pack(side="left")
        ttk.Entry(r2, textvariable=self.var_K_min, width=6).pack(side="left", padx=4)
        ttk.Entry(r2, textvariable=self.var_K_max, width=6).pack(side="left", padx=4)
        ttk.Entry(r2, textvariable=self.var_K_step, width=6).pack(side="left", padx=4)

        # Row 3: N range
        r3 = ttk.Frame(top); r3.pack(fill="x", padx=8, pady=4)
        ttk.Label(r3, text="N_turns range (min / max / step):").pack(side="left")
        ttk.Entry(r3, textvariable=self.var_N_min, width=8).pack(side="left", padx=4)
        ttk.Entry(r3, textvariable=self.var_N_max, width=8).pack(side="left", padx=4)
        ttk.Entry(r3, textvariable=self.var_N_step, width=8).pack(side="left", padx=4)
        ttk.Label(r3, text="N name precision (e.g. .2f):").pack(side="left", padx=(16,4))
        ttk.Entry(r3, textvariable=self.var_decfmt, width=6).pack(side="left")

        # Advanced (fixed params)
        adv = ttk.LabelFrame(self, text="Advanced (fixed params used for ALL combinations)")
        adv.pack(side="top", fill="x", padx=10, pady=8)

        g1 = ttk.Frame(adv); g1.pack(fill="x", padx=8, pady=3)
        ttk.Label(g1, text="Dout [mm]").pack(side="left")
        ttk.Entry(g1, textvariable=self.var_Dout, width=8).pack(side="left", padx=6)

        ttk.Label(g1, text="W [mm]").pack(side="left")
        ttk.Entry(g1, textvariable=self.var_W, width=8).pack(side="left", padx=6)

        ttk.Label(g1, text="S [mm]").pack(side="left")
        ttk.Entry(g1, textvariable=self.var_S, width=8).pack(side="left", padx=6)

        ttk.Label(g1, text="M layers").pack(side="left")
        ttk.Entry(g1, textvariable=self.var_M, width=6).pack(side="left", padx=6)

        ttk.Label(g1, text="Δz [mm] (blank→W+S)").pack(side="left")
        ttk.Entry(g1, textvariable=self.var_dz, width=8).pack(side="left", padx=6)

        ttk.Label(g1, text="Custom Δz list (comma)").pack(side="left")
        ttk.Entry(g1, textvariable=self.var_dz_list, width=14).pack(side="left", padx=6)

        dir_row = ttk.Frame(adv); dir_row.pack(fill="x", padx=8, pady=3)
        ttk.Label(dir_row, text="Layer directions (chirality):").pack(side="left")
        ttk.Button(dir_row, text="Set…", command=self._open_layer_dir_dialog).pack(side="left", padx=6)
        ttk.Label(dir_row, textvariable=self.var_layer_dir_summary).pack(side="left", padx=4)

        cfg_row = ttk.Frame(adv); cfg_row.pack(fill="x", padx=8, pady=3)
        ttk.Label(cfg_row, text="Layer configurations (CW/CCW sets):").pack(side="left")
        ttk.Button(cfg_row, text="Manage…", command=self._open_layer_config_manager).pack(side="left", padx=6)
        self.var_cfg_summary = tk.StringVar(value="Using 1 configuration")
        ttk.Label(cfg_row, textvariable=self.var_cfg_summary).pack(side="left", padx=4)

        g2 = ttk.Frame(adv); g2.pack(fill="x", padx=8, pady=3)
        ttk.Label(g2, text="Base phase [deg]").pack(side="left")
        ttk.Entry(g2, textvariable=self.var_base, width=8).pack(side="left", padx=6)

        ttk.Label(g2, text="Twist per layer [deg]").pack(side="left")
        ttk.Entry(g2, textvariable=self.var_twist, width=8).pack(side="left", padx=6)

        ttk.Label(g2, text="PTS_PER_TURN").pack(side="left")
        ttk.Entry(g2, textvariable=self.var_pts, width=6).pack(side="left", padx=6)

        g3 = ttk.Frame(adv); g3.pack(fill="x", padx=8, pady=3)
        ttk.Label(g3, text="Header (cm)  vol_res=").pack(side="left")
        ttk.Entry(g3, textvariable=self.var_vol_res, width=8).pack(side="left", padx=6)
        ttk.Label(g3, text="coil_res=").pack(side="left")
        ttk.Entry(g3, textvariable=self.var_coil_res, width=8).pack(side="left", padx=6)
        ttk.Label(g3, text="margin=").pack(side="left")
        ttk.Entry(g3, textvariable=self.var_margin, width=8).pack(side="left", padx=6)

        # Action bar
        bar = ttk.Frame(self); bar.pack(side="top", fill="x", padx=10, pady=(6,8))
        ttk.Button(bar, text="Generate variants (TXT only)", command=self.on_generate)\
            .pack(side="right")

        # Progress + log
        pframe = ttk.LabelFrame(self, text="Progress & log")
        pframe.pack(side="top", fill="both", expand=True, padx=10, pady=(0,10))
        self.prog = ttk.Progressbar(pframe, mode="determinate")
        self.prog.pack(fill="x", padx=8, pady=6)
        self.txt = tk.Text(pframe, height=16, wrap="word")
        self.txt.pack(fill="both", expand=True, padx=8, pady=(0,8))
        self._log(f"Imported module: {self.SDU.__name__} from {Path(self.SDU.__file__).resolve()}")

        # Small note to reassure about DXF
        self._log("DXF generation is DISABLED. Only Wire_Sections.txt will be written.")

    def _on_layers_changed(self, *args):
        try:
            M = int(self.var_M.get())
        except Exception:
            M = 0
        self._ensure_layer_dir_length(M)
        self.var_layer_dir_summary.set(self._format_layer_dir_summary())

    def _ensure_layer_dir_length(self, M: int):
        M = max(0, int(M))
        cur = list(self._layer_dirs)
        if len(cur) < M:
            cur.extend(["CCW"] * (M - len(cur)))
        else:
            cur = cur[:M]
        self._layer_dirs = cur
        for cfg in self._extra_layer_configs:
            cfg.ensure_length(M)

    def _format_layer_dir_summary(self) -> str:
        if not self._layer_dirs:
            return "All layers: CCW"
        unique = set(self._layer_dirs)
        if len(unique) == 1:
            val = next(iter(unique))
            return f"All layers: {val}"
        preview = ", ".join(f"L{idx}:{val}" for idx, val in enumerate(self._layer_dirs))
        return f"Layer dirs → {preview}"

    def _format_cfg_summary(self) -> str:
        total = 1 + len(self._extra_layer_configs)
        names = [cfg.name for cfg in self._extra_layer_configs]
        suffix = f" (+ {', '.join(names)})" if names else ""
        return f"Using {total} configuration(s){suffix}"

    def _open_layer_dir_dialog(self):
        try:
            M = int(self.var_M.get())
        except Exception:
            messagebox.showerror("Layer directions", "Please enter a valid M value first.")
            return
        if M <= 0:
            messagebox.showerror("Layer directions", "Layer count must be ≥ 1.")
            return
        self._ensure_layer_dir_length(M)

        dlg = tk.Toplevel(self)
        dlg.title("Layer directions")
        dlg.transient(self)
        dlg.grab_set()

        ttk.Label(
            dlg,
            text=(
                "Choose chirality per layer.\n"
                "• CCW: standard Archimedean spiral\n"
                "• CW : mirrored chirality (θ → −θ)\n\n"
                "TXT export also reverses CW point order so solvers read outer→inner."
            ),
            justify="left",
            padding=8,
        ).pack(fill="x")

        body = ttk.Frame(dlg)
        body.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        choice_vars: List[tk.StringVar] = []
        for idx in range(M):
            var = tk.StringVar(value=self._layer_dirs[idx])
            row = ttk.Frame(body)
            row.pack(fill="x", pady=3)
            ttk.Label(row, text=f"Layer {idx}").pack(side="left", padx=(0, 8))
            ttk.Radiobutton(row, text="CCW (default)", value="CCW", variable=var).pack(side="left")
            ttk.Radiobutton(row, text="CW", value="CW", variable=var).pack(side="left", padx=(8, 0))
            choice_vars.append(var)

        btns = ttk.Frame(dlg)
        btns.pack(fill="x", padx=10, pady=(0, 10))
        btns.grid_columnconfigure(0, weight=1)
        btns.grid_columnconfigure(1, weight=1)

        def _apply_and_close():
            self._layer_dirs = [var.get() for var in choice_vars]
            self.var_layer_dir_summary.set(self._format_layer_dir_summary())
            self.var_cfg_summary.set(self._format_cfg_summary())
            dlg.destroy()

        ttk.Button(btns, text="Cancel", command=dlg.destroy).grid(row=0, column=0, sticky="e", padx=4)
        ttk.Button(btns, text="OK", command=_apply_and_close).grid(row=0, column=1, sticky="w", padx=4)

    # ---------------- Helpers ----------------

    def _log(self, msg: str):
        self.txt.insert("end", msg + "\n")
        self.txt.see("end")
        self.update_idletasks()

    def _add_layer_config(self, cfg: "LayerConfig"):
        self._extra_layer_configs.append(cfg)
        self.var_cfg_summary.set(self._format_cfg_summary())

    def _remove_layer_config(self, cfg: "LayerConfig"):
        self._extra_layer_configs = [c for c in self._extra_layer_configs if c is not cfg]
        self.var_cfg_summary.set(self._format_cfg_summary())

    def _get_all_configs(self, m_layers: int) -> List["LayerConfig"]:
        # Default config mirrors the main layer_dirs values
        base = LayerConfig(name="Default", dirs=list(self._layer_dirs), apply_all=True)
        base.ensure_length(m_layers)
        configs = [base]
        for cfg in self._extra_layer_configs:
            clone = cfg.copy()
            clone.ensure_length(m_layers)
            configs.append(clone)
        return configs

    def on_browse_folder(self):
        path = filedialog.askdirectory(title="Select mother output folder")
        if path:
            self.var_mother.set(path)

    def _read_ranges(self):
        # K range: ints
        try:
            k_min  = int(self.var_K_min.get())
            k_max  = int(self.var_K_max.get())
            k_step = int(self.var_K_step.get())
            if not (k_step > 0 and k_max >= k_min and k_min >= 1):
                raise ValueError
            ks = list(range(k_min, k_max + 1, k_step))
        except Exception:
            raise ValueError("K_arms range invalid. Use positive ints; max >= min; step > 0.")

        # N range: floats
        try:
            n_min  = float(self.var_N_min.get())
            n_max  = float(self.var_N_max.get())
            n_step = float(self.var_N_step.get())
            if not (n_step > 0 and n_max >= n_min):
                raise ValueError
            ns = frange(n_min, n_max, n_step)
        except Exception:
            raise ValueError("N_turns range invalid. Use numbers; max >= min; step > 0.")

        # N format
        fmt = self.var_decfmt.get().strip() or ".2f"
        # quick sanity: must start with "." and be numeric (e.g. ".3f")
        if not (fmt.endswith("f") and fmt.startswith(".") and fmt[1:-1].isdigit()):
            raise ValueError("N precision format should look like '.2f', '.3f', etc.")

        return ks, ns, fmt

    def _read_fixed(self):
        # Geometry (mm)
        try:
            Dout = float(self.var_Dout.get())
            W    = float(self.var_W.get())
            S    = float(self.var_S.get())
            M    = int(self.var_M.get())
            dz_s = self.var_dz.get().strip()
            dz   = float(dz_s) if dz_s != "" else None
            dz_list_raw = self.var_dz_list.get().strip()
            dz_list = None
            if dz_list_raw:
                cleaned = dz_list_raw.replace(";", ",")
                parts = [p.strip() for p in cleaned.split(",") if p.strip()]
                if parts:
                    dz_list = [float(p) for p in parts]
                    if any(v <= 0 for v in dz_list):
                        raise ValueError("Custom Δz entries must be > 0.")
                    expected = max(0, M - 1)
                    if len(dz_list) != expected:
                        raise ValueError(
                            f"Custom Δz list must have exactly {expected} entries for {M} layers."
                        )
            base = float(self.var_base.get())
            tw   = float(self.var_twist.get())
            pts  = int(self.var_pts.get())
            if Dout <= 0 or W <= 0 or S < 0 or M <= 0 or pts <= 0:
                raise ValueError
        except Exception:
            raise ValueError("Fixed geometry: please check Dout>0, W>0, S>=0, M>=1, PTS_PER_TURN>=1.")

        # Header (cm)
        try:
            vr = float(self.var_vol_res.get())
            cr = float(self.var_coil_res.get())
            mg = float(self.var_margin.get())
            if vr <= 0 or cr <= 0 or mg < 0:
                raise ValueError
        except Exception:
            raise ValueError("Header fields: vol_res>0, coil_res>0, margin>=0.")

        self._ensure_layer_dir_length(M)

        return dict(
            Dout=Dout,
            W=W,
            S=S,
            M=M,
            dz=dz,
            dz_list=dz_list,
            base=base,
            tw=tw,
            pts=pts,
            vol_res=vr,
            coil_res=cr,
            margin=mg,
            layer_dirs=list(self._layer_dirs),
        )

    # ---------------- Main action ----------------

    def on_generate(self):
        if self._building:
            return
        try:
            ks, ns, nfmt = self._read_ranges()
            fx = self._read_fixed()
            mother = Path(self.var_mother.get()).resolve()
            mother.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            messagebox.showerror("Invalid inputs", str(e))
            return

        # Prepare class/const references from the imported module
        SpiralInputs = getattr(self.SDU, "SpiralInputs")
        SimHeader    = getattr(self.SDU, "SimHeader")
        build        = getattr(self.SDU, "build_multiarm_geometry")
        write_txt    = getattr(self.SDU, "write_wire_sections_txt")
        I_AMP        = getattr(self.SDU, "I_AMP", 1.0)      # current per section
        BOX_MODE     = getattr(self.SDU, "BOX_MODE", "auto")

        # Progress UI
        combos = [(K, N) for K in ks for N in ns]
        configs = self._get_all_configs(fx["M"])
        total = sum(len(combos) if cfg.apply_all or not cfg.combo_filter else len([c for c in combos if c in cfg.combo_filter]) for cfg in configs)
        if total == 0:
            messagebox.showwarning("Nothing to do", "Empty sweep (no combinations).")
            return

        self._building = True
        self.prog.config(mode="determinate", maximum=total, value=0)
        self._log(f"Starting generation: {total} combination(s) across {len(configs)} configuration(s)")

        done = 0
        skipped = 0

        # Local list to keep track of all subfolders created in this run
        outdirs: List[Path] = []

        for cfg in configs:
            cfg_combos = combos if cfg.apply_all or not cfg.combo_filter else [c for c in combos if c in cfg.combo_filter]
            if not cfg_combos:
                self._log(f"Skip config '{cfg.name}' (no combinations selected)")
                continue
            for K, N in cfg_combos:
                done += 1
                self.prog['value'] = done
                subname = safe_combo_folder_name(K, N, fmt=nfmt, m_layers=fx["M"], dirs=cfg.dirs, cfg_name=cfg.name if cfg.name != "Default" else None)
                outdir  = mother / subname
                outdir.mkdir(parents=True, exist_ok=True)
                txt_path = outdir / "Wire_Sections.txt"

                # Remember this subfolder so we can later write it into Address.txt
                outdirs.append(outdir)

                # Build the params for this combo (only K and N change)
                params = SpiralInputs(
                    Dout_mm = fx["Dout"],
                    W_mm    = fx["W"],
                    S_mm    = fx["S"],
                    N_turns = float(N),
                    K_arms  = int(K),
                    M_layers= fx["M"],
                    dz_mm   = fx["dz"],
                    layer_gaps_mm = fx["dz_list"],
                    layer_dirs = cfg.dirs,
                    base_phase_deg     = fx["base"],
                    twist_per_layer_deg= fx["tw"],
                    pts_per_turn       = fx["pts"],
                )
                sim = SimHeader(fx["vol_res"], fx["coil_res"], fx["margin"])

                try:
                    # 1) build geometry
                    arms_xy, zs, dirs = build(params)
                    # 2) write ONLY the Wire_Sections.txt (no DXF here)
                    write_txt(
                        arms_xy,
                        zs,
                        str(txt_path),
                        sim,
                        I_amp=I_AMP,
                        box=BOX_MODE,
                        section_dirs=dirs,
                    )
                    self._log(f"OK  → {subname}/Wire_Sections.txt  (Sections={len(arms_xy)})")
                except Exception as exc:
                    skipped += 1
                    self._log(f"ERR → {subname}  ({exc})")

                self.update_idletasks()

        # After generating all variants, write Address.txt in the mother folder
        # containing the absolute paths of each generated subfolder.
        write_address_file(mother, outdirs)
        ok, msg = verify_address_file(mother, outdirs)
        if ok:
            messagebox.showinfo("Generation complete", msg)
        else:
            messagebox.showwarning("Check outputs", msg)

        self._log(f"Done. Created={total - skipped}, Skipped={skipped}, Folder={mother}")
        self._building = False

    def _preview_combos(self) -> List[Tuple[int, float]]:
        ks, ns, _ = self._read_ranges()
        return [(K, N) for K in ks for N in ns]

    def _open_layer_config_manager(self):
        try:
            combos = self._preview_combos()
        except Exception:
            combos = []

        dlg = tk.Toplevel(self)
        dlg.title("Layer configuration sets")
        dlg.transient(self)
        dlg.grab_set()

        frame = ttk.Frame(dlg)
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        cols = ("Name", "Apply scope", "Dirs")
        tree = ttk.Treeview(frame, columns=cols, show="headings", height=6)
        for col in cols:
            tree.heading(col, text=col)
        tree.column("Name", width=120)
        tree.column("Apply scope", width=140)
        tree.column("Dirs", width=240)
        tree.pack(fill="both", expand=True)

        def refresh_tree():
            tree.delete(*tree.get_children())
            all_cfgs = self._get_all_configs(int(self.var_M.get() or 0))
            for cfg in all_cfgs:
                scope = "All K/N" if cfg.apply_all or cfg.combo_filter is None else f"Selected ({len(cfg.combo_filter)})"
                tree.insert("", "end", iid=cfg.name, values=(cfg.name, scope, " / ".join(cfg.dirs)))

        btn_frame = ttk.Frame(dlg)
        btn_frame.pack(fill="x", pady=(6, 0))

        def choose_dirs(base_dirs: List[str]) -> List[str]:
            temp = LayerConfig("temp", list(base_dirs))
            self._ensure_layer_dir_length(len(base_dirs))
            # reuse existing dialog for directions
            # we temporarily patch _layer_dirs to reuse logic
            saved = list(self._layer_dirs)
            self._layer_dirs = list(base_dirs)
            self._open_layer_dir_dialog()
            dirs = list(self._layer_dirs)
            self._layer_dirs = saved
            return dirs

        def on_add():
            name = simpledialog.askstring("Config name", "Enter a name for this layer configuration:", parent=dlg)
            if not name:
                return
            dirs = choose_dirs(self._layer_dirs)
            cfg = LayerConfig(name=name, dirs=dirs, apply_all=True)
            cfg.ensure_length(len(self._layer_dirs))
            self._add_layer_config(cfg)
            refresh_tree()

        def on_remove():
            sel = tree.selection()
            if not sel:
                return
            name = sel[0]
            if name == "Default":
                messagebox.showinfo("Protected", "The default configuration cannot be removed.", parent=dlg)
                return
            cfg = next((c for c in self._extra_layer_configs if c.name == name), None)
            if cfg:
                self._remove_layer_config(cfg)
                refresh_tree()

        def on_scope():
            sel = tree.selection()
            if not sel:
                return
            name = sel[0]
            cfg = next((c for c in self._extra_layer_configs if c.name == name), None)
            if not cfg:
                messagebox.showinfo("Protected", "Default configuration always applies to all.", parent=dlg)
                return
            cfg.apply_all = not cfg.apply_all
            if cfg.apply_all:
                cfg.combo_filter = None
            refresh_tree()

        def on_choose_combos():
            sel = tree.selection()
            if not sel:
                return
            name = sel[0]
            cfg = next((c for c in self._extra_layer_configs if c.name == name), None)
            if not cfg:
                return
            if not combos:
                messagebox.showwarning("Ranges not ready", "Enter valid K/N ranges first to choose specific combinations.", parent=dlg)
                return
            cfg.apply_all = False

            chooser = tk.Toplevel(dlg)
            chooser.title(f"Select combos for {cfg.name}")
            chooser.transient(dlg)
            chooser.grab_set()

            vars: List[tk.BooleanVar] = []
            for combo in combos:
                var = tk.BooleanVar(value=(cfg.combo_filter is None or combo in (cfg.combo_filter or set())))
                chk = ttk.Checkbutton(chooser, text=f"K{combo[0]} / N{combo[1]}", variable=var)
                chk.pack(anchor="w", padx=8, pady=2)
                vars.append(var)

            def select_all():
                for v in vars:
                    v.set(True)

            def clear_all():
                for v in vars:
                    v.set(False)

            btns = ttk.Frame(chooser)
            btns.pack(fill="x", pady=4)
            ttk.Button(btns, text="All", command=select_all).pack(side="left", padx=4)
            ttk.Button(btns, text="None", command=clear_all).pack(side="left", padx=4)

            def apply_and_close():
                selected = {combo for combo, flag in zip(combos, vars) if flag.get()}
                cfg.combo_filter = selected
                chooser.destroy()
                refresh_tree()

            ttk.Button(btns, text="OK", command=apply_and_close).pack(side="right", padx=4)

        ttk.Button(btn_frame, text="Add", command=on_add).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="Remove", command=on_remove).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="Toggle scope", command=on_scope).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="Choose combos…", command=on_choose_combos).pack(side="left", padx=4)

        ttk.Button(dlg, text="Close", command=dlg.destroy).pack(pady=8)

        refresh_tree()


# ------------------------------------------------------------------------------
# 4) Entry point
# ------------------------------------------------------------------------------

def main():
    # Import the user's original builder (we *only* call its public APIs).
    # This keeps the contract very tight and avoids any DXF generation.
    SDU = import_spiral_module("Spiral_Drawer_updated")  # may open a file picker if not found

    # Sanity: verify the needed symbols exist before we even show the GUI.
    required = ["SpiralInputs", "SimHeader", "build_multiarm_geometry", "write_wire_sections_txt"]
    missing = [name for name in required if not hasattr(SDU, name)]
    if missing:
        raise ImportError(
            "The selected Spiral_Drawer_updated.py does not provide required symbols:\n"
            + ", ".join(missing)
        )

    app = BatchApp(SDU)
    app.mainloop()


if __name__ == "__main__":
    main()
