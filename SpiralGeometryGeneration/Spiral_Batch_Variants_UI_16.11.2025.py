#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spiral_Batch_PerLayer_UI.py
===========================
Tkinter GUI for batch-generating *multi-layer spiral variants* with **per-layer**
sweeps of:
  • K_arms  (integer, per layer)
  • N_turns (float, per layer)
  • CW / CCW direction (per layer)

For each full combination across all layers, the script calls your existing
Spiral_Drawer_updated.py module to:

    1) build the multi-arm geometry
    2) write ONLY "Wire_Sections.txt"

Each combination gets its own subfolder inside a user-chosen "mother" folder.
Folder names encode all layer settings, e.g.:

    L1_K1_N1.0_CW_L2_K2_N10.0_CW

The script **never** writes DXF; it only produces the TXT needed by the solvers.
"""

from __future__ import annotations

import os
import sys
import importlib
import importlib.util
import shutil
from dataclasses import dataclass
from decimal import Decimal, getcontext
from pathlib import Path
from typing import List, Tuple

import tkinter as tk
from tkinter import ttk, filedialog, messagebox


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


@dataclass
class LayerSweep:
    k_min: int
    k_max: int
    k_step: int
    n_min: float
    n_max: float
    n_step: float
    allow_cw: bool
    allow_ccw: bool


def float_range(start: float, stop: float, step: float) -> List[float]:
    """
    Precise float range using Decimal to avoid drift. Inclusive on `stop`.
    """
    if step <= 0:
        raise ValueError("Step must be > 0.")
    getcontext().prec = 28
    d_start = Decimal(str(start))
    d_stop = Decimal(str(stop))
    d_step = Decimal(str(step))

    vals: List[float] = []
    v = d_start
    # Include stop with a tiny tolerance
    while v <= d_stop + Decimal("1e-12"):
        vals.append(float(v))
        v += d_step
    return vals


def make_combo_folder_name(per_layer: List[Tuple[int, float, str]], nfmt: str) -> str:
    """
    Build a folder name from per-layer (K, N, dir) tuples.
    Example: L1_K1_N1.00_CW_L2_K2_N10.00_CW
    """
    parts = []
    for idx, (K, N, d) in enumerate(per_layer, start=1):
        n_str = format(N, nfmt)
        parts.append(f"L{idx}_K{int(K)}_N{n_str}_{d}")
    return "_".join(parts)


def write_address_file(mother: Path, subfolders: List[Path], filename: str = "Address.txt") -> None:
    """
    Write one absolute subfolder path per line into mother/Address.txt.
    Duplicate paths are removed.
    """
    unique: List[str] = []
    for p in subfolders:
        s = str(Path(p).resolve())
        if s not in unique:
            unique.append(s)

    addr_path = mother / filename
    addr_path.write_text("\n".join(unique), encoding="utf-8")


def verify_address_file(mother: Path, subfolders: List[Path], filename: str = "Address.txt") -> tuple[bool, str]:
    """
    Quick sanity check that Address.txt exists and that all paths inside exist.
    """
    addr_path = mother / filename
    if not addr_path.exists():
        return False, f"{filename} was not written in {mother}"

    lines = [ln.strip() for ln in addr_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        return False, f"{filename} in {mother} is empty."

    # Check existence
    missing: List[str] = []
    for ln in lines:
        if not Path(ln).exists():
            missing.append(ln)

    if missing:
        msg = f"{filename} written, but {len(missing)} path(s) do not exist:\n" + "\n".join(missing)
        return False, msg

    return True, f"{filename} written with {len(lines)} entries."


def import_spiral_module(preferred_name: str = "Spiral_Drawer_updated"):
    """
    Import the user's existing spiral generator module.

    Strategy:
      1) Try normal import by name.
      2) If that fails, ask the user to pick Spiral_Drawer_updated.py via a file
         dialog and import it from that path.
    """
    try:
        return importlib.import_module(preferred_name)
    except Exception:
        # Not found on sys.path — ask user for the .py file
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo(
            "Locate module",
            "Please select your 'Spiral_Drawer_updated.py' so I can import it.",
            parent=root,
        )
        path = filedialog.askopenfilename(
            title="Select Spiral_Drawer_updated.py",
            filetypes=[("Python files", "*.py"), ("All files", "*.*")],
        )
        root.destroy()

        if not path:
            raise ImportError("No module selected.")

        path = os.path.abspath(path)
        folder = os.path.dirname(path)
        if folder not in sys.path:
            sys.path.insert(0, folder)

        spec = importlib.util.spec_from_file_location(preferred_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from {path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[preferred_name] = module
        spec.loader.exec_module(module)  # type: ignore[assignment]

        return module


# ---------------------------------------------------------------------------
# Main GUI
# ---------------------------------------------------------------------------


class BatchApp(tk.Tk):
    """
    Tkinter GUI to define geometry + per-layer sweeps, then batch-generate
    variants by calling the user's existing Spiral_Drawer_updated.py.
    """

    def __init__(self, SDU_module):
        super().__init__()
        self.title("Spiral Batch Variants — per-layer sweeps (TXT only)")
        self.geometry("980x640")
        self.minsize(880, 600)

        self.SDU = SDU_module
        self._building = False

        # ------------------------------------------------------------------
        # Variables (defaults)
        # ------------------------------------------------------------------
        # Output
        self.var_mother = tk.StringVar(value=str(Path.cwd() / "Spiral_Variants"))
        self.var_decfmt = tk.StringVar(value=".2f")  # N name precision

        # Geometry
        self.var_Dout = tk.StringVar(value="50.0")
        self.var_W = tk.StringVar(value="0.25")
        self.var_S = tk.StringVar(value="0.25")
        self.var_M = tk.StringVar(value="1")
        self.var_dz = tk.StringVar(value="")
        self.var_dz_list = tk.StringVar(value="")

        # Sampling / header
        self.var_base = tk.StringVar(value="0.0")
        self.var_twist = tk.StringVar(value="0.0")
        self.var_pts = tk.StringVar(value="50")
        self.var_vol_res = tk.StringVar(value="0.01")
        self.var_coil_res = tk.StringVar(value="0.005")
        self.var_margin = tk.StringVar(value="1.0")

        # Per-layer sweeps: list[dict[str, tk.Variable]]
        self.layer_sweep_vars: List[dict] = []

        # Widgets that need to be accessed later
        self.layer_frame: ttk.Frame | None = None
        self.prog: ttk.Progressbar | None = None
        self.txt: tk.Text | None = None

        # Build UI
        self._build_ui()

        # Sync layer rows with current M and attach change handler
        self._sync_layer_sweep_count()
        self.var_M.trace_add("write", self._on_M_changed)

        # Initial log
        if self.txt is not None:
            try:
                mod_path = Path(self.SDU.__file__).resolve()
            except Exception:
                mod_path = Path("<unknown>")
            self._log(f"Imported module: {self.SDU.__name__} from {mod_path}")
            self._log("DXF generation is DISABLED. Only Wire_Sections.txt will be written.")

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        # Root grid: title / notebook / action / log
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(3, weight=1)

        # Title
        header = ttk.Frame(self)
        header.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 4))
        header.columnconfigure(0, weight=1)

        ttk.Label(
            header,
            text="Spiral batch generator (TXT only)",
            font=("Segoe UI", 11, "bold"),
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="1) Set geometry/layers  2) Define per-layer sweeps  3) Generate variants",
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))

        # Notebook with two tabs
        nb = ttk.Notebook(self)
        nb.grid(row=1, column=0, sticky="nsew", padx=10, pady=4)

        tab_geom = ttk.Frame(nb)
        tab_sweep = ttk.Frame(nb)
        nb.add(tab_geom, text="Geometry & layers")
        nb.add(tab_sweep, text="Sweeps & output")

        # ------------------------------------------------------------------
        # Geometry tab
        # ------------------------------------------------------------------
        tab_geom.columnconfigure(0, weight=1)

        # Basic geometry
        lf_geom = ttk.LabelFrame(tab_geom, text="Basic geometry (mm)")
        lf_geom.grid(row=0, column=0, sticky="ew", padx=8, pady=(10, 6))
        for c in range(6):
            lf_geom.columnconfigure(c, weight=1)

        ttk.Label(lf_geom, text="Dout [mm]").grid(row=0, column=0, sticky="e", padx=4, pady=4)
        ttk.Entry(lf_geom, textvariable=self.var_Dout, width=8).grid(row=0, column=1, sticky="w", padx=4, pady=4)

        ttk.Label(lf_geom, text="W [mm]").grid(row=0, column=2, sticky="e", padx=4, pady=4)
        ttk.Entry(lf_geom, textvariable=self.var_W, width=8).grid(row=0, column=3, sticky="w", padx=4, pady=4)

        ttk.Label(lf_geom, text="S [mm]").grid(row=0, column=4, sticky="e", padx=4, pady=4)
        ttk.Entry(lf_geom, textvariable=self.var_S, width=8).grid(row=0, column=5, sticky="w", padx=4, pady=4)

        ttk.Label(lf_geom, text="M layers").grid(row=1, column=0, sticky="e", padx=4, pady=4)
        ttk.Entry(lf_geom, textvariable=self.var_M, width=6).grid(row=1, column=1, sticky="w", padx=4, pady=4)

        ttk.Label(lf_geom, text="Δz [mm] (blank → W+S)").grid(row=1, column=2, sticky="e", padx=4, pady=4)
        ttk.Entry(lf_geom, textvariable=self.var_dz, width=8).grid(row=1, column=3, sticky="w", padx=4, pady=4)

        ttk.Label(lf_geom, text="Custom Δz list (comma)").grid(row=1, column=4, sticky="e", padx=4, pady=4)
        ttk.Entry(lf_geom, textvariable=self.var_dz_list, width=18).grid(row=1, column=5, sticky="w", padx=4, pady=4)

        # Sampling & header
        lf_sim = ttk.LabelFrame(tab_geom, text="Sampling & header for Wire_Sections.txt")
        lf_sim.grid(row=1, column=0, sticky="ew", padx=8, pady=(6, 10))
        for c in range(6):
            lf_sim.columnconfigure(c, weight=1)

        ttk.Label(lf_sim, text="Base phase [deg]").grid(row=0, column=0, sticky="e", padx=4, pady=4)
        ttk.Entry(lf_sim, textvariable=self.var_base, width=8).grid(row=0, column=1, sticky="w", padx=4, pady=4)

        ttk.Label(lf_sim, text="Twist per layer [deg]").grid(row=0, column=2, sticky="e", padx=4, pady=4)
        ttk.Entry(lf_sim, textvariable=self.var_twist, width=8).grid(row=0, column=3, sticky="w", padx=4, pady=4)

        ttk.Label(lf_sim, text="PTS_PER_TURN").grid(row=0, column=4, sticky="e", padx=4, pady=4)
        ttk.Entry(lf_sim, textvariable=self.var_pts, width=8).grid(row=0, column=5, sticky="w", padx=4, pady=4)

        ttk.Label(lf_sim, text="vol_res [cm]").grid(row=1, column=0, sticky="e", padx=4, pady=4)
        ttk.Entry(lf_sim, textvariable=self.var_vol_res, width=8).grid(row=1, column=1, sticky="w", padx=4, pady=4)

        ttk.Label(lf_sim, text="coil_res [cm]").grid(row=1, column=2, sticky="e", padx=4, pady=4)
        ttk.Entry(lf_sim, textvariable=self.var_coil_res, width=8).grid(row=1, column=3, sticky="w", padx=4, pady=4)

        ttk.Label(lf_sim, text="margin [cm]").grid(row=1, column=4, sticky="e", padx=4, pady=4)
        ttk.Entry(lf_sim, textvariable=self.var_margin, width=8).grid(row=1, column=5, sticky="w", padx=4, pady=4)

        # ------------------------------------------------------------------
        # Sweep & output tab
        # ------------------------------------------------------------------
        tab_sweep.columnconfigure(1, weight=1)

        # Mother folder
        r0 = ttk.Frame(tab_sweep)
        r0.grid(row=0, column=0, columnspan=3, sticky="ew", padx=8, pady=(10, 4))
        r0.columnconfigure(1, weight=1)
        ttk.Label(r0, text="Mother output folder:").grid(row=0, column=0, sticky="w")
        ttk.Entry(r0, textvariable=self.var_mother).grid(row=0, column=1, sticky="ew", padx=(6, 6))
        ttk.Button(r0, text="Browse…", command=self.on_browse_folder).grid(row=0, column=2, sticky="e")

        # N precision
        r1 = ttk.Frame(tab_sweep)
        r1.grid(row=1, column=0, columnspan=3, sticky="w", padx=8, pady=(0, 4))
        ttk.Label(r1, text="N name precision (e.g. .2f):").grid(row=0, column=0, sticky="w")
        ttk.Entry(r1, textvariable=self.var_decfmt, width=8).grid(row=0, column=1, sticky="w", padx=(4, 0))

        # Per-layer sweeps table
        lf_sw = ttk.LabelFrame(tab_sweep, text="Per-layer sweeps & directions")
        lf_sw.grid(row=2, column=0, columnspan=3, sticky="nsew", padx=8, pady=(4, 8))
        tab_sweep.rowconfigure(2, weight=1)
        lf_sw.columnconfigure(0, weight=1)

        ttk.Label(
            lf_sw,
            text="Rows correspond to layers 1..M (set on 'Geometry & layers' tab).",
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=6, pady=(4, 2))

        self.layer_frame = ttk.Frame(lf_sw)
        self.layer_frame.grid(row=1, column=0, sticky="nsew", padx=4, pady=(0, 4))
        lf_sw.rowconfigure(1, weight=1)

        # ------------------------------------------------------------------
        # Action bar + progress / log
        # ------------------------------------------------------------------
        action = ttk.Frame(self)
        action.grid(row=2, column=0, sticky="ew", padx=10, pady=(4, 4))
        action.columnconfigure(0, weight=1)
        ttk.Button(
            action,
            text="Generate variants (TXT only)",
            command=self.on_generate,
        ).grid(row=0, column=1, sticky="e")

        pframe = ttk.LabelFrame(self, text="Progress & log")
        pframe.grid(row=3, column=0, sticky="nsew", padx=10, pady=(0, 10))
        pframe.columnconfigure(0, weight=1)
        pframe.rowconfigure(1, weight=1)

        self.prog = ttk.Progressbar(pframe, mode="determinate")
        self.prog.grid(row=0, column=0, sticky="ew", padx=8, pady=6)

        self.txt = tk.Text(pframe, height=10, wrap="word")
        self.txt.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))

    # ------------------------------------------------------------------ helpers for per-layer sweeps

    def _make_default_layer_vars(self, idx: int) -> dict:
        """
        Create tk.Variable set for one layer. idx is 0-based layer index.
        """
        # Defaults: K 1..3 step 1, N 1..3 step 0.5; both CW & CCW allowed
        return {
            "K_min": tk.StringVar(value="1"),
            "K_max": tk.StringVar(value="3"),
            "K_step": tk.StringVar(value="1"),
            "N_min": tk.StringVar(value="1.0"),
            "N_max": tk.StringVar(value="3.0"),
            "N_step": tk.StringVar(value="0.5"),
            "cw": tk.BooleanVar(value=True if idx == 0 else False),
            "ccw": tk.BooleanVar(value=True),
        }

    def _sync_layer_sweep_count(self):
        """
        Ensure self.layer_sweep_vars has exactly M entries and rebuild the table.
        """
        try:
            M = int(self.var_M.get())
        except Exception:
            M = 1
        if M < 1:
            M = 1

        old = self.layer_sweep_vars
        new: List[dict] = []

        for i in range(M):
            if i < len(old):
                new.append(old[i])
            else:
                new.append(self._make_default_layer_vars(i))

        self.layer_sweep_vars = new
        self._rebuild_layer_table()

    def _rebuild_layer_table(self):
        """
        Destroy and rebuild the rows in self.layer_frame based on layer_sweep_vars.
        """
        if self.layer_frame is None:
            return

        for child in self.layer_frame.winfo_children():
            child.destroy()

        # Header row
        headers = ["Layer", "K_min", "K_max", "K_step", "N_min", "N_max", "N_step", "CW", "CCW"]
        for col, txt in enumerate(headers):
            ttk.Label(self.layer_frame, text=txt).grid(row=0, column=col, sticky="nsew", padx=2, pady=2)

        for c in range(len(headers)):
            self.layer_frame.columnconfigure(c, weight=1 if c > 0 else 0)

        # Data rows
        for i, vars_ in enumerate(self.layer_sweep_vars):
            r = i + 1
            ttk.Label(self.layer_frame, text=f"L{i+1}").grid(row=r, column=0, sticky="w", padx=4, pady=2)

            ttk.Entry(self.layer_frame, textvariable=vars_["K_min"], width=6).grid(row=r, column=1, sticky="ew", padx=2, pady=2)
            ttk.Entry(self.layer_frame, textvariable=vars_["K_max"], width=6).grid(row=r, column=2, sticky="ew", padx=2, pady=2)
            ttk.Entry(self.layer_frame, textvariable=vars_["K_step"], width=6).grid(row=r, column=3, sticky="ew", padx=2, pady=2)

            ttk.Entry(self.layer_frame, textvariable=vars_["N_min"], width=8).grid(row=r, column=4, sticky="ew", padx=2, pady=2)
            ttk.Entry(self.layer_frame, textvariable=vars_["N_max"], width=8).grid(row=r, column=5, sticky="ew", padx=2, pady=2)
            ttk.Entry(self.layer_frame, textvariable=vars_["N_step"], width=8).grid(row=r, column=6, sticky="ew", padx=2, pady=2)

            ttk.Checkbutton(self.layer_frame, variable=vars_["cw"]).grid(row=r, column=7, sticky="n", padx=2, pady=2)
            ttk.Checkbutton(self.layer_frame, variable=vars_["ccw"]).grid(row=r, column=8, sticky="n", padx=2, pady=2)

    def _on_M_changed(self, *args):
        self._sync_layer_sweep_count()

    # ------------------------------------------------------------------ read inputs

    def _read_geom_and_header(self) -> dict:
        """
        Read geometry + sampling/header values from the UI, return a dict.
        """
        try:
            Dout = float(self.var_Dout.get())
            W = float(self.var_W.get())
            S = float(self.var_S.get())
            M = int(self.var_M.get())
            if M < 1:
                raise ValueError

            dz_s = self.var_dz.get().strip()
            dz = float(dz_s) if dz_s != "" else None

            dz_list_raw = self.var_dz_list.get().strip()
            dz_list = None
            if dz_list_raw:
                cleaned = dz_list_raw.replace(";", ",")
                parts = [p.strip() for p in cleaned.split(",") if p.strip()]
                if parts:
                    dz_list = [float(p) for p in parts]

            base = float(self.var_base.get())
            tw = float(self.var_twist.get())
            pts = int(self.var_pts.get())
            vr = float(self.var_vol_res.get())
            cr = float(self.var_coil_res.get())
            mg = float(self.var_margin.get())
        except Exception as exc:
            raise ValueError("Check geometry / sampling inputs; one or more are invalid.") from exc

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
        )

    def _read_n_format(self) -> str:
        fmt = self.var_decfmt.get().strip()
        if not fmt:
            fmt = ".2f"
        if not fmt.startswith("."):
            fmt = "." + fmt
        try:
            format(1.23, fmt)
        except Exception as exc:
            raise ValueError(f"Invalid N name precision format: {fmt}") from exc
        return fmt

    def _read_layer_sweeps(self, M: int) -> List[LayerSweep]:
        if len(self.layer_sweep_vars) < M:
            raise ValueError("Internal error: not enough layer sweep rows for M layers.")

        sweeps: List[LayerSweep] = []
        for i in range(M):
            v = self.layer_sweep_vars[i]
            try:
                k_min = int(v["K_min"].get())
                k_max = int(v["K_max"].get())
                k_step = int(v["K_step"].get())
                if k_step <= 0 or k_max < k_min or k_min < 1:
                    raise ValueError
            except Exception:
                raise ValueError(f"Layer {i+1}: invalid K_arms range.")

            try:
                n_min = float(v["N_min"].get())
                n_max = float(v["N_max"].get())
                n_step = float(v["N_step"].get())
                if n_step <= 0 or n_max < n_min:
                    raise ValueError
            except Exception:
                raise ValueError(f"Layer {i+1}: invalid N_turns range.")

            allow_cw = bool(v["cw"].get())
            allow_ccw = bool(v["ccw"].get())
            if not (allow_cw or allow_ccw):
                raise ValueError(f"Layer {i+1}: please select at least CW or CCW.")

            sweeps.append(LayerSweep(k_min, k_max, k_step, n_min, n_max, n_step, allow_cw, allow_ccw))

        return sweeps

    # ------------------------------------------------------------------ actions

    def on_browse_folder(self):
        start_dir = self.var_mother.get() or str(Path.cwd())
        folder = filedialog.askdirectory(title="Select mother output folder", initialdir=start_dir)
        if folder:
            self.var_mother.set(folder)

    def _log(self, msg: str):
        if self.txt is None:
            return
        self.txt.insert("end", msg + "\n")
        self.txt.see("end")
        self.update_idletasks()

    def on_generate(self):
        if self._building:
            return

        # Read inputs
        try:
            geom = self._read_geom_and_header()
            nfmt = self._read_n_format()
            sweeps = self._read_layer_sweeps(geom["M"])
            mother = Path(self.var_mother.get()).resolve()
            mother.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            messagebox.showerror("Invalid inputs", str(e), parent=self)
            return

        # Spiral module API
        SpiralInputs = getattr(self.SDU, "SpiralInputs")
        SimHeader = getattr(self.SDU, "SimHeader")
        build = getattr(self.SDU, "build_multiarm_geometry")
        write_txt = getattr(self.SDU, "write_wire_sections_txt")
        I_AMP = getattr(self.SDU, "I_AMP", 1.0)
        BOX_MODE = getattr(self.SDU, "BOX_MODE", "auto")

        # Build per-layer state lists
        all_layer_states: List[List[Tuple[int, float, str]]] = []
        try:
            for i, sw in enumerate(sweeps):
                Ks = list(range(sw.k_min, sw.k_max + 1, sw.k_step))
                Ns = float_range(sw.n_min, sw.n_max, sw.n_step)
                dirs: List[str] = []
                if sw.allow_ccw:
                    dirs.append("CCW")
                if sw.allow_cw:
                    dirs.append("CW")
                states = [(K, N, d) for K in Ks for N in Ns for d in dirs]
                if not states:
                    raise ValueError(f"Layer {i+1}: sweep produced no combinations.")
                all_layer_states.append(states)
        except Exception as e:
            messagebox.showerror("Invalid sweep definition", str(e), parent=self)
            return

        # Cartesian product across all layers
        from itertools import product

        combos_per_layer = list(product(*all_layer_states))
        total = len(combos_per_layer)
        if total == 0:
            messagebox.showwarning("Nothing to do", "Empty sweep (no combinations).", parent=self)
            return

        self._building = True
        if self.prog is not None:
            self.prog.config(mode="determinate", maximum=total, value=0)

        self._log(f"Starting generation: {total} combination(s) across {geom['M']} layer(s).")

        sim = SimHeader(geom["vol_res"], geom["coil_res"], geom["margin"])

        done = 0
        skipped = 0
        outdirs: List[Path] = []
        not_created: List[str] = []

        for per_layer in combos_per_layer:
            # per_layer is tuple of (K, N, dir) for each layer
            layer_arms = [int(K) for (K, _, _) in per_layer]
            layer_turns = [float(N) for (_, N, _) in per_layer]
            layer_dirs = [d for (_, _, d) in per_layer]

            # Global K/N are not very important when per-layer lists are given,
            # but we set them to layer 1 for completeness.
            K_global = layer_arms[0]
            N_global = layer_turns[0]

            # Folder name encodes all layer settings
            subname = make_combo_folder_name(list(per_layer), nfmt)
            outdir = mother / subname

            if outdir.exists():
                skipped += 1
                self._log(f"Skip (exists) → {subname}")
            else:
                outdir.mkdir(parents=True, exist_ok=True)
                txt_path = outdir / "Wire_Sections.txt"

                params = SpiralInputs(
                    Dout_mm=geom["Dout"],
                    W_mm=geom["W"],
                    S_mm=geom["S"],
                    N_turns=float(N_global),
                    K_arms=int(K_global),
                    M_layers=geom["M"],
                    dz_mm=geom["dz"],
                    layer_gaps_mm=geom["dz_list"],
                    layer_dirs=layer_dirs,
                    layer_arms=layer_arms,
                    layer_turns=layer_turns,
                    base_phase_deg=geom["base"],
                    twist_per_layer_deg=geom["tw"],
                    pts_per_turn=geom["pts"],
                )

                try:
                    arms_xy, zs, dirs = build(params)
                    write_txt(
                        arms_xy,
                        zs,
                        str(txt_path),
                        sim,
                        I_amp=I_AMP,
                        box=BOX_MODE,
                        section_dirs=dirs,
                    )
                    outdirs.append(outdir)
                    self._log(f"OK  → {subname}/Wire_Sections.txt  (Sections={len(arms_xy)})")
                except Exception as exc:
                    skipped += 1
                    not_created.append(subname)
                    self._log(f"ERR → {subname}  ({exc})")
                    if outdir.exists():
                        try:
                            shutil.rmtree(outdir)
                        except Exception as cleanup_exc:
                            self._log(f"Cleanup failed for {subname}: {cleanup_exc}")

            done += 1
            if self.prog is not None:
                self.prog.step(1)
            self.update_idletasks()

        # Address.txt in mother folder
        write_address_file(mother, outdirs)
        ok, msg = verify_address_file(mother, outdirs)
        if not_created:
            not_created_path = mother / "NotCreated.txt"
            not_created_path.write_text("\n".join(not_created), encoding="utf-8")
            self._log(f"Recorded {len(not_created)} not-created variant(s) → {not_created_path.name}")
        if ok:
            messagebox.showinfo("Generation complete", msg, parent=self)
        else:
            messagebox.showwarning("Check outputs", msg, parent=self)

        self._log(f"Done. Created={total - skipped}, Skipped={skipped}, Folder={mother}")
        self._building = False


# ---------------------------------------------------------------------------


def main():
    # Import user's original builder (no DXF, only public APIs).
    SDU = import_spiral_module("Spiral_Drawer_updated")

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
