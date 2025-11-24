#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spiral_Ultimate.py — Multi-arm spiral builder with DXF + Wire_Sections.txt export
-------------------------------------------------------------------------------
This tool creates Archimedean spirals with *K arms per layer* and *M layers*.
Each arm is treated as a separate "Section" in Wire_Sections.txt (as requested).
The DXF contains one 2D polyline per *arm* (centerlines only, for compatibility).

Key features
============
• Inputs are trace width **W** and spacing **S** (NOT arm-to-arm distance).
• K arms (per layer); M layers; fractional turns **N** supported (e.g., 0.5).
• Outside diameter **Dout** governs the exact outer edge: outermost edge = Dout.
• Interactive 3D preview (TkAgg) with toolbar (pan/zoom/save) and mouse-wheel zoom.
• Scrollable GUI with a pinned bottom bar (Draw / Save).
• File outputs:
    - spiral_multiarm_fractional.dxf   (R2000 LWPOLYLINE centerlines)
    - Wire_Sections.txt                (mm units + header, Section-i per arm)

Units
=====
• All geometry in **mm**.
• Simulation header fields in **cm** (vol_res_cm, coil_res_cm, margin_cm) exactly
  as in your existing flow.

Author's note
=============
This script merges the friendlier UI you liked with the export formatting and geometry
treatment from your original "Spiral_Plot_DXF_with_sections.py", but with an
important fix: here, **each arm is a Section**, not each layer.
"""


# driver_subprocess.py
# -------------------------------------------------------------
# Call your existing txt2dxf_sections.py from another script.
# This does NOT modify your converter at all.
# -------------------------------------------------------------
import sys
import subprocess
from pathlib import Path
import os

import math
import tkinter as tk
from tkinter import ttk, messagebox
from dataclasses import dataclass
from typing import List, Tuple, Optional

# ---- Matplotlib (use TkAgg so the toolbar works inside Tkinter) ----
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (enables 3D)

# ==============================
# ---------- CONSTANTS ---------
# ==============================

# Geometric smoothness: samples per full 360° turn along the centerline.
# Increase for smoother curves (and larger files); decrease for lighter files.
PTS_PER_TURN = 50

# Default header metadata for Wire_Sections.txt (user can override in UI; cm units).
DEF_VOL_RES_CM  = 0.010
DEF_COIL_RES_CM = 0.005
DEF_MARGIN_CM   = 1.0

# Box mode string (kept constant to match the original downstream flow).
BOX_MODE = "auto"

# Current per Section (Wire_Sections.txt).
I_AMP = 1.0

# ==============================
# ------- DATA STRUCTURES ------
# ==============================

# Type alias for a single 2D polyline in mm (list of (x_mm, y_mm)).
Polyline2D = List[Tuple[float, float]]

@dataclass
class SpiralInputs:
    """
    Bundle of user inputs. Geometry in mm, counts unitless.
    """
    Dout_mm: float              # Outer diameter of *entire spiral* (edge-to-edge) [mm]
    W_mm: float                 # Trace width [mm]
    S_mm: float                 # Spacing between adjacent turns (centerline-to-centerline pitch = W+S) [mm]
    N_turns: float              # Turns per arm (can be fractional, e.g., 0.5)
    K_arms: int                 # Arms per layer (e.g., 1, 2, 3, ...)
    M_layers: int               # Number of layers (Z levels)
    dz_mm: Optional[float]      # Inter-layer distance [mm]; if blank/≤0, defaults to W+S
    base_phase_deg: float       # Reference angular offset for arm 0 [deg]
    twist_per_layer_deg: float  # Extra rotation applied per successive layer [deg]
    pts_per_turn: int           # Sampling density along the curve
    layer_gaps_mm: Optional[List[float]] = None  # Optional per-layer spacings; overrides dz_mm when provided
    layer_dirs: Optional[List[str]] = None       # Optional per-layer chirality ("CCW"/"CW"); defaults to CCW
    layer_arms: Optional[List[int]] = None       # Optional per-layer arm counts (fallbacks to K_arms)
    layer_turns: Optional[List[float]] = None    # Optional per-layer turns (fallbacks to N_turns)

@dataclass
class SimHeader:
    """
    Header fields copied into Wire_Sections.txt (in cm).
    """
    vol_res_cm: float
    coil_res_cm: float
    margin_cm: float

# ==============================
# ---- CORE GEOMETRY LOGIC -----
# ==============================

def _archimedean_params(Dout_mm: float, W_mm: float, S_mm: float, N_turns: float, N_arm : float) -> Tuple[float, float, float, float]:
    """
    Compute key Archimedean spiral parameters from user inputs.

    Spiral centerline equation:
        r(θ) = r0 + b * θ,     θ ∈ [0, 2πN]

    where
        b     = (W + S) / (2π)                [mm/rad], i.e., centerline radius increase per radian.
        r0    = Rout − W/2 − N * (W+S)        [mm],    chosen so **outer edge** lands on Rout.
        Rout  = Dout / 2                      [mm]
        θ_max = 2πN

    Returns:
        (r0, b, Rout, theta_max)
    """
    Dout  = float(Dout_mm)
    W     = float(W_mm)
    S     = float(S_mm)
    N     = float(N_turns)
    N_a   = float(N_arm)

    Rout = 0.5 * Dout                 # outer radius [mm]
    pitch = N_a * (W + S)                     # radial pitch per 360° between adjacent centerlines [mm]
    b = pitch / (2.0 * math.pi)       # Archimedean slope [mm/rad]
    theta_max = 2.0 * math.pi * N     # end angle [rad]

    # Choose r0 so the OUTER EDGE of the last turn lands exactly on Rout.
    # That is: r_last_center + W/2 = Rout  with r_last_center = r0 + b*theta_max = r0 + N*pitch.
    r0 = Rout - 0.5 * W - N * pitch

    return r0, b, Rout, theta_max


def _single_arm_centerline_xy(Dout_mm: float, W_mm: float, S_mm: float, N_turns: float, pts_per_turn: int, N_arm: float) -> Polyline2D:
    """
    Build one spiral **centerline** (not width-offset) polyline for θ ∈ [0, 2πN].
    The curve is sampled densely enough for smoothness.
    """
    r0, b, Rout, theta_max = _archimedean_params(Dout_mm, W_mm, S_mm, N_turns, N_arm)

    if r0 <= 0.0:
        raise ValueError(
            "Geometry does not fit: inner centerline radius r0 <= 0.\n"
            "Try increasing Dout, or reducing N / W / S."
        )

    # Decide number of samples. Keep at least ~12 points, scaled by N.
    n_pts = max(12, int(math.ceil(pts_per_turn * max(N_turns, 1e-3))))
    dth = theta_max / (n_pts - 1)

    pts: Polyline2D = []
    for i in range(n_pts):
        th = i * dth
        r  = r0 + b * th
        x  = r * math.cos(th)
        y  = r * math.sin(th)
        pts.append((x, y))

    return pts


def _rotate_xy(poly: Polyline2D, angle_deg: float) -> Polyline2D:
    """
    Rotate a 2D polyline by a given angle in degrees, around the origin.
    """
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    return [(ca * x - sa * y, sa * x + ca *y) for (x, y) in poly]


def _layer_z_levels(
    M_layers: int,
    W_mm: float,
    S_mm: float,
    dz_mm: Optional[float],
    dz_list_mm: Optional[List[float]] = None,
) -> List[float]:
    """
    Return Z levels [mm] for layers.

    If a custom list of inter-layer distances is provided (one value per gap between
    consecutive layers), it overrides dz_mm. Otherwise dz_mm is used, falling back to
    the default pitch W+S when blank/≤0.
    """
    M = int(M_layers)
    if M <= 0:
        return []

    if dz_list_mm:
        deltas = [float(v) for v in dz_list_mm]
        required = max(0, M - 1)
        if len(deltas) != required:
            raise ValueError(
                f"Custom Δz list must contain exactly {required} entries for {M} layers."
            )
        if any(d <= 0 for d in deltas):
            raise ValueError("Custom Δz entries must be > 0 mm.")
        levels: List[float] = [0.0]
        acc = 0.0
        for dz in deltas:
            acc += dz
            levels.append(acc)
        return levels

    dz = float(dz_mm) if (dz_mm is not None and float(dz_mm) > 0.0) else (float(W_mm) + float(S_mm))
    return [k * dz for k in range(M)]


def _normalize_layer_dirs(M_layers: int, user_dirs: Optional[List[str]]) -> List[str]:
    """Return a sanitized chirality list (length = M_layers)."""

    M = int(M_layers)
    if M <= 0:
        return []

    dirs: List[str] = []
    for idx in range(M):
        if user_dirs and idx < len(user_dirs):
            raw = str(user_dirs[idx]).strip().upper()
            if raw not in ("CCW", "CW"):
                raise ValueError("Layer direction entries must be 'CCW' or 'CW'.")
            dirs.append(raw)
        else:
            dirs.append("CCW")
    return dirs


def _normalize_layer_counts(M_layers: int, user_vals: Optional[List[int]], default: int, label: str) -> List[int]:
    """Normalize per-layer integer counts (e.g., arms)."""

    M = int(M_layers)
    vals: List[int] = []
    if user_vals is not None:
        vals = [int(v) for v in user_vals]
    if len(vals) < M:
        vals.extend([default] * (M - len(vals)))
    vals = vals[:M]

    for idx, v in enumerate(vals):
        if v <= 0:
            raise ValueError(f"{label} for layer {idx} must be > 0")
    return vals


def _normalize_layer_turns(M_layers: int, user_vals: Optional[List[float]], default: float) -> List[float]:
    """Normalize per-layer turn counts (float)."""

    M = int(M_layers)
    vals: List[float] = []
    if user_vals is not None:
        vals = [float(v) for v in user_vals]
    if len(vals) < M:
        vals.extend([default] * (M - len(vals)))
    vals = vals[:M]

    for idx, v in enumerate(vals):
        if v <= 0:
            raise ValueError(f"Turns for layer {idx} must be > 0")
    return vals


def _apply_chirality(poly: Polyline2D, chirality: str) -> Polyline2D:
    """Return a polyline mirrored for CW layers (theta → -theta)."""

    if chirality.upper() == "CW":
        return [(x, -y) for (x, y) in poly]
    return poly


def build_multiarm_geometry(params: SpiralInputs) -> Tuple[List[Polyline2D], List[float], List[str]]:
    """
    Generate all arms across all layers, returning:
      • XY polylines (flat list)
      • matching Z levels (per polyline)
      • per-section chirality tags ("CCW"/"CW")

    Lists align index-by-index; arms are grouped layer-major, then arm index.
    """
    # Prepare Z levels, per-layer twist, and chirality controls
    # Prepare Z levels and per-layer twist
    z_levels_layer0_to_M = _layer_z_levels(
        params.M_layers,
        params.W_mm,
        params.S_mm,
        params.dz_mm,
        params.layer_gaps_mm,
    )
    layer_dirs = _normalize_layer_dirs(params.M_layers, params.layer_dirs)
    layer_arms = _normalize_layer_counts(params.M_layers, params.layer_arms, int(params.K_arms), "Arms")
    layer_turns= _normalize_layer_turns(params.M_layers, params.layer_turns, float(params.N_turns))
    twist_per_layer = float(params.twist_per_layer_deg)
    base_phase = float(params.base_phase_deg)

    all_polys: List[Polyline2D] = []
    all_z:     List[float]      = []
    all_dirs:  List[str]        = []

    for layer_idx, z in enumerate(z_levels_layer0_to_M):
        k_layer = layer_arms[layer_idx]
        n_layer = layer_turns[layer_idx]

        # Build the base arm for this layer using its own turn count and arm pitch
        base = _single_arm_centerline_xy(
            Dout_mm       = params.Dout_mm,
            W_mm          = params.W_mm,
            S_mm          = params.S_mm,
            N_turns       = n_layer,
            N_arm         = k_layer,
            pts_per_turn  = params.pts_per_turn
        )

        per_arm_deg = 360.0 / k_layer
        extra_twist = layer_idx * twist_per_layer  # optional additional rotation per higher layer
        chirality = layer_dirs[layer_idx] if layer_idx < len(layer_dirs) else "CCW"

        for arm_idx in range(k_layer):
            rot_deg = base_phase + arm_idx * per_arm_deg
            poly = _rotate_xy(base, rot_deg)
            oriented = _apply_chirality(poly, chirality)
            if extra_twist != 0.0:
                oriented = _rotate_xy(oriented, extra_twist)
            all_polys.append(oriented)
            all_z.append(z)
            all_dirs.append(chirality)

    return all_polys, all_z, all_dirs

# ==============================
# ----------- OUTPUTS ----------
# ==============================

def write_wire_sections_txt(
    arms_xy: List[Polyline2D],
    z_levels_mm: List[float],
    path: str,
    sim: SimHeader,
    I_amp: float = I_AMP,
    box: str = BOX_MODE,
    section_dirs: Optional[List[str]] = None,
) -> None:
    """
    Wire_Sections.txt format (EXACTLY as your original flow expects):
      Line 1: "mm"
      Line 2: "vol_res_cm=...,coil_res_cm=...,margin_cm=...,box=auto"
      Then for each vertex of each ARM (section):
         Section-i,x_mm,y_mm,z_mm,I

    Important difference from one earlier variant you had:
      • Here a *Section* = 1 arm (not 1 layer). Section numbering simply follows the order
        of arms in the 'arms_xy' list (which we build layer-by-layer, arm-by-arm).
      • If section_dirs is provided, sections tagged "CW" have their vertices written in
        reverse order so TXT consumers still read outer→inner on mirrored layers.
    """
    if len(arms_xy) != len(z_levels_mm):
        raise ValueError("arms_xy and z_levels_mm must have the same length.")
    if section_dirs and len(section_dirs) != len(arms_xy):
        raise ValueError("section_dirs length must match number of sections.")

    with open(path, "w", encoding="utf-8") as f:
        # Units header
        f.write("mm\n")
        # Settings header (CSV style, same exact names/ordering as before)
        f.write(
            "vol_res_cm={:.6f},coil_res_cm={:.6f},margin_cm={:.6f},box={}\n\n".format(
                float(sim.vol_res_cm), float(sim.coil_res_cm), float(sim.margin_cm), box
            )
        )

        # One Section per ARM (order is layer-major, then arm index)
        for sec_idx, (poly, z) in enumerate(zip(arms_xy, z_levels_mm), start=1):
            sec_name = f"Section-{sec_idx}"
            pts = poly
            if section_dirs:
                idx = sec_idx - 1
                if idx < len(section_dirs) and section_dirs[idx].upper() == "CW":
                    pts = list(reversed(poly))
            for (x, y) in pts:
                f.write(f"{sec_name},{x:.8f},{y:.8f},{float(z):.8f},{float(I_amp):.8f}\n")


def write_simple_dxf_lwpolylines(txt_name: str, dxf_name: str):
    # Folder where everything is (this .py file's folder)
    workdir = Path(__file__).resolve().parent
    # Build full paths in that folder
    in_txt  = workdir / txt_name
    out_dxf = workdir / dxf_name

    # Sanity checks
    conv = workdir / "txt2dxf_sections.py"
    if not conv.exists():
        raise FileNotFoundError(f"Converter script missing: {conv}")
    if not in_txt.exists():
        raise FileNotFoundError(f"Input TXT not found: {in_txt}")

    # Run the converter in the same env; capture output for debugging
    res = subprocess.run(
        [sys.executable, str(conv), str(in_txt)],
        cwd=workdir, text=True, capture_output=True
    )
    if res.returncode != 0:
        raise RuntimeError(
            "txt2dxf_sections.py failed "
            f"(exit {res.returncode}).\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
        )

    # Produced DXF should have same stem as TXT
    produced = in_txt.with_suffix(".dxf")
    if not produced.exists():
        raise FileNotFoundError(f"Expected DXF not produced: {produced}")

    # Rename/move the produced file to desired name (atomic overwrite)
    os.replace(produced, out_dxf)
    print(f"DXF ready: {out_dxf}")



# ==============================
# ----------- PREVIEW ----------
# ==============================

def plot_3d_in_window(parent_frame, arms_xy: List[Polyline2D], z_levels_mm: List[float], title="Spiral 3D preview"):
    """
    Embed a Matplotlib 3D plot inside the given Tk frame.
    • Adds a standard toolbar (pan, zoom, home, save)
    • Supports mouse-wheel zoom via Matplotlib defaults
    """
    # Build a fresh figure each time to avoid dangling references
    fig = plt.Figure(figsize=(7.4, 5.6), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)

    # Plot each arm at its Z
    for poly, z in zip(arms_xy, z_levels_mm):
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        zs = [z] * len(poly)
        ax.plot(xs, ys, zs, linewidth=1.2)

    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("z [mm]")
    ax.grid(True, alpha=0.3)

    # Equal-ish XY limits for a nicer view
    if arms_xy:
        all_x = [x for poly in arms_xy for (x, _) in poly]
        all_y = [y for poly in arms_xy for (_, y) in poly]
        xmid = 0.5 * (min(all_x) + max(all_x))
        ymid = 0.5 * (min(all_y) + max(all_y))
        extent = 0.55 * max(max(all_x) - min(all_x), max(all_y) - min(all_y), 1e-6)
        ax.set_xlim(xmid - extent, xmid + extent)
        ax.set_ylim(ymid - extent, ymid + extent)

    # Clear old canvas if present
    for child in parent_frame.winfo_children():
        child.destroy()

    # Embed canvas + toolbar
    canvas = FigureCanvasTkAgg(fig, master=parent_frame)
    toolbar = NavigationToolbar2Tk(canvas, parent_frame, pack_toolbar=False)
    toolbar.update()
    toolbar.pack(side="top", fill="x")
    canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
    canvas.draw()


# ==============================
# ------------- UI -------------
# ==============================

class ScrollFrame(ttk.Frame):
    """
    Reusable scrollable frame that hosts the form.

    Why a custom scroll frame?
    - Tkinter's Frame doesn't scroll by itself; using a Canvas as a viewport lets us
      place a big Frame inside it and scroll that Frame. This pattern is robust and
      avoids having to manage lots of individual scrollbars.
    """
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        # Canvas + vertical scrollbar
        self._canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        self._vbar   = ttk.Scrollbar(self, orient="vertical", command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=self._vbar.set)

        self._vbar.pack(side="right", fill="y")
        self._canvas.pack(side="left", fill="both", expand=True)

        # Interior frame where widgets go
        self.interior = ttk.Frame(self._canvas)
        self._window_id = self._canvas.create_window((0, 0), window=self.interior, anchor="nw")

        # Keep scrollregion in sync
        def _on_configure_interior(event):
            self._canvas.configure(scrollregion=self._canvas.bbox("all"))
        def _on_configure_canvas(event):
            self._canvas.itemconfigure(self._window_id, width=event.width)

        self.interior.bind("<Configure>", _on_configure_interior)
        self._canvas.bind("<Configure>", _on_configure_canvas)

        # Mouse wheel scroll support
        self._bind_mouse_wheel(self._canvas)

    def _bind_mouse_wheel(self, widget):
        # Windows/Mac use <MouseWheel>; many Linux setups use Button-4/5
        widget.bind_all("<MouseWheel>", self._on_mousewheel, add="+")  # Windows/Mac
        widget.bind_all("<Button-4>",   self._on_mousewheel_linux, add="+")  # Linux up
        widget.bind_all("<Button-5>",   self._on_mousewheel_linux, add="+")  # Linux down
    def _on_mousewheel(self, event):
        self._canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    def _on_mousewheel_linux(self, event):
        self._canvas.yview_scroll(-3 if event.num == 4 else +3, "units")


class SpiralApp(tk.Tk):
    """
    Tkinter GUI
    -----------
    Geometry inputs (mm): Dout, W, S, N, K, M, Δz, base phase, twist per layer
    Header (cm): vol_res_cm, coil_res_cm, margin_cm
    Buttons: Draw (3D preview), Save (DXF + TXT)

    Notes for stability:
    - We cache the last computed geometry (self._last_arms_xy, self._last_zs)
      so Save can proceed even if Draw hasn't been pressed yet (it recomputes if needed).
    - Errors (bad inputs, geometry doesn't fit) raise dialogs instead of crashing.
    """
    def __init__(self):
        super().__init__()
        self.title("Ultimate Spiral Builder (multi-arm, multi-layer)")
        self.geometry("760x680")
        self.minsize(620, 520)

        # Root grid: row 0 = scrollable form + preview, row 1 = pinned button bar
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # ---- State vars with sensible defaults ----
        # Geometry (mm)
        self.var_Dout = tk.StringVar(value="150")       # outer diameter
        self.var_W    = tk.StringVar(value="0.25")      # trace width
        self.var_S    = tk.StringVar(value="0.25")      # spacing
        self.var_N    = tk.StringVar(value="100.0")      # turns per arm (fractional OK)
        self.var_K    = tk.StringVar(value="1")        # arms per layer
        self.var_M    = tk.StringVar(value="1")        # layers
        self.var_dz   = tk.StringVar(value="1.5")      # inter-layer distance (leave blank/<=0 => W+S)
        self.var_dz_list = tk.StringVar(value="")      # optional comma list overriding Δz
        self.var_base_phase = tk.StringVar(value="0")  # deg
        self.var_twist_layer= tk.StringVar(value="0")  # deg per layer
        self.var_pts  = tk.StringVar(value=str(PTS_PER_TURN))
        self.var_layer_dirs_summary = tk.StringVar(value="All layers: CCW")
        self.var_layer_arms_summary = tk.StringVar(value="All layers: K=1")
        self.var_layer_turns_summary = tk.StringVar(value="All layers: N=100.0")

        # Header (cm)
        self.var_vol_res  = tk.StringVar(value=str(DEF_VOL_RES_CM))
        self.var_coil_res = tk.StringVar(value=str(DEF_COIL_RES_CM))
        self.var_margin   = tk.StringVar(value=str(DEF_MARGIN_CM))

        # Output paths
        self.var_dxf_path = tk.StringVar(value="spiral_multiarm_fractional.dxf")
        self.var_txt_path = tk.StringVar(value="Wire_Sections.txt")

        # --- Layout ---
        self._build_layout()

        # Placeholders for last geometry (for quick redraws/saves without recompute if unchanged)
        self._last_arms_xy: Optional[List[Polyline2D]] = None
        self._last_zs: Optional[List[float]] = None
        self._last_section_dirs: Optional[List[str]] = None

        # Per-layer chirality state (list of "CCW"/"CW")
        self._layer_dirs: List[str] = []
        self._layer_arms: List[int] = []
        self._layer_turns: List[float] = []
        self.var_M.trace_add("write", self._on_layers_changed)
        self.var_K.trace_add("write", self._on_layers_changed)
        self.var_N.trace_add("write", self._on_layers_changed)
        self._on_layers_changed()

    # Build form + preview + buttons
    def _build_layout(self):
        # Scroll container
        self.scroll = ScrollFrame(self)
        self.scroll.grid(row=0, column=0, sticky="nsew")

        f = self.scroll.interior

        # Form grid
        row = 0
        ttk.Label(f, text="Geometry (mm)", font=("Segoe UI", 11, "bold")).grid(row=row, column=0, sticky="w", pady=(8,4)); row+=1

        self._add_labeled_entry(f, "Outside diameter Dout [mm]:", self.var_Dout, row); row+=1
        self._add_labeled_entry(f, "Trace width W [mm]:",         self.var_W,    row); row+=1
        self._add_labeled_entry(f, "Spacing S [mm]:",             self.var_S,    row); row+=1
        self._add_labeled_entry(f, "Turns per arm N (fractional):", self.var_N,  row); row+=1
        self._add_labeled_entry(f, "Arms per layer K:",           self.var_K,    row); row+=1
        self._add_labeled_entry(f, "Number of layers M:",         self.var_M,    row); row+=1
        self._add_layer_arms_row(f, row); row+=1
        self._add_layer_turns_row(f, row); row+=1
        self._add_labeled_entry(f, "Inter-layer distance Δz [mm] (≤0 or blank → W+S):", self.var_dz, row); row+=1
        self._add_labeled_entry(
            f,
            "Custom Δz list per gap [mm] (comma-separated; overrides Δz):",
            self.var_dz_list,
            row,
        ); row+=1
        self._add_layer_dir_row(f, row); row+=1
        self._add_layer_kn_row(f, row); row+=1
        self._add_labeled_entry(f, "Base phase [deg]:",           self.var_base_phase, row); row+=1
        self._add_labeled_entry(f, "Twist per layer [deg]:",      self.var_twist_layer, row); row+=1
        self._add_labeled_entry(f, "Sampling PTS_PER_TURN:",      self.var_pts,  row); row+=1

        ttk.Separator(f, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", pady=8); row+=1

        ttk.Label(f, text="Header for Wire_Sections.txt (cm)", font=("Segoe UI", 11, "bold")).grid(row=row, column=0, sticky="w", pady=(8,4)); row+=1
        self._add_labeled_entry(f, "vol_res_cm:",  self.var_vol_res,  row); row+=1
        self._add_labeled_entry(f, "coil_res_cm:", self.var_coil_res, row); row+=1
        self._add_labeled_entry(f, "margin_cm:",   self.var_margin,   row); row+=1

        ttk.Separator(f, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", pady=8); row+=1

        ttk.Label(f, text="Output files", font=("Segoe UI", 11, "bold")).grid(row=row, column=0, sticky="w", pady=(8,4)); row+=1
        self._add_labeled_entry(f, "DXF path:", self.var_dxf_path, row, width=36); row+=1
        self._add_labeled_entry(f, "TXT path:", self.var_txt_path, row, width=36); row+=1

        ttk.Separator(f, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", pady=8); row+=1

        # 3D preview host
        ttk.Label(f, text="3D Preview").grid(row=row, column=0, sticky="w"); row+=1
        self.preview_host = ttk.Frame(f, height=380)
        self.preview_host.grid(row=row, column=0, columnspan=2, sticky="nsew")
        f.grid_rowconfigure(row, weight=1); f.grid_columnconfigure(0, weight=1)
        row+=1

        # Pinned bottom button bar
        btnbar = ttk.Frame(self)
        btnbar.grid(row=1, column=0, sticky="ew")
        btnbar.grid_columnconfigure(0, weight=1)
        ttk.Button(btnbar, text="Draw (preview)", command=self.on_draw).grid(row=0, column=0, padx=8, pady=6, sticky="w")
        ttk.Button(btnbar, text="Save (DXF + TXT)", command=self.on_save).grid(row=0, column=1, padx=8, pady=6, sticky="e")

    def _add_labeled_entry(self, parent, label, var, row, width=16):
        frm = ttk.Frame(parent)
        frm.grid(row=row, column=0, sticky="w", padx=4, pady=3)
        ttk.Label(frm, text=label, width=36, anchor="w").pack(side="left")
        e = ttk.Entry(frm, textvariable=var, width=width)
        e.pack(side="left")

    def _add_layer_dir_row(self, parent, row):
        frm = ttk.Frame(parent)
        frm.grid(row=row, column=0, sticky="w", padx=4, pady=3)
        ttk.Label(frm, text="Layer directions (chirality):", width=36, anchor="w").pack(side="left")
        ttk.Button(frm, text="Set…", command=self._open_layer_dir_dialog).pack(side="left")
        ttk.Label(frm, textvariable=self.var_layer_dirs_summary).pack(side="left", padx=8)

    def _add_layer_arms_row(self, parent, row):
        frm = ttk.Frame(parent)
        frm.grid(row=row, column=0, sticky="w", padx=4, pady=3)
        ttk.Label(frm, text="Arms per layer overrides:", width=36, anchor="w").pack(side="left")
        ttk.Button(frm, text="Set…", command=self._open_layer_arms_dialog).pack(side="left")
        ttk.Label(frm, textvariable=self.var_layer_arms_summary).pack(side="left", padx=8)

    def _add_layer_turns_row(self, parent, row):
        frm = ttk.Frame(parent)
        frm.grid(row=row, column=0, sticky="w", padx=4, pady=3)
        ttk.Label(frm, text="Turns per layer overrides:", width=36, anchor="w").pack(side="left")
        ttk.Button(frm, text="Set…", command=self._open_layer_turns_dialog).pack(side="left")
        ttk.Label(frm, textvariable=self.var_layer_turns_summary).pack(side="left", padx=8)

    def _on_layers_changed(self, *args):
        try:
            M = int(self.var_M.get())
        except Exception:
            M = 0
        try:
            K = int(self.var_K.get())
        except Exception:
            K = 1
        try:
            N = float(self.var_N.get())
        except Exception:
            N = 1.0

        self._ensure_layer_dir_length(M)
        self._ensure_layer_arms_length(M, K)
        self._ensure_layer_turns_length(M, N)

        self.var_layer_dirs_summary.set(self._format_layer_dir_summary())
        self.var_layer_arms_summary.set(self._format_layer_arms_summary())
        self.var_layer_turns_summary.set(self._format_layer_turns_summary())

    def _ensure_layer_dir_length(self, M: int):
        M = max(0, int(M))
        cur = list(self._layer_dirs)
        if len(cur) < M:
            cur.extend(["CCW"] * (M - len(cur)))
        else:
            cur = cur[:M]
        self._layer_dirs = cur

    def _ensure_layer_kn_length(self, M: int):
        M = max(0, int(M))
        k_list = list(self._layer_K_overrides)
        n_list = list(self._layer_N_overrides)
        if len(k_list) < M:
            k_list.extend([None] * (M - len(k_list)))
        else:
            k_list = k_list[:M]
        if len(n_list) < M:
            n_list.extend([None] * (M - len(n_list)))
        else:
            n_list = n_list[:M]
        self._layer_K_overrides = k_list
        self._layer_N_overrides = n_list

    def _format_layer_dir_summary(self) -> str:
        if not self._layer_dirs:
            return "All layers: CCW"
        unique = set(self._layer_dirs)
        if len(unique) == 1:
            val = next(iter(unique))
            return f"All layers: {val}"
        preview = ", ".join(f"L{idx}:{val}" for idx, val in enumerate(self._layer_dirs))
        return f"Layer dirs → {preview}"

    def _ensure_layer_arms_length(self, M: int, default: int):
        M = max(0, int(M))
        default = max(1, int(default))
        cur = list(self._layer_arms)
        if len(cur) < M:
            cur.extend([default] * (M - len(cur)))
        else:
            cur = cur[:M]
        self._layer_arms = [max(1, int(v)) for v in cur]

    def _format_layer_arms_summary(self) -> str:
        if not self._layer_arms:
            return "All layers: K unset"
        unique = set(self._layer_arms)
        if len(unique) == 1:
            return f"All layers: K={next(iter(unique))}"
        preview = ", ".join(f"L{idx}:K={val}" for idx, val in enumerate(self._layer_arms))
        return f"Arms → {preview}"

    def _ensure_layer_turns_length(self, M: int, default: float):
        M = max(0, int(M))
        try:
            default_val = float(default)
        except Exception:
            default_val = 1.0
        cur = list(self._layer_turns)
        if len(cur) < M:
            cur.extend([default_val] * (M - len(cur)))
        else:
            cur = cur[:M]
        self._layer_turns = [max(0.0001, float(v)) for v in cur]

    def _format_layer_turns_summary(self) -> str:
        if not self._layer_turns:
            return "All layers: N unset"
        unique = set(self._layer_turns)
        if len(unique) == 1:
            return f"All layers: N={next(iter(unique))}"
        preview = ", ".join(f"L{idx}:N={val}" for idx, val in enumerate(self._layer_turns))
        return f"Turns → {preview}"

    def _open_layer_dir_dialog(self):
        try:
            M = int(self.var_M.get())
        except Exception:
            messagebox.showerror("Layer directions", "Please enter a valid layer count first.")
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
            self.var_layer_dirs_summary.set(self._format_layer_dir_summary())
            dlg.destroy()

        ttk.Button(btns, text="Cancel", command=dlg.destroy).grid(row=0, column=0, sticky="e", padx=4)
        ttk.Button(btns, text="OK", command=_apply_and_close).grid(row=0, column=1, sticky="w", padx=4)

    def _open_layer_arms_dialog(self):
        try:
            M = int(self.var_M.get())
            default_K = int(self.var_K.get())
        except Exception:
            messagebox.showerror("Layer arms", "Please enter valid layer and arm counts first.")
            return
        if M <= 0 or default_K <= 0:
            messagebox.showerror("Layer arms", "Layer count and K must be ≥ 1.")
            return

        self._ensure_layer_arms_length(M, default_K)

        dlg = tk.Toplevel(self)
        dlg.title("Arms per layer")
        dlg.transient(self)
        dlg.grab_set()

        body = ttk.Frame(dlg)
        body.pack(fill="both", expand=True, padx=10, pady=10)

        entries: List[tk.Spinbox] = []
        for idx in range(M):
            row = ttk.Frame(body)
            row.pack(fill="x", pady=3)
            ttk.Label(row, text=f"Layer {idx}").pack(side="left", padx=(0, 8))
            spin = tk.Spinbox(row, from_=1, to=9999, width=6)
            spin.delete(0, "end")
            spin.insert(0, str(self._layer_arms[idx]))
            spin.pack(side="left")
            entries.append(spin)

        btns = ttk.Frame(dlg)
        btns.pack(fill="x", padx=10, pady=(0, 10))
        btns.grid_columnconfigure(0, weight=1)
        btns.grid_columnconfigure(1, weight=1)

        def _apply_and_close():
            try:
                vals = [max(1, int(spin.get())) for spin in entries]
            except Exception:
                messagebox.showerror("Layer arms", "Please enter integer values ≥ 1.", parent=dlg)
                return
            self._layer_arms = vals
            self.var_layer_arms_summary.set(self._format_layer_arms_summary())
            dlg.destroy()

        ttk.Button(btns, text="Cancel", command=dlg.destroy).grid(row=0, column=0, sticky="e", padx=4)
        ttk.Button(btns, text="OK", command=_apply_and_close).grid(row=0, column=1, sticky="w", padx=4)

    def _open_layer_turns_dialog(self):
        try:
            M = int(self.var_M.get())
            default_N = float(self.var_N.get())
        except Exception:
            messagebox.showerror("Layer turns", "Please enter valid layer count and N first.")
            return
        if M <= 0 or default_N <= 0:
            messagebox.showerror("Layer turns", "Layer count and N must be > 0.")
            return

        self._ensure_layer_turns_length(M, default_N)

        dlg = tk.Toplevel(self)
        dlg.title("Turns per layer")
        dlg.transient(self)
        dlg.grab_set()

        body = ttk.Frame(dlg)
        body.pack(fill="both", expand=True, padx=10, pady=10)

        entries: List[tk.Entry] = []
        for idx in range(M):
            row = ttk.Frame(body)
            row.pack(fill="x", pady=3)
            ttk.Label(row, text=f"Layer {idx}").pack(side="left", padx=(0, 8))
            ent = ttk.Entry(row, width=10)
            ent.insert(0, str(self._layer_turns[idx]))
            ent.pack(side="left")
            entries.append(ent)

        btns = ttk.Frame(dlg)
        btns.pack(fill="x", padx=10, pady=(0, 10))
        btns.grid_columnconfigure(0, weight=1)
        btns.grid_columnconfigure(1, weight=1)

        def _apply_and_close():
            try:
                vals = [float(ent.get()) for ent in entries]
                if any(v <= 0 for v in vals):
                    raise ValueError
            except Exception:
                messagebox.showerror("Layer turns", "Please enter numeric values > 0.", parent=dlg)
                return
            self._layer_turns = vals
            self.var_layer_turns_summary.set(self._format_layer_turns_summary())
            dlg.destroy()

        ttk.Button(btns, text="Cancel", command=dlg.destroy).grid(row=0, column=0, sticky="e", padx=4)
        ttk.Button(btns, text="OK", command=_apply_and_close).grid(row=0, column=1, sticky="w", padx=4)

    # --- helpers to read and validate inputs ---
    def _read_inputs(self) -> SpiralInputs:
        try:
            Dout = float(self.var_Dout.get())
            W    = float(self.var_W.get())
            S    = float(self.var_S.get())
            N    = float(self.var_N.get())
            K    = int(self.var_K.get())
            M    = int(self.var_M.get())
            dz_s = self.var_dz.get().strip()
            dz   = float(dz_s) if dz_s != "" else None
            dz_list_raw = self.var_dz_list.get().strip()
            dz_list = None
            if dz_list_raw:
                cleaned = dz_list_raw.replace(";", ",")
                parts = [p.strip() for p in cleaned.split(",") if p.strip() != ""]
                if not parts:
                    dz_list = None
                else:
                    dz_list = [float(p) for p in parts]
                    if any(v <= 0 for v in dz_list):
                        raise ValueError("Custom Δz entries must be > 0.")
                    expected = max(0, M - 1)
                    if len(dz_list) != expected:
                        raise ValueError(
                            f"Custom Δz list must have exactly {expected} entries for {M} layers."
                        )
            base = float(self.var_base_phase.get())
            twist= float(self.var_twist_layer.get())
            pts  = int(self.var_pts.get())

            if Dout <= 0 or W <= 0 or S < 0 or N <= 0 or K <= 0 or M <= 0:
                raise ValueError

        except Exception:
            raise ValueError("Please enter valid positive numbers (S can be 0).")

        self._ensure_layer_dir_length(M)
        self._ensure_layer_arms_length(M, K)
        self._ensure_layer_turns_length(M, N)
        layer_dirs = list(self._layer_dirs)
        layer_arms = list(self._layer_arms)
        layer_turns = list(self._layer_turns)

        return SpiralInputs(
            Dout_mm=Dout, W_mm=W, S_mm=S, N_turns=N, K_arms=K, M_layers=M, dz_mm=dz,
            base_phase_deg=base, twist_per_layer_deg=twist, pts_per_turn=pts,
            layer_gaps_mm=dz_list,
            layer_dirs=layer_dirs,
            layer_arms=layer_arms,
            layer_turns=layer_turns,
        )

    def _read_header(self) -> SimHeader:
        try:
            vr = float(self.var_vol_res.get()); cr = float(self.var_coil_res.get()); mg = float(self.var_margin.get())
            if vr <= 0 or cr <= 0 or mg < 0:
                raise ValueError
        except Exception:
            raise ValueError("Header fields must be positive numbers (margin_cm can be 0).")
        return SimHeader(vr, cr, mg)

    # --- actions ---
    def on_draw(self):
        """
        Build geometry and show the preview.
        Robust error handling shows a friendly message if geometry does not fit, etc.
        """
        try:
            params = self._read_inputs()
            arms_xy, zs, dirs = build_multiarm_geometry(params)
        except Exception as e:
            messagebox.showerror("Draw failed", str(e))
            return

        # Store last geometry to allow saving immediately
        self._last_arms_xy = arms_xy
        self._last_zs = zs
        self._last_section_dirs = dirs

        try:
            plot_3d_in_window(self.preview_host, arms_xy, zs, title="Spiral 3D preview")
        except Exception as e:
            messagebox.showerror("Plot error", f"Failed to render preview:\n{e}")

    def on_save(self):
        """
        Save DXF + TXT using current geometry (rebuild if needed).
        """
        # Ensure geometry exists (if user presses Save first)
        if self._last_arms_xy is None or self._last_zs is None or self._last_section_dirs is None:
            try:
                params = self._read_inputs()
                arms_xy, zs, dirs = build_multiarm_geometry(params)
            except Exception as e:
                messagebox.showerror("Save failed", str(e))
                return
        else:
            arms_xy, zs, dirs = self._last_arms_xy, self._last_zs, self._last_section_dirs

        # Read header and paths
        try:
            sim = self._read_header()
        except Exception as e:
            messagebox.showerror("Header error", str(e)); return

        dxf_path = self.var_dxf_path.get().strip() or "spiral_multiarm_fractional.dxf"
        txt_path = self.var_txt_path.get().strip() or "Wire_Sections.txt"

        # enforce extensions (non-destructive)
        if not dxf_path.lower().endswith(".dxf"):
            dxf_path += ".dxf"
        if not txt_path.lower().endswith(".txt"):
            txt_path += ".txt"

        # sanity checks
        if not arms_xy or not zs:
            messagebox.showerror("Save failed", "Geometry is empty."); return

        try:
            # Always write TXT first so the converter has an input.
            write_wire_sections_txt(
                arms_xy,
                zs,
                txt_path,
                sim,
                I_amp=I_AMP,
                box=BOX_MODE,
                section_dirs=dirs,
            )
            # Now convert TXT → DXF (with robust checks inside)
            write_simple_dxf_lwpolylines(txt_path, dxf_path)
        except Exception as e:
            messagebox.showerror("Write failed", f"Could not write outputs:\n{e}")
            return

        messagebox.showinfo("Saved", f"DXF → {dxf_path}\nTXT → {txt_path}\n\nSections (arms): {len(arms_xy)}")

# ==============================
# ------------ MAIN ------------
# ==============================

if __name__ == "__main__":
    app = SpiralApp()
    app.mainloop()
