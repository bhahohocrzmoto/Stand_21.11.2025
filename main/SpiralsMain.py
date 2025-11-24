#!/usr/bin/env python3
"""
Central orchestration GUI for spiral generation, solver automation, and plotting.

This panel keeps the existing specialised UIs but wires them together so a user can:
1) Launch the spiral batch UI to generate geometry + Address.txt
2) Verify Address.txt contents
3) Run FastSolver conversion + solver automation with a chosen permittivity
4) Configure ports in a friendlier popup and generate plots
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

REPO_ROOT = Path(__file__).resolve().parents[1]
SPIRAL_UI = REPO_ROOT / "SpiralGeometryGeneration" / "Spiral_Batch_Variants_UI_16.11.2025.py"
FAST_UI = REPO_ROOT / "FastSolver" / "Automation" / "fast_solver_batch_ui.py"
AUTOMATE = REPO_ROOT / "FastSolver" / "Automation" / "automate_solvers.py"
PLOT_GEN = REPO_ROOT / "FastSolver" / "PlotGeneration" / "PlotGeneration.py"

sys.path.insert(0, str(REPO_ROOT))
from FastSolver.PlotGeneration import PlotGeneration as PG  # type: ignore  # noqa: E402


# ---------------- helpers -----------------

def read_address_entries(address_file: Path) -> List[Path]:
    cleaned = address_file.read_text().splitlines()
    entries: List[Path] = []
    for line in cleaned:
        stripped = line.strip().strip('"').strip("'")
        if not stripped:
            continue
        p = Path(stripped)
        if not p.is_absolute():
            p = address_file.parent / p
        entries.append(p.resolve())
    return entries


def log_subprocess(cmd: List[str], log_widget: tk.Text) -> bool:
    log_widget.insert("end", f"\n$ {' '.join(cmd)}\n")
    log_widget.see("end")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if proc.stdout:
            log_widget.insert("end", proc.stdout)
        if proc.stderr:
            log_widget.insert("end", proc.stderr)
        log_widget.see("end")
        return True
    except subprocess.CalledProcessError as exc:  # noqa: BLE001
        log_widget.insert("end", exc.stdout or "")
        log_widget.insert("end", exc.stderr or "")
        log_widget.insert("end", f"Command failed: {exc}\n")
        log_widget.see("end")
        messagebox.showerror("Command failed", f"{cmd[0]} exited with status {exc.returncode}")
        return False


# ---------------- Port configuration popup -----------------

class PortsPopup(tk.Toplevel):
    def __init__(self, master: tk.Tk, address_file: Path, log_widget: tk.Text):
        super().__init__(master)
        self.title("Ports & plotting")
        self.address_file = address_file
        self.log = log_widget
        self.geometry("840x520")
        self.transient(master)
        self.grab_set()

        self.spiral_paths = self._load_spiral_paths()
        self.ports: Dict[Path, Dict[str, Dict[str, object]]] = self._load_existing_ports()

        self._build_ui()

    def _load_spiral_paths(self) -> List[Path]:
        try:
            paths = read_address_entries(self.address_file)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Address read error", str(exc), parent=self)
            return []
        existing = [p for p in paths if (p / "FastSolver" / "CapacitanceMatrix.txt").exists()]
        if not existing:
            messagebox.showwarning("No solved spirals", "No FastSolver outputs found yet.", parent=self)
        return existing

    def _load_existing_ports(self) -> Dict[Path, Dict[str, Dict[str, object]]]:
        configs: Dict[Path, Dict[str, Dict[str, object]]] = {}
        for path in self.spiral_paths:
            cfg_path = PG.ensure_analysis_dirs(path)["ports_config"]
            if cfg_path.exists():
                try:
                    data = json.loads(cfg_path.read_text())
                    configs[path] = data.get("ports", {})
                except Exception:  # noqa: BLE001
                    configs[path] = {}
        return configs

    def _build_ui(self):
        left = ttk.Frame(self)
        left.pack(side="left", fill="y", padx=8, pady=8)

        ttk.Label(left, text="Spiral variations").pack(anchor="w")
        self.tree = ttk.Treeview(left, columns=("name", "conductors"), show="headings", height=18)
        self.tree.heading("name", text="Folder")
        self.tree.heading("conductors", text="# conductors")
        self.tree.column("name", width=340)
        self.tree.column("conductors", width=90, anchor="center")
        self.tree.pack(fill="y", expand=True)
        for path in self.spiral_paths:
            n = self._count_conductors(path)
            self.tree.insert("", "end", iid=str(path), values=(path.name, n))
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        right = ttk.Frame(self)
        right.pack(side="right", fill="both", expand=True, padx=8, pady=8)

        form = ttk.LabelFrame(right, text="Port definition")
        form.pack(fill="x")
        row = ttk.Frame(form); row.pack(fill="x", pady=4, padx=6)
        ttk.Label(row, text="Port name:").pack(side="left")
        self.var_port_name = tk.StringVar(value="Port1")
        ttk.Entry(row, textvariable=self.var_port_name, width=20).pack(side="left", padx=6)

        ttk.Label(row, text="Type:").pack(side="left")
        self.var_type = tk.StringVar(value="series")
        ttk.Combobox(row, textvariable=self.var_type, state="readonly", values=("series", "parallel", "custom_pm1"), width=14).pack(side="left", padx=6)

        vec_row = ttk.Frame(form); vec_row.pack(fill="x", pady=4, padx=6)
        ttk.Label(vec_row, text="Signs vector (+1/-1, space separated):").pack(side="left")
        self.var_signs = tk.StringVar(value="")
        ttk.Entry(vec_row, textvariable=self.var_signs, width=55).pack(side="left", padx=6)

        btns = ttk.Frame(right)
        btns.pack(fill="x", pady=6)
        ttk.Button(btns, text="Apply to selected", command=self._apply_to_selected).pack(side="left", padx=4)
        ttk.Button(btns, text="Apply to all", command=self._apply_to_all).pack(side="left", padx=4)
        ttk.Button(btns, text="Remove port from selected", command=self._remove_from_selected).pack(side="left", padx=4)

        self.summary = tk.Text(right, height=12)
        self.summary.pack(fill="both", expand=True, pady=(6, 0))
        self._refresh_summary()

        action = ttk.Frame(self)
        action.pack(fill="x", side="bottom", pady=8, padx=10)
        ttk.Button(action, text="Save & generate plots", command=self._save_and_run).pack(side="right", padx=6)
        ttk.Button(action, text="Cancel", command=self.destroy).pack(side="right")

        # Select first item by default
        if self.tree.get_children():
            self.tree.selection_set(self.tree.get_children()[0])
            self._on_select()

    def _on_select(self, event=None):
        sel = self.tree.selection()
        if not sel:
            return
        path = Path(sel[0])
        n = self._count_conductors(path)
        self.var_signs.set(" ".join(["+1"] * n))

    def _count_conductors(self, path: Path) -> int:
        cap = path / "FastSolver" / "CapacitanceMatrix.txt"
        try:
            matrix = PG.load_capacitance_matrix(cap)
            return matrix.shape[0]
        except Exception:
            return 0

    def _apply_to_selected(self):
        sel = self.tree.selection()
        if not sel:
            return
        self._apply_to_paths([Path(item) for item in sel])

    def _apply_to_all(self):
        self._apply_to_paths([Path(item) for item in self.tree.get_children()])

    def _apply_to_paths(self, paths: List[Path]):
        name = self.var_port_name.get().strip()
        if not name:
            messagebox.showwarning("Missing name", "Please enter a port name.", parent=self)
            return
        signs_raw = [s for s in self.var_signs.get().replace(",", " ").split() if s]
        for path in paths:
            n = self._count_conductors(path)
            if n and len(signs_raw) != n:
                messagebox.showwarning(
                    "Size mismatch",
                    f"{path.name}: expected {n} entries, got {len(signs_raw)}",
                    parent=self,
                )
                return
        for path in paths:
            self.ports.setdefault(path, {})[name] = {
                "type": self.var_type.get(),
                "signs": [float(s) for s in signs_raw],
            }
        self._refresh_summary()

    def _remove_from_selected(self):
        sel = self.tree.selection()
        if not sel:
            return
        path = Path(sel[0])
        name = self.var_port_name.get().strip()
        if path in self.ports and name in self.ports[path]:
            self.ports[path].pop(name)
            self._refresh_summary()

    def _refresh_summary(self):
        self.summary.delete("1.0", "end")
        for path, ports in self.ports.items():
            self.summary.insert("end", f"{path.name}:\n")
            for pname, pdata in ports.items():
                self.summary.insert("end", f"  - {pname}: {pdata.get('type')} | signs={pdata.get('signs')}\n")
        self.summary.see("end")

    def _save_and_run(self):
        if not self.ports:
            messagebox.showwarning("No ports", "Define at least one port before generating plots.", parent=self)
            return
        for path, ports in self.ports.items():
            dirs = PG.ensure_analysis_dirs(path)
            cfg_path = dirs["ports_config"]
            cfg_path.write_text(json.dumps({"ports": ports}, indent=2))
        self.log.insert("end", "Saved ports_config.json for all selected spirals.\n")
        self.log.see("end")
        self._run_plots()
        self.destroy()

    def _run_plots(self):
        records: List[Dict[str, object]] = []
        for path in self.spiral_paths:
            ports = self.ports.get(path, {})
            PG.process_spiral(path, records, ports_override=ports, auto_reuse_ports=True)
        summary_path = self.address_file.parent / "ports_summary.csv"
        if records:
            import pandas as pd  # local import to avoid heavy import at startup

            pd.DataFrame(records).to_csv(summary_path, index=False)
            self.log.insert("end", f"Plot generation complete. Summary → {summary_path}\n")
        else:
            self.log.insert("end", "Plot generation finished (no records written).\n")
        self.log.see("end")


# ---------------- Main app -----------------

class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Spirals main panel")
        self.geometry("940x720")

        self.var_address = tk.StringVar()
        self.var_eps = tk.StringVar(value="3.5")

        self._build_ui()

    def _build_ui(self):
        top = ttk.LabelFrame(self, text="1) Geometry generation")
        top.pack(fill="x", padx=10, pady=8)
        ttk.Label(top, text="Use the existing batch UI to generate spirals and Address.txt").pack(side="left", padx=6)
        ttk.Button(top, text="Open generator", command=self._launch_spiral_ui).pack(side="right", padx=6)

        mid = ttk.LabelFrame(self, text="2) Address & solver setup")
        mid.pack(fill="x", padx=10, pady=8)

        row = ttk.Frame(mid); row.pack(fill="x", pady=4, padx=6)
        ttk.Label(row, text="Address.txt:").pack(side="left")
        ttk.Entry(row, textvariable=self.var_address, width=80).pack(side="left", padx=6)
        ttk.Button(row, text="Browse…", command=self._browse_address).pack(side="left")
        ttk.Button(row, text="Verify", command=self._verify_address).pack(side="left", padx=4)

        eps_row = ttk.Frame(mid); eps_row.pack(fill="x", pady=4, padx=6)
        ttk.Label(eps_row, text="Permittivity (eps_r):").pack(side="left")
        ttk.Entry(eps_row, textvariable=self.var_eps, width=12).pack(side="left", padx=6)

        solver = ttk.LabelFrame(self, text="3) Solve")
        solver.pack(fill="x", padx=10, pady=8)
        ttk.Button(solver, text="Run conversion + solvers", command=self._run_pipeline).pack(side="left", padx=6, pady=6)
        ttk.Button(solver, text="Configure ports / plots", command=self._open_ports_popup).pack(side="left", padx=6)

        log_frame = ttk.LabelFrame(self, text="Log")
        log_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.log = tk.Text(log_frame, wrap="word")
        self.log.pack(fill="both", expand=True)

    def _launch_spiral_ui(self):
        if not SPIRAL_UI.exists():
            messagebox.showerror("Missing script", f"Cannot find {SPIRAL_UI}")
            return
        try:
            proc = subprocess.Popen(
                [sys.executable, str(SPIRAL_UI)],
                cwd=str(SPIRAL_UI.parent),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Launch failed", str(exc))
            return

        self.log.insert("end", f"Launched spiral generator UI (pid {proc.pid}).\n")
        self.log.see("end")

        def _check_proc():
            ret = proc.poll()
            if ret is None:
                return
            out, err = proc.communicate()
            if ret != 0:
                messagebox.showerror(
                    "Generator exited", err or out or f"Exited with status {ret}", parent=self
                )
                self.log.insert("end", err or out or f"Generator exited with {ret}\n")
            elif out or err:
                self.log.insert("end", (out or "") + (err or ""))
            self.log.see("end")

        # Surface immediate failures instead of silently ignoring them
        self.after(1200, _check_proc)

    def _browse_address(self):
        path = filedialog.askopenfilename(title="Select Address.txt", filetypes=[("Address", "Address.txt"), ("Text", "*.txt")])
        if path:
            self.var_address.set(path)

    def _verify_address(self):
        path = Path(self.var_address.get())
        if not path.is_file():
            messagebox.showerror("Address missing", "Select a valid Address.txt first.")
            return False
        try:
            entries = read_address_entries(path)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Invalid Address.txt", str(exc))
            return False
        missing = [p for p in entries if not p.exists()]
        if missing:
            messagebox.showwarning("Missing folders", "\n".join(str(m) for m in missing))
            return False
        messagebox.showinfo("Address check", f"{len(entries)} folders found.")
        return True

    def _run_pipeline(self):
        if not self._verify_address():
            return
        addr = Path(self.var_address.get())
        eps = self.var_eps.get().strip() or "1"
        # 1) Convert Wire_Sections to solver formats
        ok = log_subprocess([sys.executable, str(FAST_UI), "--non-interactive", str(addr)], self.log)
        if not ok:
            return
        # 2) Run solvers
        ok = log_subprocess([sys.executable, str(AUTOMATE), str(addr), eps], self.log)
        if ok:
            messagebox.showinfo("Solvers complete", "FastHenry/FasterCap runs finished.")
            self._open_ports_popup()

    def _open_ports_popup(self):
        if not self.var_address.get():
            messagebox.showwarning("Address needed", "Select Address.txt first.")
            return
        popup = PortsPopup(self, Path(self.var_address.get()), self.log)
        popup.wait_window()


def main():
    app = MainApp()
    app.mainloop()


if __name__ == "__main__":
    main()
