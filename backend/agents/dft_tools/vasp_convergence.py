from __future__ import annotations
from dataclasses import dataclass
from typing import Union, List, Tuple, Dict, Any, Optional

from langchain_core.tools import tool

# Reuse your existing tools
from backend.agents.dft_tools.pmg_vasp import write_vasp_scf, parse_vasp_energy
from backend.agents.dft_tools.pmg_run import run_local
from utils import _norm_kpts, _mev

@dataclass
class ConvergenceResult:
    grid: List[Any]                 # encuts (int) or kpt strings
    energies: List[float]           # total energies (eV)
    best_index: int                 # index of chosen setting
    best_value: Any                 # chosen encut or kpts
    deltas_mev: List[float]         # successive ΔE in meV/atom (if natoms given) else meV/cell

    def as_dict(self) -> Dict[str, Any]:
        return {
            "grid": self.grid,
            "energies_ev": self.energies,
            "deltas_mev": self.deltas_mev,
            "best_index": self.best_index,
            "best_value": self.best_value,
        }


@tool
def converge_encut(
    struct: str,
    workdir: str,
    encut_list: List[int],
    *,
    kpts: Union[str, List[int], Tuple[int,int,int]] = "4x4x4",
    natoms: Optional[int] = None,
    tol_mev: float = 1.0,
    ismear: int = 1,
    sigma: float = 0.2,
) -> Dict[str, Any]:
    """
    Run SCF calculations over a list of ENCUT values and choose a converged ENCUT.

    Args:
      struct: Path to POSCAR/CONTCAR (input structure).
      workdir: Directory under which subfolders per-encut will be created.
      encut_list: e.g. [300, 400, 500, 600]
      kpts: k-point mesh; accepts 'NxNxN', 'N N N', or [N,N,N].
      natoms: Number of atoms in the structure (for meV/atom deltas). If None, deltas are meV/cell.
      tol_mev: Convergence threshold on successive ΔE (meV/atom if natoms else meV/cell).
      ismear, sigma: Smearing settings for metals (adjust as needed).

    Returns:
      dict with {grid, energies_ev, deltas_mev, best_index, best_value}
    """
    kmesh = _norm_kpts(kpts)
    energies: List[float] = []
    deltas_mev: List[float] = []

    # Ensure deterministic order
    grid = list(sorted(set(int(x) for x in encut_list)))

    for enc in grid:
        subdir = f"{workdir.rstrip('/')}/ecut_{enc}"
        # 1) write inputs
        write_vasp_scf.invoke({
            "struct": struct,
            "workdir": subdir,
            "kpts": kmesh,
            "encut": enc,
            "ismear": ismear,
            "sigma": sigma,
        })
        # 2) run
        run_local.invoke({"cmd": "vasp_std", "workdir": subdir})
        # 3) parse energy
        e = parse_vasp_energy.invoke({"workdir": subdir})
        energies.append(float(e))

        if len(energies) > 1:
            delta = abs(energies[-1] - energies[-2])
            deltas_mev.append(_mev(delta, natoms))

    best_idx = len(grid) - 1
    for i in range(1, len(energies)):
        mevd = _mev(abs(energies[i] - energies[i - 1]), natoms)
        if mevd <= tol_mev:
            best_idx = i
            break

    result = ConvergenceResult(
        grid=grid,
        energies=energies,
        deltas_mev=deltas_mev,
        best_index=best_idx,
        best_value=grid[best_idx],
    )
    return result.as_dict()
@tool
def converge_kpoints(
    struct: str,
    workdir: str,
    kpt_list: List[Union[str, List[int], Tuple[int,int,int]]],
    encut: int,
    *,
    natoms: Union[int, None] = None,
    tol_mev: float = 1.0,
    ismear: int = 1,
    sigma: float = 0.2,
) -> Dict[str, Any]:
    """
    Run SCF calculations over a list of k-point meshes and choose a converged mesh.

    Args:
      struct: Path to POSCAR/CONTCAR (input structure).
      workdir: Directory under which subfolders per-kpts will be created.
      kpt_list: e.g. ["2x2x2","3x3x3","4x4x4","5x5x5"] (len 3 lists/tuples work too)
      encut: Fixed ENCUT (eV) chosen from converge_encut.
      natoms: Number of atoms in structure (for meV/atom), else meV/cell.
      tol_mev: Convergence threshold on successive ΔE.
      ismear, sigma: Smearing settings.

    Returns:
      dict with {grid, energies_ev, deltas_mev, best_index, best_value}
    """
    # normalize and sort by total k-points (rough measure)
    normed = [(kp, _norm_kpts(kp)) for kp in kpt_list]
    def ksize(s: str) -> int:
        a, b, c = (int(x) for x in s.split("x"))
        return a * b * c
    grid = [s for _, s in sorted(normed, key=lambda kv: ksize(kv[1]))]

    energies: List[float] = []
    deltas_mev: List[float] = []

    for km in grid:
        subdir = f"{workdir.rstrip('/')}/k_{km.replace('x','_')}"
        # 1) write inputs
        write_vasp_scf.invoke({
            "struct": struct,
            "workdir": subdir,
            "kpts": km,
            "encut": encut,
            "ismear": ismear,
            "sigma": sigma,
        })
        # 2) run
        run_local.invoke({"cmd": "vasp_std", "workdir": subdir})
        # 3) parse
        e = parse_vasp_energy.invoke({"workdir": subdir})
        energies.append(float(e))

        if len(energies) > 1:
            delta = abs(energies[-1] - energies[-2])
            deltas_mev.append(_mev(delta, natoms))

    # choose first index where |ΔE| < tol_mev
    best_idx = len(grid) - 1
    for i in range(1, len(energies)):
        mevd = _mev(abs(energies[i] - energies[i - 1]), natoms)
        if mevd <= tol_mev:
            best_idx = i
            break

    result = ConvergenceResult(
        grid=grid,
        energies=energies,
        deltas_mev=deltas_mev,
        best_index=best_idx,
        best_value=grid[best_idx],
    )
    return result.as_dict()