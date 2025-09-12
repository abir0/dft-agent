from __future__ import annotations
from typing import List, Dict, Any, Tuple

def _norm_kpts(k: str | List[int] | Tuple[int,int,int]) -> str:
    """Normalize kpts to the 'NxNxN' string required by writers."""
    if isinstance(k, str):
        k = k.replace(" ", "x")
        if "x" not in k:
            # single number -> cube mesh
            return f"{k}x{k}x{k}"
        return k
    if isinstance(k, (list, tuple)) and len(k) == 3:
        return f"{k[0]}x{k[1]}x{k[2]}"
    raise ValueError(f"Invalid kpts format: {k!r}")


def _mev(delta_e_ev: float, natoms: int | None) -> float:
    """Convert absolute ΔE (eV/cell) to meV/atom if natoms provided, else meV/cell."""
    if natoms and natoms > 0:
        return 1000.0 * delta_e_ev / natoms
    return 1000.0 * delta_e_ev