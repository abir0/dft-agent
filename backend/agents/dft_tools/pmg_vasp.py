from typing import Optional, Dict, Any, Tuple, List, Union
from langchain_core.tools import tool
from pathlib import Path
from typing import Optional, Dict, Any
from pymatgen.io.vasp.sets import MPStaticSet, MPRelaxSet
from pymatgen.io.vasp.outputs import Outcar
from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.core import Structure

def _load_structure(struct_path: str) -> Structure:
    sp = Path(struct_path)
    if not sp.exists():
        raise FileNotFoundError(f"Structure file not found: {struct_path}")
    return Structure.from_file(str(sp))

def _norm_kpts(k: Union[str, List[int], Tuple[int,int,int]]) -> str:
    if isinstance(k, str):
        k = k.strip().replace(" ", "x")
        if "x" not in k:
            return f"{k}x{k}x{k}"
        return k
    if isinstance(k, (list, tuple)) and len(k) == 3:
        a, b, c = k
        return f"{int(a)}x{int(b)}x{int(c)}"
    raise ValueError(f"Invalid kpts format: {k!r}")

def _write_kpoints_file(workdir: Path, kpts: str) -> None:
    a, b, c = (int(x) for x in kpts.split("x"))
    kp = Kpoints.monkhorst_automatic(kpts=(a, b, c))
    (workdir / "KPOINTS").write_text(kp.__str__())

@tool
def write_vasp_scf(
    struct: str,
    workdir: str,
    kpts: Optional[Union[str, List[int], Tuple[int,int,int]]] = None,
    encut: Optional[int] = None,
    ismear: int = 1,
    sigma: float = 0.2,
    reciprocal_density: float = 60.0,
    potcar_functional: str = "PBE",
) -> str:
    """Write a VASP single-point (SCF) input set at `workdir`.
    Args:
      struct: path to POSCAR/CONTCAR/CIF.
      kpts: "NxNxN" or [N,N,N]. If omitted, uses reciprocal_density.
      encut, ismear, sigma: common INCAR controls.
      reciprocal_density: fallback k-point density when kpts not given.
    Returns: workdir
    """
    s = _load_structure(struct)
    w = Path(workdir); w.mkdir(parents=True, exist_ok=True)

    user = {"ISMEAR": ismear, "SIGMA": sigma}
    if encut is not None: user["ENCUT"] = int(encut)
    try:
        vset = MPStaticSet(
            s,
            user_incar_settings=user,
            reciprocal_density=reciprocal_density,
            potcar_functional=potcar_functional,
        )
    except TypeError:
        vset = MPStaticSet(
            s,
            user_incar_settings=user,
            reciprocal_density=reciprocal_density,
        )

    vset.write_input(str(w))

    if kpts is not None:
        _write_kpoints_file(w, _norm_kpts(kpts))  # overrides MPStaticSet auto KPOINTS

    return str(w)

@tool
def write_vasp_relax(
    struct: str,
    workdir: str,
    kpts: Optional[Union[str, List[int], Tuple[int,int,int]]] = None,
    encut: Optional[int] = None,
    isif: int = 2,
    ibrion: int = 2,
    nsw: int = 80,
    ediff: float = 1e-5,
    ismear: int = 1,
    sigma: float = 0.2,
    reciprocal_density: float = 60.0,
    potcar_functional: str = "PBE",
) -> str:
    """Write a VASP ionic relaxation input set at `workdir`.
    Args:
      struct: path to POSCAR/CONTCAR/CIF.
      kpts: "NxNxN" or [N,N,N]. If omitted, uses reciprocal_density.
      encut, isif, ibrion, nsw, ediff, ismear, sigma: INCAR controls.
    Returns: workdir
    """
    s = _load_structure(struct)
    w = Path(workdir); w.mkdir(parents=True, exist_ok=True)

    user = {
        "ISIF": isif,
        "IBRION": ibrion,
        "NSW": nsw,
        "EDIFF": ediff,
        "ISMEAR": ismear,
        "SIGMA": sigma,
    }
    if encut is not None: user["ENCUT"] = int(encut)
    try:
        vset = MPRelaxSet(
            s,
            user_incar_settings=user,
            potcar_functional=potcar_functional,
        )
    except TypeError:
        vset = MPRelaxSet(
            s,
            user_incar_settings=user,
        )
    vset.write_input(str(w))

    if kpts is not None:
        _write_kpoints_file(w, _norm_kpts(kpts))

    return str(w)
@tool
def parse_vasp_energy(workdir: str) -> float:
    """
    Returns final free energy TOTEN (eV) from OUTCAR.
    """
    outcar = Outcar(str(Path(workdir)/"OUTCAR"))

    if hasattr(outcar, "final_energy") and outcar.final_energy is not None:
        return float(outcar.final_energy)

    txt = Path(workdir, "OUTCAR").read_text(errors="ignore")
    for line in reversed(txt.splitlines()):
        if "free  energy   TOTEN" in line:
            return float(line.split("=")[1].split()[0])
    raise ValueError("VASP final energy not found in OUTCAR")
