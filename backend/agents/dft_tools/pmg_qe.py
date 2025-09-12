from pathlib import Path
from typing import Dict, Any, Tuple,Optional
from pymatgen.io.pwscf import PWInput, PWOutput

def _kpts_tuple(kpts: str) -> Tuple[int,int,int]:
    a,b,c = [int(x) for x in kpts.lower().split("x")]
    return a,b,c
@tool
def write_qe_scf(struct, workdir: str,
                 ecutwfc: float = 40.0,
                 ecutrho: Optional[int] = None,
                 kpts: str = "12x12x12",
                 input_overrides: Dict[str,Any] = None):
    """
    Writes a minimal scf.in for QE using pymatgen's PWInput.
    You still need to ensure pseudopotential (.UPF) names are resolvable on your machine.
    """
    if ecutrho is None:
        ecutrho = ecutwfc * 4
    Path(workdir).mkdir(parents=True, exist_ok=True)
    kx,ky,kz = _kpts_tuple(kpts)

    # Minimal input dicts
    control = dict(calculation="scf", prefix="calc", outdir="./tmp")
    system  = dict(ecutwfc=ecutwfc, occupations="smearing")
    electrons = dict(conv_thr=1e-8)

    if input_overrides:
        # shallow override
        control.update(input_overrides.get("control", {}))
        system.update(input_overrides.get("system", {}))
        electrons.update(input_overrides.get("electrons", {}))

    pw = PWInput(
        structure=struct,
        pseudo=None,          # supply mapping if you want PWInput to emit PSEUDO_DIR block
        control=control,
        system=system,
        electrons=electrons,
        kpoints_grid=(kx,ky,kz),
        kpoints_shift=(0,0,0),
    )
    (Path(workdir)/"scf.in").write_text(str(pw))

def parse_qe_energy(workdir: str) -> float:
    """
    Parse total energy (eV) from scf.out. Simple grep (pymatgen's QE output
    parser may not always be installed; this is robust enough for a smoke test).
    """
    txt = Path(workdir, "scf.out").read_text(errors="ignore")
    for line in txt.splitlines():
        if "total energy" in line and "Ry" in line:
            ry = float(line.split("=")[1].split("Ry")[0])
            return ry * 13.605693009
    raise ValueError("QE total energy not found in scf.out")
