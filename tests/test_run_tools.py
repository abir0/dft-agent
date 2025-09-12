import tempfile
from pathlib import Path
from pymatgen.core import Structure, Lattice
from backend.agents.dft_tools.pmg_vasp import write_vasp_relax, write_vasp_scf, parse_vasp_energy
def _dummy_poscar(p: Path):
    s = Structure(Lattice.cubic(5.43), ["Si","Si"], [[0,0,0],[0.25,0.25,0.25]])
    s.to(fmt="poscar", filename=str(p))

def test_write_and_parse():
    tmp = Path(tempfile.mkdtemp())
    print("\n[Test tmp_path is:]",tmp)
    poscar = tmp / "POSCAR_Si"
    _dummy_poscar(poscar)

    scf_dir = tmp / "scf"
    write_vasp_scf.invoke({"struct": str(poscar), "workdir": str(scf_dir), "kpts": "4x4x4"})
    assert (scf_dir / "INCAR").exists()
    assert (scf_dir / "POSCAR").exists()
    assert (scf_dir / "KPOINTS").exists()
    assert (scf_dir / "POTCAR").exists()

    relax_dir = tmp / "relax"
    write_vasp_relax.invoke({"struct": str(poscar), "workdir": str(relax_dir), "kpts": [3, 3, 3]})
    assert (relax_dir / "INCAR").exists()
    assert (relax_dir / "POSCAR").exists()
    assert (relax_dir / "KPOINTS").exists()
    assert (relax_dir / "POTCAR").exists()
