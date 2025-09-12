import json
import base64
import pytest
import math
from pathlib import Path
from backend.agents.dft_tools.struct_tools import (
    build_structure, make_supercell, make_slab,
    place_adsorbate_on_slab, set_initial_magnetism
)
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def _as_struct(json_str):
    from pymatgen.core import Structure
    return Structure.from_dict(json.loads(json_str))

def _b64(s: str) -> str:
    return base64.b64encode(s.encode()).decode()

def _poscar_text():
    # Minimal cubic Si (2 sites) POSCAR text
    return """Si
1.0
5.430000 0.000000 0.000000
0.000000 5.430000 0.000000
0.000000 0.000000 5.430000
Si
2
Direct
0.000000 0.000000 0.000000
0.250000 0.250000 0.250000
"""

def _cif_text():
    return """# CIF file for diamond Si
data_Si
_symmetry_space_group_name_H-M   'F d -3 m'
_cell_length_a   5.430
_cell_length_b   5.430
_cell_length_c   5.430
_cell_angle_alpha   90
_cell_angle_beta    90
_cell_angle_gamma   90
_symmetry_Int_Tables_number 227
_chemical_formula_structural Si
_chemical_formula_sum 'Si2'

loop_
  _symmetry_equiv_pos_as_xyz
   'x, y, z'
   '-x, -y, -z'
   'y, z, x'
   '-y, -z, -x'
   'z, x, y'
   '-z, -x, -y'

loop_
  _atom_site_label
  _atom_site_type_symbol
  _atom_site_fract_x
  _atom_site_fract_y
  _atom_site_fract_z
   Si1  Si  0.00000  0.00000  0.00000
   Si2  Si  0.25000  0.25000  0.25000
"""
def _derive_cubic_a0(s):
    a,b,c = s.lattice.a, s.lattice.b, s.lattice.c
    # Candidates for a₀ if cell is cubic, or hex setting of fcc(111)
    candidates = [a, b, c, a/math.sqrt(2), b/math.sqrt(2), c/math.sqrt(3)]
    # Pick the candidate closest to 5.43
    return min(candidates, key=lambda x: abs(x - 5.43))
# (format, content_fn, encoder_fn)
CASES = [
    ("poscar", _poscar_text, lambda s: s),
    ("poscar", _poscar_text, _b64),
    ("cif",    _cif_text,    lambda s: s),
    ("cif",    _cif_text,    _b64),
]

@pytest.mark.parametrize(
    "fmt,maker,encoder",
    [
        ("poscar", _poscar_text, lambda s: s),
        ("poscar", _poscar_text, _b64),
        ("cif",    _cif_text,    lambda s: s),
        ("cif",    _cif_text,    _b64),
    ],
    ids=["poscar-text","poscar-b64","cif-text","cif-b64"],
)
def test_build_structure_from_file_variants(fmt, maker, encoder):
    content = encoder(maker())
    out = build_structure.invoke({"source":"file","file_content":content,"file_format":fmt})
    s = _as_struct(out)

    # Stable checks
    assert s.composition.reduced_formula == "Si"
    a0 = _derive_cubic_a0(s)
    assert a0 == pytest.approx(5.43, rel=1e-2, abs=5e-2)
    assert len(s) >= 1


def test_build_from_spacegroup():
    out = build_structure.invoke({
        "source": "spacegroup",
        "spacegroup": "Fm-3m",
        "species": ["Si", "Si"],
        "lattice_constants": {"a": 5.43, "b": 5.43, "c": 5.43, "alpha": 90, "beta": 90, "gamma": 90},
        "frac_coords": [[0,0,0],[0.25,0.25,0.25]],
    })
    s = _as_struct(out)
    sym = SpacegroupAnalyzer(s).get_space_group_symbol()
    assert sym.replace(" ", "") == "Fm-3m".replace(" ", "")

def test_build_from_lattice():
    out = build_structure.invoke({
        "source": "lattice",
        "lattice_dict": {"type": "cubic", "params": [5.43, 5.43, 5.43, 90, 90, 90]},
        "species": ["Si"],
        "frac_coords": [[0,0,0]],
    })
    s = _as_struct(out)
    assert s.composition.reduced_formula == "Si"

def test_make_supercell_and_slab_and_adsorbate():
    # Start from simple Si POSCAR
    base_json = build_structure.invoke({
        "source": "file",
        "file_content": _poscar_text(),
        "file_format": "poscar",
    })
    base = _as_struct(base_json)
    n0 = len(base)
    sc_json = make_supercell.invoke({"struct_json": base_json, "mult": (2,2,2)})
    s_sc = _as_struct(sc_json)
    assert len(s_sc) == n0 * 8

    slab_json = make_slab.invoke({"struct_json": base_json, "miller": (1,1,1), "min_slab": 10.0, "min_vac": 15.0})
    slab = _as_struct(slab_json)
    assert slab.lattice.c > 20  # vacuum added

    slab_ads = place_adsorbate_on_slab.invoke({
        "slab_json": slab_json,
        "ads": "CO",
        "site": "fcc",
        "height": 1.8,
        "orientation": "C-down"
    })
    s_ads = _as_struct(slab_ads)
    assert len(s_ads) > len(slab)  # adsorbate added


def test_set_initial_magnetism():
    base = build_structure.invoke({
        "source": "spacegroup",
        "spacegroup": "Fm-3m",
        "species": ["Fe"],
        "lattice_constants": {"a": 2.86, "b": 2.86, "c": 2.86, "alpha": 90, "beta": 90, "gamma": 90},
        "frac_coords": [[0,0,0]],
    })
    magd = set_initial_magnetism.invoke({
        "struct_json": base,
        "moments": {"Fe": 2.2}
    })
    s = _as_struct(magd)
    assert "magmom" in s.site_properties
    assert s.site_properties["magmom"][0] == pytest.approx(2.2)