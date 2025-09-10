from langchain_core.tools import tool
from typing import Literal, Optional, Dict, Any, Tuple, List
from pathlib import Path
from pymatgen.core import Structure, Lattice
from pymatgen.core.structure import IMolecule
from pymatgen.core.surface import SlabGenerator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.periodic_table import Element
from pymatgen.io.vasp import Poscar, Xdatcar
from pymatgen.io.cif import CifParser
from pymatgen.core import Molecule
from pymatgen.ext.matproj import MPRester
from ase.build import add_adsorbate
from ase import Atoms
import json, base64, io

# ---------- helpers
def _to_pm(atoms: Atoms) -> Structure:
    cell = atoms.get_cell()
    sp = atoms.get_chemical_symbols()
    frac = atoms.get_scaled_positions()
    return Structure(Lattice(cell), sp, frac)

def _to_ase(struct: Structure) -> Atoms:
    return Atoms(symbols=[s.specie.symbol for s in struct],
                 scaled_positions=struct.frac_coords,
                 cell=struct.lattice.matrix, pbc=True)

def _read_text_or_b64(text_or_b64: str) -> str:
    # allow users to send file content as plain text or base64
    try:
        return base64.b64decode(text_or_b64).decode()
    except Exception:
        return text_or_b64

# ---------- tools (LLM-callable)

@tool
def build_structure(
        source: Literal["file", "spacegroup", "materials_project", "lattice"],
        # Arguments for 'file' source
        file_content: Optional[str] = None,
        file_format: Optional[Literal["cif", "poscar"]] = None,
        # Arguments for 'spacegroup' source
        spacegroup: Optional[str] = None,
        species: Optional[List[str]] = None,
        lattice_constants: Optional[Dict[str, float]] = None,  # e.g. {"a": 3.2, "c": 5.1}
        frac_coords: Optional[List[List[float]]] = None,
        # Arguments for 'materials_project' source
        mp_id: Optional[str] = None,
        api_key: Optional[str] = None,
        # Arguments for 'lattice' source (replaces build_bulk)
        lattice_dict: Optional[Dict[str, Any]] = None,
        # e.g. {"type": "cubic", "params": [4.05, 4.05, 4.05, 90, 90, 90]}
) -> str:
    """
    Builds a crystal structure from one of four sources: a file, a spacegroup, the Materials Project, or direct lattice parameters.

    Args:
        source: The method to use for building the structure. Must be one of 'file', 'spacegroup', 'materials_project', or 'lattice'.

        --- For source='file' ---
        file_content: The content of the file (e.g., CIF or POSCAR) as a plain text string or base64 encoded.
        file_format: The format of the file content, either 'cif' or 'poscar'.

        --- For source='spacegroup' ---
        spacegroup: The spacegroup symbol or number (e.g., 'Fm-3m' or 225).
        species: A list of element symbols for each site.
        lattice_constants: A dictionary specifying lattice constants like {"a": 3.52, "b": 3.52, "c": 3.52, "alpha": 90, ...}.
        frac_coords: The fractional coordinates for each site.

        --- For source='materials_project' ---
        mp_id: The Materials Project ID of the structure (e.g., 'mp-30').
        api_key: Your personal Materials Project API key.

        --- For source='lattice' ---
        lattice_dict: A dictionary defining the lattice, e.g. {"type": "cubic", "params": [a,b,c,alpha,beta,gamma]}.
        species: The element symbol for the species on the basis sites.
        frac_coords: The fractional coordinates of the basis sites.
    """
    s = None
    if source == "file":
        txt = _read_text_or_b64(file_content)
        if file_format == "cif":
            s = CifParser.from_string(txt).get_structures(primitive=False)[0]
        else:
            s = Structure.from_str(txt, fmt="poscar")
    elif source == "spacegroup":
        a = lattice_constants.get("a")
        b = lattice_constants.get("b")
        c = lattice_constants.get("c")
        alpha = lattice_constants.get("alpha")
        beta = lattice_constants.get("beta")
        gamma = lattice_constants.get("gamma")
        lat = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
        s = Structure.from_spacegroup(spacegroup, lat, species, frac_coords)
    elif source == "materials_project":
        with MPRester(api_key) as mpr:
            s = mpr.get_structure_by_material_id(mp_id)
    elif source == "lattice":
        a, b, c, al, be, ga = lattice_dict["params"]
        lat = Lattice.from_parameters(a, b, c, al, be, ga)
        s = Structure(lat, species * len(frac_coords), frac_coords)

    if not s:
        raise ValueError(f"Could not build structure with source='{source}' and provided arguments.")

    sga = SpacegroupAnalyzer(s)
    s = sga.get_conventional_standard_structure()
    return json.dumps(s.as_dict())



@tool
def make_supercell(struct_json: str, mult: Tuple[int,int,int]) -> str:
    """Return supercell structure JSON."""
    s = Structure.from_dict(json.loads(struct_json))
    s.make_supercell(mult)
    return json.dumps(s.as_dict())

@tool
def make_slab(struct_json: str, miller: Tuple[int,int,int], min_slab: float=12.0, min_vac: float=15.0, center: bool=True) -> str:
    """Generate a surface slab via pymatgen's SlabGenerator (from a bulk structure)."""
    bulk = Structure.from_dict(json.loads(struct_json))
    sg = SlabGenerator(bulk, miller, min_slab, min_vac, center_slab=center, in_unit_planes=True)
    slab = sg.get_slabs(bonds=None, max_broken_bonds=0)[0]
    return json.dumps(slab.as_dict())

@tool
def place_adsorbate_on_slab(slab_json: str, ads: Literal["CO","O","H","N2"], site: Literal["ontop","bridge","fcc","hcp"], height: float=1.8, orientation: Literal["C-down","O-down","flat"]="C-down") -> str:
    """
    Crude adsorbate placement: convert slab to ASE, add adsorbate by named site guess.
    Returned as Structure JSON for downstream writers.
    """
    slab = Structure.from_dict(json.loads(slab_json))
    ase_slab = _to_ase(slab)
    # build small molecules
    mol = Molecule.from_formula(ads) if ads not in ("CO",) else Molecule(["C","O"], [[0,0,0],[1.15,0,0]])
    ase_mol = Atoms(symbols=[sp.symbol for sp in mol.species], positions=mol.cart_coords)
    # orientation (very simple)
    if orientation == "O-down" and ads=="CO":
        ase_mol = Atoms(symbols=["O","C"], positions=[[0,0,0],[1.15,0,0]])
    # site mapping: ASE add_adsorbate accepts 'fcc','hcp','bridge' or (x,y)
    add_adsorbate(ase_slab, ase_mol, height, position=site if site in ("fcc","hcp","bridge") else site)
    ase_slab.center(vacuum=ase_slab.cell.lengths()[2])  # keep vacuum
    return json.dumps(_to_pm(ase_slab).as_dict())

@tool
def set_initial_magnetism(struct_json: str, moments: Dict[str,float]) -> str:
    """
    Apply initial magnetic moments by species (e.g. {"Fe":2.2, "O":0.0}) as site properties 'magmom'.
    Writers may translate this to INCAR (MAGMOM) or QE flags.
    """
    s = Structure.from_dict(json.loads(struct_json))
    mags = [moments.get(sp.symbol, 0.0) for sp in s.species]
    s.add_site_property("magmom", mags)
    return json.dumps(s.as_dict())

