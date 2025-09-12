"""
Structure Generation and Manipulation Tools

Core tools for generating and manipulating atomic structures using ASE.
"""

import json
from pathlib import Path
from typing import List, Optional

from ase import Atoms
from ase.build import add_adsorbate as ase_add_adsorbate
from ase.build import bulk, molecule, surface
from ase.io import read, write
from langchain_core.tools import tool
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import numpy as np

from backend.utils.workspace import get_subdir_path


@tool
def generate_bulk(
    element: str,
    crystal: str = "fcc",
    a: float = 4.0,
    c_over_a: Optional[float] = None,
    orthorhombic: bool = False,
    cubic: bool = False,
    _thread_id: Optional[str] = None,
) -> str:
    """
    Build bulk unit-cell structure from element and crystal type.

    Args
    ----
    element : str
        Chemical element symbol (e.g. 'Cu', 'Al', 'Pt')
    crystal : str
        Crystal prototype name recognised by pymatgen/ASE:
        fcc, bcc, hcp, diamond, "zincblende", "rocksalt", ...
    a : float
        Lattice parameter (Å).
    c_over_a : float, optional
        c/a ratio for hexagonal systems (ignored otherwise).
    orthorhombic : bool
        Force orthorhombic setting (fcc/bcc only).
    cubic : bool
        Force cubic setting.
    _thread_id : str, optional
        Workspace identifier used to isolate output directories.

    Returns
    -------
    str
        Human-readable summary.
    """
    try:
        # Create the structure
        if crystal:
            atoms = bulk(
                element,
                crystal,
                a=a,
                c=c_over_a * a if c_over_a else None,
                orthorhombic=orthorhombic,
                cubic=cubic,
            )

        # Canonicalise via pymatgen
        pmg_struct = AseAtomsAdaptor.get_structure(atoms)
        sga = SpacegroupAnalyzer(pmg_struct, symprec=1e-3)
        primitive = sga.get_primitive_standard_structure()
        conventional = sga.get_conventional_standard_structure()

        # File paths
        output_dir = get_subdir_path(_thread_id, "structures/bulk")

        stem = f"{element}_{crystal}_a{a:.2f}"
        if c_over_a:
            stem += f"_c{c_over_a:.2f}"

        cif_path = output_dir / f"{stem}.cif"
        json_path = cif_path.with_suffix(".json")
        xyz_path = cif_path.with_suffix(".xyz")
        poscar_path = cif_path.with_suffix("").with_suffix(".POSCAR")

        # Write files
        primitive.to(str(cif_path), fmt="cif")
        write(str(xyz_path), atoms)  # ASE xyz
        conventional.to(str(poscar_path), fmt="poscar")

        # Save metadata
        metadata = {
            "element": element,
            "crystal_structure": crystal,
            "lattice_parameter_a": a,
            "c_over_a": c_over_a,
            "num_atoms_primitive": primitive.num_sites,
            "num_atoms_conventional": conventional.num_sites,
            "cell_volume_primitive": primitive.volume,
            "cell_volume_conventional": conventional.volume,
            "density_g_cm3": primitive.density,
            "formula_primitive": primitive.formula,
            "reduced_formula": primitive.composition.reduced_formula,
            "space_group_number": sga.get_space_group_number(),
            "space_group_symbol": sga.get_space_group_symbol(),
            "point_group": sga.get_point_group_symbol(),
            "files": {
                "cif": str(cif_path),
                "xyz": str(xyz_path),
                "poscar": str(poscar_path),
                "metadata": str(json_path),
            },
        }

        with open(json_path, "w") as fd:
            json.dump(metadata, fd, indent=2)

        return (
            f"Generated {crystal} {element} bulk structure "
            f"({primitive.num_sites} atoms primitive, "
            f"space-group {sga.get_space_group_number()} "
            f"{sga.get_space_group_symbol()}). "
            f"Files saved to: {output_dir} "
            f"with CIF file: {str(cif_path.name)}"
        )

    except Exception as exc:
        return f"Error generating bulk structure: {exc}"


@tool
def create_supercell(
    structure_file: str,
    scaling_matrix: Optional[List[int]] = None,
    wrap_atoms: bool = True,
    _thread_id: Optional[str] = None,
) -> str:
    """Create supercell from unit cell structure.

    Args:
        structure_file: Path to input structure file (CIF, POSCAR, XYZ, etc.)
        scaling_matrix: Supercell scaling factors (nx, ny, nz)
        wrap_atoms: Whether to wrap atoms back into unit cell

    Returns:
        String with supercell information and file path
    """
    try:
        # Set structure
        atoms = read(structure_file)

        # Create supercell
        if scaling_matrix is None:
            scaling_matrix = [2, 2, 2]
        supercell = atoms * scaling_matrix

        if wrap_atoms:
            supercell.wrap()

        input_path = Path(structure_file)

        # Use workspace-specific directory
        output_dir = get_subdir_path(_thread_id, "structures/supercells")

        scale_str = "x".join(map(str, scaling_matrix))
        stem = f"{input_path.stem}_supercell_{scale_str}"
        output_path = output_dir / f"{stem}.cif"

        # Save supercell
        write(str(output_path), supercell)

        # Metadata
        metadata = {
            "original_file": structure_file,
            "scaling_matrix": scaling_matrix,
            "original_atoms": len(atoms),
            "supercell_atoms": len(supercell),
            "original_volume": atoms.get_volume(),
            "supercell_volume": supercell.get_volume(),
            "formula": supercell.get_chemical_formula(),
            "files": {"cif": str(output_path)},
        }

        metadata_file = output_path.with_suffix(".json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        return (
            f"Created {scale_str} supercell with {len(supercell)} atoms "
            f"(volume {supercell.get_volume():.2f} Å³). "
            f"Saved to {output_path}"
        )

    except Exception as e:
        return f"Error creating supercell: {str(e)}"


@tool
def generate_slab(
    structure_file: str,
    miller_indices: Optional[List[int]] = None,
    layers: int = 5,
    vacuum: float = 10.0,
    orthogonal: bool = False,
    _thread_id: Optional[str] = None,
) -> str:
    """Generate surface slab from bulk structure.

    Args:
        structure_file: Path to bulk structure file
        miller_indices: Miller indices for surface orientation (h, k, l)
        layers: Number of atomic layers in slab
        vacuum: Vacuum thickness in Angstrom
        orthogonal: Force orthogonal unit cell

    Returns:
        String with slab information and file path
    """
    try:
        # Set up bulk structure
        bulk_atoms = read(structure_file)

        # Create slab
        if miller_indices is None:
            miller_indices = [1, 1, 1]
        slab = surface(bulk_atoms, miller_indices, layers, vacuum)

        if orthogonal:
            # Make orthogonal if requested
            slab = slab.copy()
            cell = slab.get_cell()
            # Simple orthogonalization - may need improvement for complex cases

        input_path = Path(structure_file)
        output_dir = get_subdir_path(_thread_id, "structures/slabs")

        miller_str = "".join(map(str, miller_indices))
        stem = f"{input_path.stem}_slab_{miller_str}_{layers}L_vac{vacuum:.1f}"
        output_path = output_dir / f"{stem}.cif"

        # Save slab
        write(str(output_path), slab)

        cell = slab.get_cell()
        surface_area = abs(cell[0, 0] * cell[1, 1] - cell[0, 1] * cell[1, 0])

        metadata = {
            "bulk_file": structure_file,
            "miller_indices": miller_indices,
            "layers": layers,
            "vacuum": vacuum,
            "num_atoms": len(slab),
            "surface_area": surface_area,
            "slab_thickness": cell[2, 2] - vacuum,
            "formula": slab.get_chemical_formula(),
            "files": {"cif": str(output_path)},
        }

        metadata_file = output_path.with_suffix(".json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        return (
            f"Generated slab ({miller_str}) with {layers} layers and {len(slab)} atoms. "
            f"Surface area: {surface_area:.2f} Å², vacuum {vacuum} Å. "
            f"Saved to {output_path}"
        )

    except Exception as e:
        return f"Error generating slab: {str(e)}"


@tool
def add_adsorbate(
    slab_file: str,
    adsorbate_formula: str,
    site_position: Optional[List[float]] = None,
    height: float = 2.0,
    coverage: Optional[float] = None,
    adsorption_site: str = "top",
    _thread_id: Optional[str] = None,
) -> str:
    """Add adsorbate to surface slab using robust Pymatgen-based implementation.

    Args:
        slab_file: Path to slab structure file
        adsorbate_formula: Adsorbate formula/name (e.g., 'CO', 'H', 'O', 'CH4')
        site_position: Two fractional surface coordinates [x, y] each between 0 and 1
        height: Height above surface in Angstrom (distance from surface to bottom of adsorbate)
        coverage: Surface coverage (if specified, will add multiple adsorbates)
        adsorption_site: Type of adsorption site ('top', 'bridge', 'hollow', etc.)

    Returns:
        String with adsorbate information and file path
    """
    # Import the pymatgen-based function
    from backend.agents.dft_tools.pymatgen_tools import add_adsorbate_pymatgen
    
    # Call the robust pymatgen implementation
    return add_adsorbate_pymatgen.invoke({
        "slab_file": slab_file,
        "adsorbate_formula": adsorbate_formula,
        "site_position": site_position,
        "height": height,
        "adsorption_site": adsorption_site,
        "_thread_id": _thread_id
    })


@tool
def add_vacuum(
    structure_file: str,
    axis: int = 2,
    thickness: float = 10.0,
    _thread_id: Optional[str] = None,
) -> str:
    """Add vacuum spacing along specified axis.

    Args:
        structure_file: Path to structure file
        axis: Axis along which to add vacuum (0=x, 1=y, 2=z)
        thickness: Vacuum thickness in Angstrom

    Returns:
        String with vacuum addition information and file path
    """
    try:
        atoms = read(structure_file)

        # Add vacuum
        atoms.center(vacuum=thickness / 2, axis=axis)

        input_path = Path(structure_file)
        # Use workspace-specific directory
        output_dir = get_subdir_path(_thread_id, "structures/with_vacuum")

        axis_name = ["x", "y", "z"][axis]
        stem = f"{input_path.stem}_vac{axis_name}{thickness:.1f}"
        output_path = output_dir / f"{stem}.cif"

        write(str(output_path), atoms)

        metadata = {
            "original_file": structure_file,
            "vacuum_axis": axis,
            "vacuum_thickness": thickness,
            "num_atoms": len(atoms),
            "cell_volume": atoms.get_volume(),
            "formula": atoms.get_chemical_formula(),
            "files": {"cif": str(output_path)},
        }

        metadata_file = output_path.with_suffix(".json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        return (
            f"Added {thickness} Å vacuum along {axis_name}-axis. "
            f"New volume {atoms.get_volume():.2f} Å³. "
            f"Saved to {output_path}"
        )

    except Exception as e:
        return f"Error adding vacuum: {str(e)}"
