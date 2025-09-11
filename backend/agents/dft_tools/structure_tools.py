"""
Structure Generation and Manipulation Tools

Core tools for generating and manipulating atomic structures using ASE.
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple

from ase import Atoms
from ase.build import add_adsorbate as ase_add_adsorbate
from ase.build import bulk, molecule, surface
from ase.io import read, write
from langchain_core.tools import tool

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
    """Build bulk unit cell structure from element and crystal type.

    Args:
        element: Chemical element symbol (e.g., 'Cu', 'Al', 'Pt')
        crystal: Crystal structure type ('fcc', 'bcc', 'hcp', 'diamond', 'zincblende', 'rocksalt', 'cesiumchloride', 'fluorite', 'wurtzite')
        a: Lattice parameter in Angstrom
        c_over_a: c/a ratio for hexagonal structures (default: ideal ratio)
        orthorhombic: Use orthorhombic unit cell for fcc and bcc
        cubic: Use cubic unit cell

    Returns:
        String with structure information and file path
    """
    try:
        # Create bulk structure
        atoms = bulk(
            element,
            crystal,
            a=a,
            c=c_over_a * a if c_over_a else None,
            orthorhombic=orthorhombic,
            cubic=cubic,
        )

        # Get workspace-specific output directory
        if _thread_id:
            output_dir = get_subdir_path(_thread_id, "structures/bulk")
        else:
            # Fallback to outputs directory if no thread_id
            output_dir = Path("data/outputs/structures/bulk")
            output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        filename = f"{element}_{crystal}_a{a:.2f}"
        if c_over_a:
            filename += f"_c{c_over_a:.2f}"
        filename += ".cif"

        filepath = output_dir / filename

        # Save structure
        write(str(filepath), atoms)

        # Create metadata
        metadata = {
            "element": element,
            "crystal_structure": crystal,
            "lattice_parameter_a": a,
            "c_over_a": c_over_a,
            "num_atoms": len(atoms),
            "cell_volume": atoms.get_volume(),
            "formula": atoms.get_chemical_formula(),
            "filepath": str(filepath),
        }

        metadata_file = filepath.with_suffix(".json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        return (
            f"Generated {crystal} {element} bulk structure with {len(atoms)} atoms. "
            f"Lattice parameter a={a:.3f} Å. Saved as {filepath}"
        )

    except Exception as e:
        return f"Error generating bulk structure: {str(e)}"


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
        # Set default scaling matrix if not provided
        if scaling_matrix is None:
            scaling_matrix = [2, 2, 2]

        # Read structure
        atoms = read(structure_file)

        # Create supercell
        supercell = atoms * scaling_matrix

        if wrap_atoms:
            supercell.wrap()

        # Generate output filename
        input_path = Path(structure_file)

        # Use workspace-specific directory if thread_id is available
        if _thread_id:
            output_dir = get_subdir_path(_thread_id, "structures/supercells")
        else:
            # Fallback to outputs directory
            output_dir = Path("data/outputs/structures/supercells")
            output_dir.mkdir(parents=True, exist_ok=True)

        scale_str = "x".join(map(str, scaling_matrix))
        output_filename = f"{input_path.stem}_supercell_{scale_str}{input_path.suffix}"
        output_path = output_dir / output_filename

        # Save supercell
        write(str(output_path), supercell)

        # Create metadata
        metadata = {
            "original_file": structure_file,
            "scaling_matrix": scaling_matrix,
            "original_atoms": len(atoms),
            "supercell_atoms": len(supercell),
            "original_volume": atoms.get_volume(),
            "supercell_volume": supercell.get_volume(),
            "supercell_formula": supercell.get_chemical_formula(),
            "filepath": str(output_path),
        }

        metadata_file = output_path.with_suffix(".json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        return (
            f"Created {scale_str} supercell with {len(supercell)} atoms "
            f"(from {len(atoms)} atoms). Volume: {supercell.get_volume():.2f} Å³. "
            f"Saved as {output_path}"
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
        # Set default miller indices if not provided
        if miller_indices is None:
            miller_indices = [1, 1, 1]

        # Read bulk structure
        bulk_atoms = read(structure_file)

        # Create slab
        slab = surface(bulk_atoms, miller_indices, layers, vacuum)

        if orthogonal:
            # Make orthogonal if requested
            slab = slab.copy()
            cell = slab.get_cell()
            # Simple orthogonalization - may need improvement for complex cases

        # Generate output filename
        input_path = Path(structure_file)

        # Use workspace-specific directory if thread_id is available
        if _thread_id:
            output_dir = get_subdir_path(_thread_id, "structures/slabs")
        else:
            # Fallback to outputs directory
            output_dir = Path("data/outputs/structures/slabs")
            output_dir.mkdir(parents=True, exist_ok=True)

        miller_str = "".join(map(str, miller_indices))
        output_filename = f"{input_path.stem}_slab_{miller_str}_{layers}L_vac{vacuum:.1f}{input_path.suffix}"
        output_path = output_dir / output_filename

        # Save slab
        write(str(output_path), slab)

        # Calculate surface area
        cell = slab.get_cell()
        surface_area = abs(cell[0, 0] * cell[1, 1] - cell[0, 1] * cell[1, 0])

        # Create metadata
        metadata = {
            "bulk_file": structure_file,
            "miller_indices": miller_indices,
            "layers": layers,
            "vacuum": vacuum,
            "num_atoms": len(slab),
            "surface_area": surface_area,
            "slab_thickness": cell[2, 2] - vacuum,
            "formula": slab.get_chemical_formula(),
            "filepath": str(output_path),
        }

        metadata_file = output_path.with_suffix(".json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        return (
            f"Generated ({miller_str}) slab with {layers} layers and {len(slab)} atoms. "
            f"Surface area: {surface_area:.2f} Å². Vacuum: {vacuum} Å. "
            f"Saved as {output_path}"
        )

    except Exception as e:
        return f"Error generating slab: {str(e)}"


@tool
def add_adsorbate(
    slab_file: str,
    adsorbate_formula: str,
    site_position_x: float = 0.5,
    site_position_y: float = 0.5,
    height: float = 2.0,
    coverage: Optional[float] = None,
    _thread_id: Optional[str] = None,
) -> str:
    """Add adsorbate to surface slab.

    Args:
        slab_file: Path to slab structure file
        adsorbate_formula: Adsorbate formula/name (e.g., 'CO', 'H', 'O', 'CH4')
        site_position_x: X fractional coordinate on surface (0.0-1.0)
        site_position_y: Y fractional coordinate on surface (0.0-1.0)
        height: Height above surface in Angstrom
        coverage: Surface coverage (if specified, will add multiple adsorbates)

    Returns:
        String with adsorbate information and file path
    """
    try:
        # Read slab
        slab = read(slab_file)

        # Create adsorbate molecule
        if adsorbate_formula in ["H", "O", "N", "C", "S"]:
            # Single atom adsorbates
            adsorbate = Atoms(adsorbate_formula)
        elif adsorbate_formula == "CO":
            adsorbate = molecule("CO")
        elif adsorbate_formula == "H2":
            adsorbate = molecule("H2")
        elif adsorbate_formula == "O2":
            adsorbate = molecule("O2")
        elif adsorbate_formula == "N2":
            adsorbate = molecule("N2")
        elif adsorbate_formula == "H2O":
            adsorbate = molecule("H2O")
        elif adsorbate_formula == "CH4":
            adsorbate = molecule("CH4")
        else:
            # Try to create as molecule or atom
            try:
                adsorbate = molecule(adsorbate_formula)
            except Exception:
                adsorbate = Atoms(adsorbate_formula)

        # Add adsorbate to slab
        site_position = (site_position_x, site_position_y)
        ase_add_adsorbate(slab, adsorbate, height, position=site_position)

        # Generate output filename
        input_path = Path(slab_file)

        # Use workspace-specific directory if thread_id is available
        if _thread_id:
            output_dir = get_subdir_path(_thread_id, "structures/with_adsorbates")
        else:
            # Fallback to outputs directory
            output_dir = Path("data/outputs/structures/with_adsorbates")
            output_dir.mkdir(parents=True, exist_ok=True)

        pos_str = f"x{site_position[0]:.2f}y{site_position[1]:.2f}"
        output_filename = f"{input_path.stem}_{adsorbate_formula}_{pos_str}_h{height:.1f}{input_path.suffix}"
        output_path = output_dir / output_filename

        # Save structure with adsorbate
        write(str(output_path), slab)

        # Create metadata
        metadata = {
            "slab_file": slab_file,
            "adsorbate": adsorbate_formula,
            "site_position": site_position,
            "height": height,
            "coverage": coverage,
            "total_atoms": len(slab),
            "adsorbate_atoms": len(adsorbate),
            "formula": slab.get_chemical_formula(),
            "filepath": str(output_path),
        }

        metadata_file = output_path.with_suffix(".json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        return (
            f"Added {adsorbate_formula} adsorbate at position {site_position} "
            f"with height {height} Å. Total atoms: {len(slab)}. "
            f"Saved as {output_path}"
        )

    except Exception as e:
        return f"Error adding adsorbate: {str(e)}"


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
        # Read structure
        atoms = read(structure_file)

        # Add vacuum
        atoms.center(vacuum=thickness / 2, axis=axis)

        # Generate output filename
        input_path = Path(structure_file)
        output_dir = input_path.parent

        axis_name = ["x", "y", "z"][axis]
        output_filename = (
            f"{input_path.stem}_vac{axis_name}{thickness:.1f}{input_path.suffix}"
        )
        output_path = output_dir / output_filename

        # Save structure
        write(str(output_path), atoms)

        # Create metadata
        metadata = {
            "original_file": structure_file,
            "vacuum_axis": axis,
            "vacuum_thickness": thickness,
            "num_atoms": len(atoms),
            "cell_volume": atoms.get_volume(),
            "formula": atoms.get_chemical_formula(),
            "filepath": str(output_path),
        }

        metadata_file = output_path.with_suffix(".json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        return (
            f"Added {thickness} Å vacuum along {axis_name}-axis. "
            f"New cell volume: {atoms.get_volume():.2f} Å³. "
            f"Saved as {output_path}"
        )

    except Exception as e:
        return f"Error adding vacuum: {str(e)}"
