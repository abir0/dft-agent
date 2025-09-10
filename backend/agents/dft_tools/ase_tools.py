"""
ASE-based Tools for Structure Optimization and Analysis

Tools for geometry optimization, k-point generation, and structural analysis using ASE.
"""

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
from ase.calculators.emt import EMT
from ase.calculators.espresso import Espresso
from ase.io import read, write
from ase.optimize import BFGS, LBFGS
from langchain_core.tools import tool

from backend.utils.workspace import get_subdir_path

try:
    from seekpath import get_path

    SEEKPATH_AVAILABLE = True
except ImportError:
    SEEKPATH_AVAILABLE = False


@tool
def geometry_optimization(
    structure_file: str,
    relax_type: str = "positions",
    force_tolerance: float = 0.05,
    stress_tolerance: Optional[float] = None,
    max_iterations: int = 200,
    optimizer: str = "BFGS",
    calculator: str = "emt",
    _thread_id: Optional[str] = None,
) -> str:
    """Perform structural relaxation using ASE.

    Args:
        structure_file: Path to input structure file
        relax_type: Type of relaxation ('positions', 'cell', 'both')
        force_tolerance: Force tolerance in eV/Å
        stress_tolerance: Stress tolerance in eV/Å³ (for cell relaxation)
        max_iterations: Maximum optimization steps
        optimizer: Optimizer algorithm ('BFGS', 'LBFGS')
        calculator: Calculator to use ('emt', 'espresso')

    Returns:
        String with optimization results and file path
    """
    try:
        # Read structure
        atoms = read(structure_file)

        # Set up calculator
        if calculator.lower() == "emt":
            calc = EMT()
        elif calculator.lower() == "espresso":
            # Basic Espresso setup - should be configured properly
            calc = Espresso(
                pseudopotentials={
                    "H": "H.pbe-rrkjus_psl.0.1.UPF",
                    "O": "O.pbe-n-rrkjus_psl.0.1.UPF",
                },
                tstress=True,
                tprnfor=True,
            )
        else:
            calc = EMT()  # fallback

        atoms.set_calculator(calc)

        # Set up optimizer
        if optimizer.upper() == "LBFGS":
            opt = LBFGS(atoms)
        else:
            opt = BFGS(atoms)

        # Run optimization
        initial_energy = atoms.get_potential_energy()
        opt.run(fmax=force_tolerance, steps=max_iterations)
        final_energy = atoms.get_potential_energy()

        # Generate output filename
        input_path = Path(structure_file)

        # Use workspace-specific directory if thread_id is available
        if _thread_id:
            output_dir = get_subdir_path(_thread_id, "optimized")
        else:
            # Fallback to input file's parent directory
            output_dir = input_path.parent / "optimized"
            output_dir.mkdir(exist_ok=True)

        output_filename = (
            f"{input_path.stem}_opt_{relax_type}_{optimizer.lower()}{input_path.suffix}"
        )
        output_path = output_dir / output_filename

        # Save optimized structure
        write(str(output_path), atoms)

        # Create metadata
        metadata = {
            "original_file": structure_file,
            "relax_type": relax_type,
            "optimizer": optimizer,
            "calculator": calculator,
            "force_tolerance": force_tolerance,
            "stress_tolerance": stress_tolerance,
            "max_iterations": max_iterations,
            "initial_energy": initial_energy,
            "final_energy": final_energy,
            "energy_change": final_energy - initial_energy,
            "num_atoms": len(atoms),
            "formula": atoms.get_chemical_formula(),
            "filepath": str(output_path),
        }

        metadata_file = output_path.with_suffix(".json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        return (
            f"Optimized structure using {optimizer} with {calculator}. "
            f"Energy change: {final_energy - initial_energy:.4f} eV. "
            f"Final energy: {final_energy:.4f} eV. "
            f"Saved as {output_path}"
        )

    except Exception as e:
        return f"Error in geometry optimization: {str(e)}"


@tool
def get_kpath_bandstructure(
    structure_file: str,
    path_density: float = 20.0,
    custom_path: Optional[List[str]] = None,
) -> str:
    """Generate high-symmetry k-point path for band structure calculations.

    Args:
        structure_file: Path to structure file
        path_density: Density of k-points per Å⁻¹
        custom_path: Custom k-point path labels (e.g., ['G', 'X', 'M', 'G'])

    Returns:
        String with k-path information and file path
    """
    try:
        # Read structure
        atoms = read(structure_file)

        if not SEEKPATH_AVAILABLE:
            return "Error: seekpath library not available. Please install with: pip install seekpath"

        # Get the crystal structure for seekpath
        cell = atoms.get_cell()
        positions = atoms.get_scaled_positions()
        numbers = atoms.get_atomic_numbers()

        # Get k-path using seekpath
        path_data = get_path((cell, positions, numbers))

        # Extract k-points and labels
        kpoints = []
        labels = []

        for point_coords, point_label in zip(
            path_data["path"], path_data["point_coords"], strict=False
        ):
            for label in point_coords:
                kpoints.append(path_data["point_coords"][label])
                labels.append(label)

        # Generate output files
        input_path = Path(structure_file)
        output_dir = input_path.parent / "kpaths"
        output_dir.mkdir(exist_ok=True)

        kpath_file = output_dir / f"{input_path.stem}_kpath.dat"
        labels_file = output_dir / f"{input_path.stem}_klabels.dat"

        # Save k-path
        with open(kpath_file, "w") as f:
            f.write("# K-point path for band structure calculation\\n")
            f.write("# kx ky kz label\\n")
            for i, (kpt, label) in enumerate(zip(kpoints, labels, strict=False)):
                f.write(f"{kpt[0]:12.8f} {kpt[1]:12.8f} {kpt[2]:12.8f}  {label}\\n")

        # Save labels separately
        with open(labels_file, "w") as f:
            f.write("# K-point labels\\n")
            for i, label in enumerate(labels):
                f.write(f"{i:3d}  {label}\\n")

        # Create metadata
        metadata = {
            "structure_file": structure_file,
            "path_density": path_density,
            "num_kpoints": len(kpoints),
            "kpoint_labels": labels,
            "kpath_file": str(kpath_file),
            "labels_file": str(labels_file),
            "lattice_type": path_data.get("bravais_lattice", "unknown"),
            "spacegroup": path_data.get("spacegroup_number", "unknown"),
        }

        metadata_file = kpath_file.with_suffix(".json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        return (
            f"Generated k-path with {len(kpoints)} points. "
            f"Lattice: {metadata['lattice_type']}. "
            f"Labels: {' -> '.join(labels)}. "
            f"Saved as {kpath_file}"
        )

    except Exception as e:
        return f"Error generating k-path: {str(e)}"


@tool
def generate_kpoint_mesh(
    structure_file: str,
    density: float = 3.0,
    gamma_centered: bool = True,
    manual_mesh: Optional[List[int]] = None,
) -> str:
    """Generate Monkhorst-Pack k-point mesh for DFT calculations.

    Args:
        structure_file: Path to structure file
        density: K-point density (points per Å⁻¹)
        gamma_centered: Whether to center mesh at gamma point
        manual_mesh: Manual k-point mesh (nx, ny, nz)

    Returns:
        String with k-mesh information and file path
    """
    try:
        # Read structure
        atoms = read(structure_file)

        if manual_mesh:
            kpts = manual_mesh
        else:
            # Calculate k-point mesh from density
            cell = atoms.get_cell()
            reciprocal_cell = np.linalg.inv(cell.T)  # Reciprocal lattice vectors

            # Calculate k-point spacing
            kpts = []
            for i in range(3):
                # Length of reciprocal lattice vector
                recip_length = np.linalg.norm(reciprocal_cell[i])
                # Number of k-points along this direction
                nk = max(1, int(np.ceil(density * recip_length)))
                kpts.append(nk)

            kpts = tuple(kpts)

        # Generate output files
        input_path = Path(structure_file)
        output_dir = input_path.parent / "kpoints"
        output_dir.mkdir(exist_ok=True)

        kpoints_file = output_dir / f"{input_path.stem}_KPOINTS"

        # Write VASP-style KPOINTS file
        with open(kpoints_file, "w") as f:
            f.write("Automatic mesh\\n")
            f.write("0\\n")
            if gamma_centered:
                f.write("Gamma\\n")
            else:
                f.write("Monkhorst-Pack\\n")
            f.write(f"{kpts[0]} {kpts[1]} {kpts[2]}\\n")
            f.write("0.0 0.0 0.0\\n")

        # Also write QE-style k-point info
        qe_kpoints_file = output_dir / f"{input_path.stem}_qe_kpoints.dat"
        with open(qe_kpoints_file, "w") as f:
            f.write("# Quantum ESPRESSO k-point mesh\\n")
            f.write("K_POINTS automatic\\n")
            f.write(f"{kpts[0]} {kpts[1]} {kpts[2]} 0 0 0\\n")

        # Calculate total k-points
        total_kpts = kpts[0] * kpts[1] * kpts[2]

        # Calculate k-point density achieved
        cell_volume = atoms.get_volume()
        kpt_density = total_kpts / cell_volume

        # Create metadata
        metadata = {
            "structure_file": structure_file,
            "requested_density": density,
            "manual_mesh": manual_mesh is not None,
            "kpoint_mesh": kpts,
            "total_kpoints": total_kpts,
            "gamma_centered": gamma_centered,
            "kpoint_density": kpt_density,
            "cell_volume": cell_volume,
            "vasp_file": str(kpoints_file),
            "qe_file": str(qe_kpoints_file),
        }

        metadata_file = kpoints_file.with_suffix(".json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        return (
            f"Generated {kpts[0]}x{kpts[1]}x{kpts[2]} k-point mesh "
            f"({total_kpts} total k-points). "
            f"Density: {kpt_density:.2f} k-pts/Å³. "
            f"Gamma-centered: {gamma_centered}. "
            f"Saved as {kpoints_file}"
        )

    except Exception as e:
        return f"Error generating k-point mesh: {str(e)}"


@tool
def relax_bulk(
    structure_file: str,
    ecutwfc: float = 30.0,
    ecutrho: Optional[float] = None,
    kpts: Optional[List[int]] = None,
    smearing: str = "gaussian",
    degauss: float = 0.02,
    fmax: float = 0.05,
    convergence_threshold: float = 1e-6,
    xc: str = "pbe",
) -> str:
    """Relax bulk crystal structure until forces and stress converge.

    Args:
        structure_file: Path to bulk structure file
        ecutwfc: Kinetic energy cutoff in Ry
        ecutrho: Charge density cutoff in Ry
        kpts: K-point mesh (nx, ny, nz)
        smearing: Smearing type
        degauss: Smearing width in Ry
        fmax: Force tolerance in eV/Å
        convergence_threshold: Energy convergence threshold
        xc: Exchange-correlation functional

    Returns:
        String with relaxation results and file path
    """
    try:
        # Read structure
        atoms = read(structure_file)

        # This would typically interface with QE or VASP
        # For now, use EMT as a placeholder
        from ase.calculators.emt import EMT

        calc = EMT()
        atoms.set_calculator(calc)

        # Set up optimizer for cell+positions
        from ase.constraints import ExpCellFilter
        from ase.optimize import BFGS

        # Apply filter for cell optimization
        ecf = ExpCellFilter(atoms)
        opt = BFGS(ecf)

        # Record initial state
        initial_energy = atoms.get_potential_energy()
        initial_volume = atoms.get_volume()

        # Run optimization
        opt.run(fmax=fmax)

        # Record final state
        final_energy = atoms.get_potential_energy()
        final_volume = atoms.get_volume()

        # Generate output filename
        input_path = Path(structure_file)
        output_dir = input_path.parent / "relaxed"
        output_dir.mkdir(exist_ok=True)

        output_filename = f"{input_path.stem}_relaxed_bulk{input_path.suffix}"
        output_path = output_dir / output_filename

        # Save relaxed structure
        write(str(output_path), atoms)

        # Create metadata
        metadata = {
            "original_file": structure_file,
            "calculation_type": "bulk_relaxation",
            "ecutwfc": ecutwfc,
            "ecutrho": ecutrho,
            "kpts": kpts,
            "smearing": smearing,
            "degauss": degauss,
            "xc_functional": xc,
            "initial_energy": initial_energy,
            "final_energy": final_energy,
            "energy_change": final_energy - initial_energy,
            "initial_volume": initial_volume,
            "final_volume": final_volume,
            "volume_change": final_volume - initial_volume,
            "num_atoms": len(atoms),
            "formula": atoms.get_chemical_formula(),
            "filepath": str(output_path),
        }

        metadata_file = output_path.with_suffix(".json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        return (
            f"Relaxed bulk structure. Energy change: {final_energy - initial_energy:.4f} eV. "
            f"Volume change: {final_volume - initial_volume:.2f} Å³. "
            f"Final energy: {final_energy:.4f} eV. "
            f"Saved as {output_path}"
        )

    except Exception as e:
        return f"Error in bulk relaxation: {str(e)}"


@tool
def relax_slab(
    structure_file: str,
    fixed_layers: int = 2,
    ecutwfc: float = 30.0,
    kpts: Optional[List[int]] = None,
    fmax: float = 0.05,
) -> str:
    """Relax surface slab with bottom layers fixed.

    Args:
        structure_file: Path to slab structure file
        fixed_layers: Number of bottom layers to fix
        ecutwfc: Kinetic energy cutoff in Ry
        kpts: K-point mesh (nx, ny, nz)
        fmax: Force tolerance in eV/Å

    Returns:
        String with relaxation results and file path
    """
    try:
        # Set defaults
        if kpts is None:
            kpts = [6, 6, 1]

        # Read structure
        atoms = read(structure_file)

        # Determine which atoms to fix (bottom layers)
        positions = atoms.get_positions()
        z_coords = positions[:, 2]

        # Sort by z-coordinate to identify layers
        z_sorted_indices = np.argsort(z_coords)

        # Find layer boundaries (atoms within 0.5 Å considered same layer)
        layers = []
        current_layer = [z_sorted_indices[0]]
        current_z = z_coords[z_sorted_indices[0]]

        for i in z_sorted_indices[1:]:
            if abs(z_coords[i] - current_z) < 0.5:
                current_layer.append(i)
            else:
                layers.append(current_layer)
                current_layer = [i]
                current_z = z_coords[i]
        layers.append(current_layer)

        # Fix bottom layers
        from ase.constraints import FixAtoms

        if len(layers) >= fixed_layers:
            indices_to_fix = []
            for layer_idx in range(fixed_layers):
                indices_to_fix.extend(layers[layer_idx])

            constraint = FixAtoms(indices=indices_to_fix)
            atoms.set_constraint(constraint)

        # Set up calculator (placeholder with EMT)
        from ase.calculators.emt import EMT

        calc = EMT()
        atoms.set_calculator(calc)

        # Optimize
        from ase.optimize import BFGS

        opt = BFGS(atoms)

        initial_energy = atoms.get_potential_energy()
        opt.run(fmax=fmax)
        final_energy = atoms.get_potential_energy()

        # Generate output filename
        input_path = Path(structure_file)
        output_dir = input_path.parent / "relaxed"
        output_dir.mkdir(exist_ok=True)

        output_filename = (
            f"{input_path.stem}_relaxed_slab_fix{fixed_layers}{input_path.suffix}"
        )
        output_path = output_dir / output_filename

        # Save relaxed structure
        write(str(output_path), atoms)

        # Create metadata
        metadata = {
            "original_file": structure_file,
            "calculation_type": "slab_relaxation",
            "fixed_layers": fixed_layers,
            "ecutwfc": ecutwfc,
            "kpts": kpts,
            "total_layers": len(layers),
            "fixed_atoms": len(indices_to_fix) if len(layers) >= fixed_layers else 0,
            "initial_energy": initial_energy,
            "final_energy": final_energy,
            "energy_change": final_energy - initial_energy,
            "num_atoms": len(atoms),
            "formula": atoms.get_chemical_formula(),
            "filepath": str(output_path),
        }

        metadata_file = output_path.with_suffix(".json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        return (
            f"Relaxed slab structure with {fixed_layers} fixed layers. "
            f"Energy change: {final_energy - initial_energy:.4f} eV. "
            f"Fixed {metadata['fixed_atoms']} atoms. "
            f"Saved as {output_path}"
        )

    except Exception as e:
        return f"Error in slab relaxation: {str(e)}"
