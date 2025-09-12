"""
DFT Calculator Tools for ASE

Tools for running DFT calculations using ASE with Quantum ESPRESSO calculator.
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from ase import Atoms
from ase.calculators.espresso import Espresso
from ase.io import read, write
from ase.optimize import BFGS, LBFGS
from langchain_core.tools import tool

from backend.utils.workspace import get_subdir_path


@tool
def run_dft_calculation(
    structure_file: str,
    calculation_type: str = "scf",
    functional: str = "pbe",
    ecutwfc: float = 40.0,
    ecutrho: Optional[float] = None,
    kpts: Optional[List[int]] = None,
    smearing: str = "gaussian",
    degauss: float = 0.02,
    convergence_threshold: float = 1e-6,
    max_iterations: int = 100,
    pseudopotential_dir: Optional[str] = None,
    qe_command: str = "pw.x",
    _thread_id: Optional[str] = None,
) -> str:
    """Run DFT calculation using ASE with Quantum ESPRESSO.

    Args:
        structure_file: Path to structure file
        calculation_type: Type of calculation ('scf', 'relax', 'vc-relax', 'bands')
        functional: Exchange-correlation functional ('pbe', 'pbesol', 'lda')
        ecutwfc: Kinetic energy cutoff in Ry
        ecutrho: Charge density cutoff in Ry (default: 4*ecutwfc)
        kpts: K-point mesh (nx, ny, nz)
        smearing: Smearing type ('gaussian', 'methfessel-paxton', 'fermi-dirac')
        degauss: Smearing width in Ry
        convergence_threshold: Energy convergence threshold
        max_iterations: Maximum SCF iterations
        pseudopotential_dir: Directory containing pseudopotential files
        qe_command: Quantum ESPRESSO command (e.g., 'pw.x', '/path/to/pw.x')

    Returns:
        String with calculation results and file paths
    """
    try:
        # Read structure
        atoms = read(structure_file)
        
        # Set defaults
        if ecutrho is None:
            ecutrho = 4.0 * ecutwfc
        
        if kpts is None:
            # Default k-point mesh based on cell size
            cell = atoms.get_cell()
            kpts = []
            for i in range(3):
                # Simple heuristic: ~1 k-point per 2 Å
                length = np.linalg.norm(cell[i])
                nk = max(1, int(np.ceil(length / 2.0)))
                kpts.append(nk)
            kpts = tuple(kpts)
        
        # Get unique elements
        elements = list(set(atoms.get_chemical_symbols()))
        
        # Set up pseudopotentials using the actual database
        pseudopotentials = {}
        if pseudopotential_dir is None:
            # Use the actual pslibrary database
            pp_db_path = Path("data/inputs/pseudopotentials/pp_mapping_pslibrary.json")
            if pp_db_path.exists():
                with open(pp_db_path, 'r') as f:
                    pp_database = json.load(f)
                
                for element in elements:
                    if element in pp_database:
                        element_pps = pp_database[element]
                        # Find matching pseudopotential
                        for pp in element_pps:
                            if (pp['functional'] == functional and 
                                pp['type'].upper() == 'PAW'):
                                pseudopotentials[element] = pp['filename']
                                break
            else:
                # Fallback to default naming
                for element in elements:
                    pseudopotentials[element] = f"{element}.{functional}-kjpaw_psl.1.0.0.UPF"
        else:
            # Use provided pseudopotential directory
            pp_dir = Path(pseudopotential_dir)
            for element in elements:
                # Look for pseudopotential files
                pp_files = list(pp_dir.glob(f"{element}.*"))
                if pp_files:
                    pseudopotentials[element] = pp_files[0].name
                else:
                    pseudopotentials[element] = f"{element}.{functional}-kjpaw_psl.1.0.0.UPF"
        
        # Set up Quantum ESPRESSO calculator
        input_data = {
            'control': {
                'calculation': calculation_type,
                'restart_mode': 'from_scratch',
                'prefix': 'pwscf',
                'outdir': './tmp',
                'pseudo_dir': str(pseudopotential_dir) if pseudopotential_dir else './pseudos',
                'verbosity': 'high',
                'tprnfor': True,
                'tstress': True,
            },
            'system': {
                'ecutwfc': ecutwfc,
                'ecutrho': ecutrho,
                'occupations': 'smearing',
                'smearing': smearing,
                'degauss': degauss,
            },
            'electrons': {
                'conv_thr': convergence_threshold,
                'mixing_beta': 0.7,
                'mixing_mode': 'plain',
                'diagonalization': 'david',
                'electron_maxstep': max_iterations,
            },
        }
        
        # Add relaxation parameters if needed
        if calculation_type in ['relax', 'vc-relax']:
            input_data['ions'] = {
                'ion_dynamics': 'bfgs',
                'ion_temperature': 'not_controlled',
            }
        
        if calculation_type == 'vc-relax':
            input_data['cell'] = {
                'cell_dynamics': 'bfgs',
                'cell_temperature': 'not_controlled',
            }
        
        # Create calculator - try new API first, fallback to old API
        try:
            # Try new ASE API (4.0+)
            from ase.calculators.espresso import EspressoProfile
            profile = EspressoProfile(argv=[qe_command])
            calc = Espresso(
                input_data=input_data,
                pseudopotentials=pseudopotentials,
                kpts=kpts,
                profile=profile,
            )
        except ImportError:
            # Fallback to old API (3.25.0)
            calc = Espresso(
                input_data=input_data,
                pseudopotentials=pseudopotentials,
                kpts=kpts,
                command=qe_command,
            )
        
        atoms.set_calculator(calc)
        
        # Run calculation on server
        print(f"Running {calculation_type} calculation on server...")
        print(f"Elements: {elements}")
        print(f"Pseudopotentials: {pseudopotentials}")
        print(f"K-points: {kpts}")
        print(f"Cutoff: {ecutwfc} Ry")
        print(f"QE command: {qe_command}")
        
        # Get initial energy (this triggers the calculation)
        initial_energy = atoms.get_potential_energy()
        
        # Get forces and stress if available
        forces = None
        stress = None
        try:
            forces = atoms.get_forces()
            stress = atoms.get_stress()
        except:
            pass
        
        # Generate output directory
        input_path = Path(structure_file)
        
        if _thread_id:
            output_dir = get_subdir_path(_thread_id, "calculations/dft_results")
        else:
            output_dir = Path("data/outputs/calculations/dft_results")
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results
        output_filename = f"{input_path.stem}_{calculation_type}_{functional}_ecut{ecutwfc:.0f}"
        output_path = output_dir / f"{output_filename}.cif"
        
        write(str(output_path), atoms)
        
        # Create metadata
        metadata = {
            "original_file": structure_file,
            "calculation_type": calculation_type,
            "functional": functional,
            "ecutwfc": ecutwfc,
            "ecutrho": ecutrho,
            "kpts": kpts,
            "smearing": smearing,
            "degauss": degauss,
            "convergence_threshold": convergence_threshold,
            "max_iterations": max_iterations,
            "pseudopotentials": pseudopotentials,
            "elements": elements,
            "total_energy": initial_energy,
            "num_atoms": len(atoms),
            "formula": atoms.get_chemical_formula(),
            "cell_volume": atoms.get_volume(),
            "forces_available": forces is not None,
            "stress_available": stress is not None,
            "output_file": str(output_path),
            "qe_command": qe_command,
        }
        
        if forces is not None:
            metadata["max_force"] = float(np.max(np.linalg.norm(forces, axis=1)))
            metadata["rms_force"] = float(np.sqrt(np.mean(np.sum(forces**2, axis=1))))
        
        if stress is not None:
            metadata["max_stress"] = float(np.max(np.abs(stress)))
        
        metadata_file = output_dir / f"{output_filename}.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Create summary
        summary = f"DFT {calculation_type.upper()} calculation completed successfully!\n"
        summary += f"Functional: {functional.upper()}\n"
        summary += f"Total energy: {initial_energy:.6f} eV\n"
        summary += f"Formula: {atoms.get_chemical_formula()}\n"
        summary += f"Atoms: {len(atoms)}\n"
        summary += f"Volume: {atoms.get_volume():.2f} Å³\n"
        summary += f"K-points: {kpts}\n"
        summary += f"Cutoff: {ecutwfc} Ry\n"
        
        if forces is not None:
            summary += f"Max force: {metadata['max_force']:.4f} eV/Å\n"
            summary += f"RMS force: {metadata['rms_force']:.4f} eV/Å\n"
        
        if stress is not None:
            summary += f"Max stress: {metadata['max_stress']:.4f} GPa\n"
        
        summary += f"\nResults saved to: {output_path}\n"
        summary += f"Metadata saved to: {metadata_file}\n"
        
        return summary
        
    except Exception as e:
        return f"Error in DFT calculation: {str(e)}"


@tool
def optimize_structure_dft(
    structure_file: str,
    functional: str = "pbe",
    ecutwfc: float = 40.0,
    kpts: Optional[List[int]] = None,
    fmax: float = 0.05,
    max_steps: int = 200,
    relax_cell: bool = False,
    pseudopotential_dir: Optional[str] = None,
    qe_command: str = "pw.x",
    _thread_id: Optional[str] = None,
) -> str:
    """Optimize structure using DFT with Quantum ESPRESSO.

    Args:
        structure_file: Path to structure file
        functional: Exchange-correlation functional ('pbe', 'pbesol', 'lda')
        ecutwfc: Kinetic energy cutoff in Ry
        kpts: K-point mesh (nx, ny, nz)
        fmax: Force tolerance in eV/Å
        max_steps: Maximum optimization steps
        relax_cell: Whether to relax cell parameters
        pseudopotential_dir: Directory containing pseudopotential files
        qe_command: Quantum ESPRESSO command

    Returns:
        String with optimization results and file paths
    """
    try:
        # Read structure
        atoms = read(structure_file)
        
        # Set defaults
        if kpts is None:
            cell = atoms.get_cell()
            kpts = []
            for i in range(3):
                length = np.linalg.norm(cell[i])
                nk = max(1, int(np.ceil(length / 2.0)))
                kpts.append(nk)
            kpts = tuple(kpts)
        
        # Get unique elements
        elements = list(set(atoms.get_chemical_symbols()))
        
        # Set up pseudopotentials
        pseudopotentials = {}
        pp_db_path = Path("data/inputs/pseudopotentials/pslibrary_database.json")
        if pp_db_path.exists():
            with open(pp_db_path, 'r') as f:
                pp_database = json.load(f)
            
            for element in elements:
                if element in pp_database['pseudopotentials']:
                    element_pps = pp_database['pseudopotentials'][element]['available_pseudopotentials']
                    for pp in element_pps:
                        if (pp['functional'] == functional and 
                            pp['type'].lower() == 'paw' and
                            pp['recommended']):
                            pseudopotentials[element] = pp['filename']
                            break
        
        # Set up calculator for optimization
        calculation_type = "vc-relax" if relax_cell else "relax"
        
        input_data = {
            'control': {
                'calculation': calculation_type,
                'restart_mode': 'from_scratch',
                'prefix': 'pwscf',
                'outdir': './tmp',
                'pseudo_dir': str(pseudopotential_dir) if pseudopotential_dir else './pseudos',
                'verbosity': 'high',
                'tprnfor': True,
                'tstress': True,
            },
            'system': {
                'ecutwfc': ecutwfc,
                'ecutrho': 4.0 * ecutwfc,
                'occupations': 'smearing',
                'smearing': 'gaussian',
                'degauss': 0.02,
            },
            'electrons': {
                'conv_thr': 1e-6,
                'mixing_beta': 0.7,
                'mixing_mode': 'plain',
                'diagonalization': 'david',
            },
            'ions': {
                'ion_dynamics': 'bfgs',
                'ion_temperature': 'not_controlled',
            },
        }
        
        if relax_cell:
            input_data['cell'] = {
                'cell_dynamics': 'bfgs',
                'cell_temperature': 'not_controlled',
            }
        
        # Create calculator - try new API first, fallback to old API
        try:
            # Try new ASE API (4.0+)
            from ase.calculators.espresso import EspressoProfile
            profile = EspressoProfile(argv=[qe_command])
            calc = Espresso(
                input_data=input_data,
                pseudopotentials=pseudopotentials,
                kpts=kpts,
                profile=profile,
            )
        except ImportError:
            # Fallback to old API (3.25.0)
            calc = Espresso(
                input_data=input_data,
                pseudopotentials=pseudopotentials,
                kpts=kpts,
                command=qe_command,
            )
        
        atoms.set_calculator(calc)
        
        # Record initial state
        initial_energy = atoms.get_potential_energy()
        initial_volume = atoms.get_volume()
        
        # Set up optimizer
        if relax_cell:
            from ase.constraints import ExpCellFilter
            ecf = ExpCellFilter(atoms)
            opt = BFGS(ecf)
        else:
            opt = BFGS(atoms)
        
        # Run optimization on server
        print(f"Optimizing structure with {calculation_type} on server...")
        print(f"QE command: {qe_command}")
        opt.run(fmax=fmax, steps=max_steps)
        
        # Record final state
        final_energy = atoms.get_potential_energy()
        final_volume = atoms.get_volume()
        
        # Generate output directory
        input_path = Path(structure_file)
        
        if _thread_id:
            output_dir = get_subdir_path(_thread_id, "calculations/optimized")
        else:
            output_dir = Path("data/outputs/calculations/optimized")
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save optimized structure
        output_filename = f"{input_path.stem}_opt_{functional}_ecut{ecutwfc:.0f}"
        output_path = output_dir / f"{output_filename}.cif"
        
        write(str(output_path), atoms)
        
        # Create metadata
        metadata = {
            "original_file": structure_file,
            "calculation_type": "structure_optimization",
            "functional": functional,
            "ecutwfc": ecutwfc,
            "kpts": kpts,
            "relax_cell": relax_cell,
            "fmax": fmax,
            "max_steps": max_steps,
            "pseudopotentials": pseudopotentials,
            "elements": elements,
            "initial_energy": initial_energy,
            "final_energy": final_energy,
            "energy_change": final_energy - initial_energy,
            "initial_volume": initial_volume,
            "final_volume": final_volume,
            "volume_change": final_volume - initial_volume,
            "num_atoms": len(atoms),
            "formula": atoms.get_chemical_formula(),
            "output_file": str(output_path),
            "qe_command": qe_command,
        }
        
        metadata_file = output_dir / f"{output_filename}.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Create summary
        summary = f"Structure optimization completed successfully!\n"
        summary += f"Functional: {functional.upper()}\n"
        summary += f"Energy change: {final_energy - initial_energy:.6f} eV\n"
        summary += f"Volume change: {final_volume - initial_volume:.2f} Å³\n"
        summary += f"Final energy: {final_energy:.6f} eV\n"
        summary += f"Formula: {atoms.get_chemical_formula()}\n"
        summary += f"Atoms: {len(atoms)}\n"
        summary += f"Cell relaxation: {relax_cell}\n"
        summary += f"\nOptimized structure saved to: {output_path}\n"
        summary += f"Metadata saved to: {metadata_file}\n"
        
        return summary
        
    except Exception as e:
        return f"Error in structure optimization: {str(e)}"


@tool
def relax_slab_dft(
    structure_file: str,
    output_dir: str,
    fixed_layers: int = 2,
    ecutwfc: float = 30.0,
    ecutrho: Optional[float] = None,
    kpts: Optional[List[int]] = None,
    fmax: float = 0.05,
    convergence_threshold: float = 1e-6,
    xc: str = "pbe",
    smearing: str = "gaussian",
    degauss: float = 0.02,
    _thread_id: Optional[str] = None,
) -> str:
    """Relax surface slab with DFT, fixing bottom layers to mimic bulk region.
    
    This function is essential for heterogeneous catalysis studies where you need
    to separate the bulk region (fixed) from the surface region (relaxable).
    
    Args:
        structure_file: Path to slab structure file
        output_dir: Directory for calculation outputs
        fixed_layers: Number of bottom layers to fix (mimics bulk region)
        ecutwfc: Kinetic energy cutoff in Ry
        ecutrho: Charge density cutoff in Ry (default: 4 * ecutwfc)
        kpts: K-point mesh (nx, ny, nz) - z should be 1 for slabs
        fmax: Force tolerance in eV/Å
        convergence_threshold: Electronic convergence threshold
        xc: Exchange-correlation functional
        smearing: Smearing type for metals
        degauss: Smearing width in Ry
        _thread_id: Thread ID for workspace management
        
    Returns:
        String with relaxation results and file path
    """
    try:
        import numpy as np
        from pathlib import Path
        from ase.io import read, write
        from ase.constraints import FixAtoms
        from ase.optimize import BFGS
        from ase.calculators.espresso import Espresso
        
        # Set defaults
        if ecutrho is None:
            ecutrho = 4 * ecutwfc
        if kpts is None:
            kpts = [6, 6, 1]  # Typical for slabs
        
        # Ensure z-direction k-points is 1 for slabs
        if len(kpts) == 3 and kpts[2] != 1:
            print(f"Warning: Setting k-points z-direction to 1 for slab calculation")
            kpts[2] = 1
        
        # Read slab structure
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
        
        print(f"Identified {len(layers)} layers in slab structure")
        
        # Fix bottom layers
        indices_to_fix = []
        if len(layers) >= fixed_layers:
            for layer_idx in range(fixed_layers):
                indices_to_fix.extend(layers[layer_idx])
                print(f"Fixing layer {layer_idx + 1} with {len(layers[layer_idx])} atoms")
            
            constraint = FixAtoms(indices=indices_to_fix)
            atoms.set_constraint(constraint)
            print(f"Total atoms fixed: {len(indices_to_fix)} out of {len(atoms)}")
        else:
            print(f"Warning: Slab has only {len(layers)} layers, cannot fix {fixed_layers} layers")
        
        # Set up output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create calculator - try new API first, fallback to old API
        try:
            # Try new ASE API (4.0+)
            from ase.calculators.espresso import EspressoProfile
            profile = EspressoProfile(argv=["pw.x"])
            calc = Espresso(
                input_data={
                    "control": {
                        "calculation": "relax",
                        "restart_mode": "from_scratch",
                        "prefix": "slab_relax",
                        "outdir": str(output_path / "tmp"),
                        "pseudo_dir": "data/pseudopotentials",
                        "tprnfor": True,
                        "tstress": True,
                    },
                    "system": {
                        "ecutwfc": ecutwfc,
                        "ecutrho": ecutrho,
                        "occupations": smearing,
                        "degauss": degauss,
                        "smearing": smearing,
                    },
                    "electrons": {
                        "electron_maxstep": 200,
                        "conv_thr": convergence_threshold,
                        "mixing_beta": 0.7,
                    },
                    "ions": {
                        "ion_dynamics": "bfgs",
                    },
                },
                pseudopotentials={
                    "H": "H.pbe-rrkjus_psl.0.1.UPF",
                    "C": "C.pbe-n-rrkjus_psl.0.1.UPF",
                    "N": "N.pbe-n-rrkjus_psl.0.1.UPF",
                    "O": "O.pbe-n-rrkjus_psl.0.1.UPF",
                    "Si": "Si.pbe-n-rrkjus_psl.0.1.UPF",
                    "Cu": "Cu.pbe-n-rrkjus_psl.0.1.UPF",
                    "Pt": "Pt.pbe-n-rrkjus_psl.0.1.UPF",
                    "Pd": "Pd.pbe-n-rrkjus_psl.0.1.UPF",
                    "Au": "Au.pbe-n-rrkjus_psl.0.1.UPF",
                    "Ag": "Ag.pbe-n-rrkjus_psl.0.1.UPF",
                },
                kpts=kpts,
                profile=profile,
            )
        except ImportError:
            # Fallback to old API (3.25.0)
            calc = Espresso(
                input_data={
                    "control": {
                        "calculation": "relax",
                        "restart_mode": "from_scratch",
                        "prefix": "slab_relax",
                        "outdir": str(output_path / "tmp"),
                        "pseudo_dir": "data/pseudopotentials",
                        "tprnfor": True,
                        "tstress": True,
                    },
                    "system": {
                        "ecutwfc": ecutwfc,
                        "ecutrho": ecutrho,
                        "occupations": smearing,
                        "degauss": degauss,
                        "smearing": smearing,
                    },
                    "electrons": {
                        "electron_maxstep": 200,
                        "conv_thr": convergence_threshold,
                        "mixing_beta": 0.7,
                    },
                    "ions": {
                        "ion_dynamics": "bfgs",
                    },
                },
                pseudopotentials={
                    "H": "H.pbe-rrkjus_psl.0.1.UPF",
                    "C": "C.pbe-n-rrkjus_psl.0.1.UPF",
                    "N": "N.pbe-n-rrkjus_psl.0.1.UPF",
                    "O": "O.pbe-n-rrkjus_psl.0.1.UPF",
                    "Si": "Si.pbe-n-rrkjus_psl.0.1.UPF",
                    "Cu": "Cu.pbe-n-rrkjus_psl.0.1.UPF",
                    "Pt": "Pt.pbe-n-rrkjus_psl.0.1.UPF",
                    "Pd": "Pd.pbe-n-rrkjus_psl.0.1.UPF",
                    "Au": "Au.pbe-n-rrkjus_psl.0.1.UPF",
                    "Ag": "Ag.pbe-n-rrkjus_psl.0.1.UPF",
                },
                kpts=kpts,
                command="pw.x",
            )
        
        atoms.set_calculator(calc)
        
        # Run relaxation
        print(f"Starting DFT slab relaxation with {fixed_layers} fixed layers...")
        initial_energy = atoms.get_potential_energy()
        
        # Use ASE optimizer for better control
        opt = BFGS(atoms, logfile=str(output_path / "optimization.log"))
        opt.run(fmax=fmax)
        
        final_energy = atoms.get_potential_energy()
        
        # Save relaxed structure
        relaxed_file = output_path / "relaxed_slab.xyz"
        write(str(relaxed_file), atoms)
        
        # Save with constraint information
        constraint_file = output_path / "constraint_info.txt"
        with open(constraint_file, "w") as f:
            f.write(f"Slab relaxation with {fixed_layers} fixed layers\n")
            f.write(f"Total atoms: {len(atoms)}\n")
            f.write(f"Fixed atoms: {len(indices_to_fix)}\n")
            f.write(f"Relaxable atoms: {len(atoms) - len(indices_to_fix)}\n")
            f.write(f"Fixed atom indices: {indices_to_fix}\n")
            f.write(f"Energy change: {final_energy - initial_energy:.6f} eV\n")
            f.write(f"Final energy: {final_energy:.6f} eV\n")
        
        # Create summary
        summary = (
            f"DFT slab relaxation completed successfully!\n"
            f"Fixed layers: {fixed_layers} (mimicking bulk region)\n"
            f"Relaxable layers: {len(layers) - fixed_layers} (surface region)\n"
            f"Total atoms: {len(atoms)}\n"
            f"Fixed atoms: {len(indices_to_fix)}\n"
            f"Energy change: {final_energy - initial_energy:.6f} eV\n"
            f"Final energy: {final_energy:.6f} eV\n"
            f"Relaxed structure saved to: {relaxed_file}\n"
            f"Constraint info saved to: {constraint_file}"
        )
        
        return summary
        
    except Exception as e:
        return f"Error in DFT slab relaxation: {str(e)}"


@tool
def test_hydrogen_atom(
    qe_command: str = "pw.x",
    pseudopotential_dir: Optional[str] = None,
    _thread_id: Optional[str] = None,
) -> str:
    """Test DFT calculation with a simple hydrogen atom on server.

    Args:
        qe_command: Quantum ESPRESSO command (default: "pw.x" for server)
        pseudopotential_dir: Directory containing pseudopotential files

    Returns:
        String with test results
    """
    try:
        # Create hydrogen atom in a box
        from ase import Atoms
        
        # Create H atom in 10x10x10 Å box
        atoms = Atoms('H', positions=[(5, 5, 5)], cell=[10, 10, 10], pbc=True)
        
        # Set up pseudopotentials
        pseudopotentials = {'H': 'H.pbe-kjpaw_psl.1.0.0.UPF'}
        
        # Set up calculator
        input_data = {
            'control': {
                'calculation': 'scf',
                'restart_mode': 'from_scratch',
                'prefix': 'pwscf',
                'outdir': './tmp',
                'pseudo_dir': str(pseudopotential_dir) if pseudopotential_dir else './pseudos',
                'verbosity': 'high',
            },
            'system': {
                'ecutwfc': 30.0,
                'ecutrho': 120.0,
                'occupations': 'smearing',
                'smearing': 'gaussian',
                'degauss': 0.02,
            },
            'electrons': {
                'conv_thr': 1e-6,
                'mixing_beta': 0.7,
                'mixing_mode': 'plain',
                'diagonalization': 'david',
            },
        }
        
        # Create calculator - try new API first, fallback to old API
        try:
            # Try new ASE API (4.0+)
            from ase.calculators.espresso import EspressoProfile
            profile = EspressoProfile(argv=[qe_command])
            calc = Espresso(
                input_data=input_data,
                pseudopotentials=pseudopotentials,
                kpts=(1, 1, 1),  # Single k-point for isolated atom
                profile=profile,
            )
        except ImportError:
            # Fallback to old API (3.25.0)
            calc = Espresso(
                input_data=input_data,
                pseudopotentials=pseudopotentials,
                kpts=(1, 1, 1),  # Single k-point for isolated atom
                command=qe_command,
            )
        
        atoms.set_calculator(calc)
        
        print("Testing hydrogen atom calculation on server...")
        print(f"QE command: {qe_command}")
        print(f"Pseudopotential: {pseudopotentials['H']}")
        
        # Run calculation on server
        energy = atoms.get_potential_energy()
        
        # Save test results
        if _thread_id:
            output_dir = get_subdir_path(_thread_id, "calculations/test_results")
        else:
            output_dir = Path("data/outputs/calculations/test_results")
            output_dir.mkdir(parents=True, exist_ok=True)
        
        test_file = output_dir / "hydrogen_atom_test.cif"
        write(str(test_file), atoms)
        
        # Create test metadata
        test_metadata = {
            "test_type": "hydrogen_atom",
            "qe_command": qe_command,
            "pseudopotential": pseudopotentials['H'],
            "energy": energy,
            "cell_size": [10, 10, 10],
            "kpts": [1, 1, 1],
            "ecutwfc": 30.0,
            "ecutrho": 120.0,
            "test_file": str(test_file),
            "status": "success",
        }
        
        metadata_file = output_dir / "hydrogen_atom_test.json"
        with open(metadata_file, "w") as f:
            json.dump(test_metadata, f, indent=2)
        
        summary = f"Hydrogen atom test completed successfully!\n"
        summary += f"Energy: {energy:.6f} eV\n"
        summary += f"QE command: {qe_command}\n"
        summary += f"Pseudopotential: {pseudopotentials['H']}\n"
        summary += f"Test file saved to: {test_file}\n"
        summary += f"Metadata saved to: {metadata_file}\n"
        
        return summary
        
    except Exception as e:
        return f"Error in hydrogen atom test: {str(e)}"
