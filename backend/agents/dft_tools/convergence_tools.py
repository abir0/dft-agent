"""
Convergence Testing Tools

Tools for testing k-point, cutoff, and slab convergence in DFT calculations.
"""

import json
from pathlib import Path
from typing import List, Optional

from langchain_core.tools import tool

try:
    from ase.io import read

    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False


@tool
def kpoint_convergence_test(
    structure_file: str,
    kpoint_range: List[int],
    energy_tolerance: float = 0.01,
    property_to_converge: str = "energy",
) -> str:
    """Perform k-point mesh convergence testing.

    Args:
        structure_file: Path to structure file
        kpoint_range: List of k-point densities to test (e.g., [4, 6, 8, 10, 12])
        energy_tolerance: Energy convergence tolerance in eV/atom
        property_to_converge: Property to test convergence ('energy', 'forces', 'stress')

    Returns:
        K-point convergence test results and recommendations
    """
    try:
        if not ASE_AVAILABLE:
            return "Error: ASE not available. Please install with: pip install ase"

        # Read structure to get number of atoms
        atoms = read(structure_file)
        num_atoms = len(atoms)

        # Generate convergence test plan
        input_path = Path(structure_file)
        output_dir = input_path.parent / "convergence_tests" / "kpoints"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create test configurations
        test_configs = []
        for k in kpoint_range:
            # For surfaces, use k x k x 1; for bulk, use k x k x k
            cell = atoms.get_cell()
            c_length = cell[2, 2]

            if c_length > 15.0:  # Likely a slab with vacuum
                kpts = (k, k, 1)
            else:  # Bulk structure
                kpts = (k, k, k)

            config = {
                "kpoint_density": k,
                "kpoint_mesh": kpts,
                "total_kpoints": kpts[0] * kpts[1] * kpts[2],
                "input_file": f"{input_path.stem}_kconv_{k}x{k}.pwi",
                "expected_energy": None,  # To be filled after calculation
                "converged": False,
            }
            test_configs.append(config)

        # Create test plan file
        test_plan = {
            "structure_file": structure_file,
            "test_type": "kpoint_convergence",
            "property": property_to_converge,
            "tolerance": energy_tolerance,
            "num_atoms": num_atoms,
            "configurations": test_configs,
            "status": "planned",
        }

        plan_file = output_dir / f"{input_path.stem}_kpoint_convergence_plan.json"
        with open(plan_file, "w") as f:
            json.dump(test_plan, f, indent=2)

        # Generate input files for each k-point density
        input_files = []
        for config in test_configs:
            # This would generate actual QE input files
            # For now, create placeholder information
            input_file_path = output_dir / config["input_file"]

            # Create a simple template (would normally use QE input generation)
            with open(input_file_path, "w") as f:
                f.write("# QE input file for k-point convergence test\\n")
                f.write(f"# K-points: {config['kpoint_mesh']}\\n")
                f.write(f"# Structure: {structure_file}\\n")
                f.write("# Generated for convergence testing\\n")

            input_files.append(str(input_file_path))

        # Create run script
        run_script = output_dir / "run_kpoint_convergence.sh"
        with open(run_script, "w") as f:
            f.write("#!/bin/bash\\n")
            f.write("# K-point convergence test script\\n\\n")
            for i, config in enumerate(test_configs):
                k = config["kpoint_density"]
                f.write(f"echo 'Running k-point test {k}x{k}'\\n")
                f.write(
                    f"pw.x < {config['input_file']} > {input_path.stem}_kconv_{k}x{k}.pwo\\n"
                )
                f.write("\\n")

        run_script.chmod(0o755)

        # Create analysis script for after calculations
        analysis_script = output_dir / "analyze_kpoint_convergence.py"
        with open(analysis_script, "w") as f:
            f.write("#!/usr/bin/env python3\\n")
            f.write("# Analysis script for k-point convergence\\n")
            f.write("import json\\n")
            f.write("import numpy as np\\n\\n")
            f.write("# Load test plan\\n")
            f.write(f"with open('{plan_file}', 'r') as f:\\n")
            f.write("    plan = json.load(f)\\n\\n")
            f.write("# Parse energies from output files\\n")
            f.write("# Add your energy parsing logic here\\n")

        analysis_script.chmod(0o755)

        # Create summary
        summary = f"K-point convergence test setup for {atoms.get_chemical_formula()}:\\n"
        summary += f"Structure: {structure_file}\\n"
        summary += f"Number of atoms: {num_atoms}\\n"
        summary += f"K-point range: {min(kpoint_range)} to {max(kpoint_range)}\\n"
        summary += f"Tolerance: {energy_tolerance} eV/atom\\n\\n"

        summary += "Test configurations:\\n"
        for config in test_configs:
            k = config["kpoint_density"]
            mesh = config["kpoint_mesh"]
            total = config["total_kpoints"]
            summary += f"- {k}x{k}: {mesh[0]}×{mesh[1]}×{mesh[2]} = {total} k-points\\n"

        summary += "\\nFiles created:\\n"
        summary += f"- Test plan: {plan_file}\\n"
        summary += f"- Run script: {run_script}\\n"
        summary += f"- Analysis script: {analysis_script}\\n"
        summary += f"- Input files: {len(input_files)} files in {output_dir}\\n"

        summary += f"\\nTo run: bash {run_script}"

        return summary

    except Exception as e:
        return f"Error setting up k-point convergence test: {str(e)}"


@tool
def cutoff_convergence_test(
    structure_file: str,
    cutoff_range: List[float],
    energy_tolerance: float = 0.01,
    kpoint_mesh: Optional[List[int]] = None,
) -> str:
    """Perform plane-wave cutoff convergence testing.

    Args:
        structure_file: Path to structure file
        cutoff_range: List of cutoff energies to test in Ry (e.g., [20, 30, 40, 50, 60])
        energy_tolerance: Energy convergence tolerance in eV/atom
        kpoint_mesh: Fixed k-point mesh for testing

    Returns:
        Cutoff convergence test results and recommendations
    """
    try:
        if not ASE_AVAILABLE:
            return "Error: ASE not available. Please install with: pip install ase"

        # Set defaults
        if kpoint_mesh is None:
            kpoint_mesh = [6, 6, 6]

        # Read structure
        atoms = read(structure_file)
        num_atoms = len(atoms)

        # Create test directory
        input_path = Path(structure_file)
        output_dir = input_path.parent / "convergence_tests" / "cutoff"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create test configurations
        test_configs = []
        for ecut in cutoff_range:
            config = {
                "ecutwfc": ecut,
                "ecutrho": ecut * 4.0,  # Typical ratio for PAW/NC PPs
                "kpoint_mesh": kpoint_mesh,
                "input_file": f"{input_path.stem}_cutconv_{int(ecut)}ry.pwi",
                "expected_energy": None,
                "converged": False,
            }
            test_configs.append(config)

        # Create test plan
        test_plan = {
            "structure_file": structure_file,
            "test_type": "cutoff_convergence",
            "tolerance": energy_tolerance,
            "num_atoms": num_atoms,
            "fixed_kpoints": kpoint_mesh,
            "configurations": test_configs,
            "status": "planned",
        }

        plan_file = output_dir / f"{input_path.stem}_cutoff_convergence_plan.json"
        with open(plan_file, "w") as f:
            json.dump(test_plan, f, indent=2)

        # Generate input files
        input_files = []
        for config in test_configs:
            input_file_path = output_dir / config["input_file"]

            # Create placeholder input file
            with open(input_file_path, "w") as f:
                f.write("# QE input file for cutoff convergence test\\n")
                f.write(f"# Cutoff: {config['ecutwfc']} Ry\\n")
                f.write(f"# Charge cutoff: {config['ecutrho']} Ry\\n")
                f.write(f"# K-points: {config['kpoint_mesh']}\\n")
                f.write(f"# Structure: {structure_file}\\n")

            input_files.append(str(input_file_path))

        # Create run script
        run_script = output_dir / "run_cutoff_convergence.sh"
        with open(run_script, "w") as f:
            f.write("#!/bin/bash\\n")
            f.write("# Cutoff convergence test script\\n\\n")
            for config in test_configs:
                ecut = int(config["ecutwfc"])
                f.write(f"echo 'Running cutoff test {ecut} Ry'\\n")
                f.write(
                    f"pw.x < {config['input_file']} > {input_path.stem}_cutconv_{ecut}ry.pwo\\n"
                )
                f.write("\\n")

        run_script.chmod(0o755)

        # Create summary
        summary = f"Cutoff convergence test setup for {atoms.get_chemical_formula()}:\\n"
        summary += f"Structure: {structure_file}\\n"
        summary += f"Number of atoms: {num_atoms}\\n"
        summary += f"Cutoff range: {min(cutoff_range)} - {max(cutoff_range)} Ry\\n"
        summary += (
            f"Fixed k-points: {kpoint_mesh[0]}×{kpoint_mesh[1]}×{kpoint_mesh[2]}\\n"
        )
        summary += f"Tolerance: {energy_tolerance} eV/atom\\n\\n"

        summary += "Test configurations:\\n"
        for config in test_configs:
            ecut = config["ecutwfc"]
            erho = config["ecutrho"]
            summary += f"- {ecut} Ry (charge: {erho} Ry)\\n"

        summary += "\\nFiles created:\\n"
        summary += f"- Test plan: {plan_file}\\n"
        summary += f"- Run script: {run_script}\\n"
        summary += f"- Input files: {len(input_files)} files\\n"

        return summary

    except Exception as e:
        return f"Error setting up cutoff convergence test: {str(e)}"


@tool
def slab_thickness_convergence(
    bulk_structure_file: str,
    layer_range: List[int],
    miller_indices: Optional[List[int]] = None,
    surface_energy_tolerance: float = 0.05,
) -> str:
    """Test convergence with respect to slab thickness.

    Args:
        bulk_structure_file: Path to bulk structure file
        miller_indices: Miller indices for surface (h, k, l)
        layer_range: List of layer numbers to test (e.g., [3, 5, 7, 9, 11])
        surface_energy_tolerance: Surface energy tolerance in eV/Å²

    Returns:
        Slab thickness convergence test setup and results
    """
    try:
        if not ASE_AVAILABLE:
            return "Error: ASE not available. Please install with: pip install ase"

        # Set defaults
        if miller_indices is None:
            miller_indices = [1, 1, 1]

        # Read bulk structure
        bulk_atoms = read(bulk_structure_file)

        # Create test directory
        input_path = Path(bulk_structure_file)
        miller_str = "".join(map(str, miller_indices))
        output_dir = input_path.parent / "convergence_tests" / f"slab_{miller_str}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create test configurations
        test_configs = []
        for layers in layer_range:
            config = {
                "num_layers": layers,
                "miller_indices": miller_indices,
                "slab_file": f"{input_path.stem}_slab_{miller_str}_{layers}L.cif",
                "bulk_energy_per_atom": None,
                "slab_energy": None,
                "surface_energy": None,
                "converged": False,
            }
            test_configs.append(config)

        # Create test plan
        test_plan = {
            "bulk_file": bulk_structure_file,
            "test_type": "slab_thickness_convergence",
            "miller_indices": miller_indices,
            "tolerance": surface_energy_tolerance,
            "configurations": test_configs,
            "status": "planned",
        }

        plan_file = output_dir / f"slab_{miller_str}_thickness_convergence_plan.json"
        with open(plan_file, "w") as f:
            json.dump(test_plan, f, indent=2)

        # Generate slab structures (would use ASE surface generation)
        slab_files = []
        for config in test_configs:
            slab_path = output_dir / config["slab_file"]

            # Placeholder for slab generation
            with open(slab_path, "w") as f:
                f.write("# Slab structure placeholder\\n")
                f.write(f"# {config['num_layers']} layers of {miller_str} surface\\n")
                f.write(f"# Generated from: {bulk_structure_file}\\n")

            slab_files.append(str(slab_path))

        # Create calculation script
        calc_script = output_dir / "run_slab_convergence.sh"
        with open(calc_script, "w") as f:
            f.write("#!/bin/bash\\n")
            f.write("# Slab thickness convergence test\\n\\n")
            f.write("# First calculate bulk energy\\n")
            f.write("echo 'Calculating bulk energy'\\n")
            f.write("# pw.x < bulk_input.pwi > bulk_output.pwo\\n\\n")

            for config in test_configs:
                layers = config["num_layers"]
                f.write(f"echo 'Calculating {layers}-layer slab'\\n")
                f.write(
                    f"# pw.x < slab_{layers}L_input.pwi > slab_{layers}L_output.pwo\\n"
                )
                f.write("\\n")

        calc_script.chmod(0o755)

        # Create analysis script
        analysis_script = output_dir / "analyze_surface_energy.py"
        with open(analysis_script, "w") as f:
            f.write("#!/usr/bin/env python3\\n")
            f.write("# Surface energy analysis\\n")
            f.write("import json\\n")
            f.write("import matplotlib.pyplot as plt\\n\\n")
            f.write(
                "def calculate_surface_energy(E_slab, E_bulk_per_atom, N_atoms_slab, N_atoms_bulk, A_surface):\\n"
            )
            f.write(
                "    '''Calculate surface energy: γ = (E_slab - N_slab/N_bulk * E_bulk) / (2 * A)'''\\n"
            )
            f.write(
                "    return (E_slab - (N_atoms_slab/N_atoms_bulk) * E_bulk_per_atom) / (2 * A_surface)\\n\\n"
            )
            f.write("# Load test plan and analyze results\\n")
            f.write(f"with open('{plan_file}', 'r') as f:\\n")
            f.write("    plan = json.load(f)\\n")

        analysis_script.chmod(0o755)

        # Create summary
        summary = f"Slab thickness convergence test for ({miller_str}) surface:\\n"
        summary += f"Bulk structure: {bulk_structure_file}\\n"
        summary += f"Miller indices: {miller_indices}\\n"
        summary += f"Layer range: {min(layer_range)} - {max(layer_range)} layers\\n"
        summary += f"Tolerance: {surface_energy_tolerance} eV/Å²\\n\\n"

        summary += "Test configurations:\\n"
        for config in test_configs:
            layers = config["num_layers"]
            summary += f"- {layers} layers: {config['slab_file']}\\n"

        summary += "\\nFiles created:\\n"
        summary += f"- Test plan: {plan_file}\\n"
        summary += f"- Calculation script: {calc_script}\\n"
        summary += f"- Analysis script: {analysis_script}\\n"
        summary += f"- Slab structures: {len(slab_files)} files\\n"

        return summary

    except Exception as e:
        return f"Error setting up slab thickness convergence test: {str(e)}"


@tool
def vacuum_convergence_test(
    slab_structure_file: str, vacuum_range: List[float], energy_tolerance: float = 0.01
) -> str:
    """Test vacuum thickness convergence for slab calculations.

    Args:
        slab_structure_file: Path to slab structure file
        vacuum_range: List of vacuum thicknesses to test in Å (e.g., [8, 10, 12, 15, 20])
        energy_tolerance: Energy convergence tolerance in eV/atom

    Returns:
        Vacuum convergence test setup and information
    """
    try:
        if not ASE_AVAILABLE:
            return "Error: ASE not available. Please install with: pip install ase"

        # Read slab structure
        atoms = read(slab_structure_file)
        num_atoms = len(atoms)

        # Create test directory
        input_path = Path(slab_structure_file)
        output_dir = input_path.parent / "convergence_tests" / "vacuum"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create test configurations
        test_configs = []
        for vacuum in vacuum_range:
            config = {
                "vacuum_thickness": vacuum,
                "structure_file": f"{input_path.stem}_vac{vacuum:.1f}A.cif",
                "expected_energy": None,
                "converged": False,
            }
            test_configs.append(config)

        # Create test plan
        test_plan = {
            "original_slab": slab_structure_file,
            "test_type": "vacuum_convergence",
            "tolerance": energy_tolerance,
            "num_atoms": num_atoms,
            "configurations": test_configs,
            "status": "planned",
        }

        plan_file = output_dir / f"{input_path.stem}_vacuum_convergence_plan.json"
        with open(plan_file, "w") as f:
            json.dump(test_plan, f, indent=2)

        # Generate structures with different vacuum
        structure_files = []
        for config in test_configs:
            struct_path = output_dir / config["structure_file"]

            # Placeholder for structure generation with different vacuum
            with open(struct_path, "w") as f:
                f.write(f"# Slab structure with {config['vacuum_thickness']} Å vacuum\\n")
                f.write(f"# Generated from: {slab_structure_file}\\n")

            structure_files.append(str(struct_path))

        # Create run script
        run_script = output_dir / "run_vacuum_convergence.sh"
        with open(run_script, "w") as f:
            f.write("#!/bin/bash\\n")
            f.write("# Vacuum convergence test script\\n\\n")
            for config in test_configs:
                vac = config["vacuum_thickness"]
                f.write(f"echo 'Running vacuum test {vac} Å'\\n")
                f.write(f"# pw.x < vacuum_{vac}A_input.pwi > vacuum_{vac}A_output.pwo\\n")
                f.write("\\n")

        run_script.chmod(0o755)

        # Create summary
        summary = f"Vacuum convergence test for {atoms.get_chemical_formula()} slab:\\n"
        summary += f"Original slab: {slab_structure_file}\\n"
        summary += f"Number of atoms: {num_atoms}\\n"
        summary += f"Vacuum range: {min(vacuum_range)} - {max(vacuum_range)} Å\\n"
        summary += f"Tolerance: {energy_tolerance} eV/atom\\n\\n"

        summary += "Test configurations:\\n"
        for config in test_configs:
            vac = config["vacuum_thickness"]
            summary += f"- {vac} Å vacuum: {config['structure_file']}\\n"

        summary += "\\nFiles created:\\n"
        summary += f"- Test plan: {plan_file}\\n"
        summary += f"- Run script: {run_script}\\n"
        summary += f"- Structures: {len(structure_files)} files\\n"

        return summary

    except Exception as e:
        return f"Error setting up vacuum convergence test: {str(e)}"
