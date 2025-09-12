"""
Quantum ESPRESSO Interface Tools

Tools for generating QE input files, submitting jobs, and parsing output.
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from ase.io import read, write
from langchain_core.tools import tool

from backend.settings import settings
from backend.utils.workspace import get_subdir_path


def get_kpoints(atoms, kspacing: float = 0.15) -> list:
    """Returns the kpoints of a given ase atoms object and specific kspacing."""
    cell = atoms.get_cell()

    # Calculate k-points based on reciprocal lattice vectors
    kpts = []
    for i in range(3):
        if np.linalg.norm(cell[i]) > 0.1:  # Avoid division by zero for very small cells
            k_val = 2 * (
                int(np.ceil(2 * np.pi / np.linalg.norm(cell[i]) / kspacing)) // 2 + 1
            )
            # Ensure odd k-points for better sampling
            if k_val % 2 == 0 and k_val > 1:
                k_val -= 1
            elif k_val % 2 == 0:
                k_val += 1
            kpts.append(max(1, k_val))
        else:
            kpts.append(1)

    return kpts


@tool
def generate_qe_input(
    structure_file: str,
    calculation: str = "scf",
    ecutwfc: float = 30.0,
    ecutrho: Optional[float] = None,
    kpts: Optional[List[int]] = None,
    kspacing: float = 0.15,
    occupations: str = "smearing",
    smearing: str = "gaussian",
    degauss: float = 0.02,
    pseudopotentials: Optional[Dict[str, str]] = None,
    job_name: str = "pwscf",
    restart_mode: str = "from_scratch",
    input_dft: str = "PBE",
    disk_io: str = "none",
    _thread_id: Optional[str] = None,
) -> str:
    """Generate Quantum ESPRESSO input file (pw.x) from structure and parameters using ASE.

    Args:
        structure_file: Path to structure file readable by ASE
        calculation: QE calculation type ('scf', 'relax', 'bands', 'nscf', 'vc-relax')
        ecutwfc: Wavefunction cutoff in Ry
        ecutrho: Charge density cutoff in Ry (default: 4*ecutwfc)
        kpts: Monkhorst-Pack grid [nx, ny, nz]; auto if None
        kspacing: Reciprocal lattice spacing in Angstrom^-1 (default: 0.15)
        occupations: Occupation method
        smearing: Smearing type (if occupations='smearing')
        degauss: Smearing width in Ry
        pseudopotentials: Dict mapping element → pseudopotential filename
        job_name: Prefix for job and filenames
        restart_mode: QE restart mode
        input_dft: XC functional (default PBE)
        disk_io: QE disk_io setting
        _thread_id: Optional workspace thread ID

    Returns:
        Message string with summary of generated input
    """
    try:
        # Input validation
        valid_calculations = ["scf", "relax", "bands", "nscf", "vc-relax", "ensemble"]
        if calculation not in valid_calculations:
            raise ValueError(f"Invalid calculation type '{calculation}'")

        valid_occupations = [
            "smearing",
            "tetrahedra",
            "tetrahedra_lin",
            "tetrahedra_opt",
            "fixed",
            "from_input",
        ]
        if occupations not in valid_occupations:
            raise ValueError(f"Invalid occupation type '{occupations}'")

        valid_smearing = [
            "gaussian",
            "methfessel-paxton",
            "marzari-vanderbilt",
            "fermi-dirac",
        ]
        if smearing not in valid_smearing:
            raise ValueError(f"Invalid smearing type '{smearing}'")

        if ecutwfc <= 0:
            raise ValueError("ecutwfc must be positive")

        # Resolve structure file path relative to workspace if thread_id is provided
        structure_path = Path(structure_file)
        if not structure_path.exists():
            raise FileNotFoundError(f"Structure file not found: {structure_path}")

        # Handle different input formats
        if str(structure_path).endswith(".pwi") and calculation != "ensemble":
            # For .pwi files, try to read the corresponding .pwo file first
            pwo_file = str(structure_path).replace(".pwi", ".pwo")
            if os.path.exists(pwo_file):
                atoms = read(pwo_file)
            else:
                atoms = read(str(structure_path))
        else:
            atoms = read(str(structure_path))

        # Get unique elements and validate structure
        elements = sorted(list(set(atoms.get_chemical_symbols())))
        nat = len(atoms)
        ntyp = len(elements)

        if nat == 0:
            raise ValueError("Structure contains no atoms")

        # Set ecutrho default for charge density
        if ecutrho is None:
            ecutrho = 4.0 * ecutwfc

        # Generate k-points automatically
        if kpts is None:
            kpts = get_kpoints(atoms, kspacing)

        # Default pseudopotentials
        if pseudopotentials is None:
            pseudopotentials = {}
            for el in elements:
                # Use more comprehensive PP selection
                if el in ["H", "He"]:
                    pseudopotentials[el] = f"{el}.pbe-rrkjus_psl.1.0.0.UPF"
                elif el in ["Li", "Be", "B", "C", "N", "O", "F", "Ne"]:
                    pseudopotentials[el] = f"{el}.pbe-n-rrkjus_psl.1.0.0.UPF"
                elif el in ["Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar"]:
                    pseudopotentials[el] = f"{el}.pbe-n-rrkjus_psl.1.0.0.UPF"
                else:
                    # For transition metals and heavy elements
                    pseudopotentials[el] = f"{el}.pbe-spn-rrkjus_psl.1.0.0.UPF"

        for k, v in pseudopotentials.items():
            pseudopotentials[k] = Path(str(v)).name  # Use only filename

        # Use workspace-specific directory if thread_id is available
        output_dir = get_subdir_path(_thread_id, "calculations")

        # Generate filename
        input_filename = f"{job_name}_{calculation}"
        if occupations == "smearing":
            input_filename += f"_{smearing}"
        input_filename += f"_ecut{ecutwfc:.0f}.pwi"
        input_filepath = output_dir / input_filename

        pseudo_dir = f"{settings.ROOT_PATH}/WORKSPACE/pseudos"

        lspinorb = any("rel" in pp for pp in pseudopotentials.values())

        # QE input data dictionary
        input_data = {
            "control": {
                "calculation": calculation,
                "restart_mode": restart_mode,
                "prefix": job_name,
                "outdir": output_dir / "tmp",
                "pseudo_dir": pseudo_dir,
                "verbosity": "high",
                "disk_io": disk_io,
            },
            "system": {
                "ibrav": 0,  # Use cell vectors
                "nat": nat,
                "ntyp": ntyp,
                "ecutwfc": ecutwfc,
                "ecutrho": ecutrho,
                "occupations": occupations,
                "input_dft": input_dft,
                "lspinorb": lspinorb,
                "noncolin": lspinorb,
            },
            "electrons": {
                "conv_thr": 1e-8,
                "electron_maxstep": 200,
                "mixing_beta": 0.7,
                "mixing_mode": "plain",
                "diagonalization": "david",
            },
        }

        # Add smearing if provided
        if occupations == "smearing":
            input_data["system"]["smearing"] = smearing
            input_data["system"]["degauss"] = degauss

        # Add calculation-specific parameters
        # Add relaxation options
        if calculation in ["relax", "vc-relax"]:
            input_data["ions"] = {
                "ion_dynamics": "bfgs",
                "forc_conv_thr": 1e-4,
            }
        if calculation == "vc-relax":
            input_data["cell"] = {"cell_dynamics": "bfgs"}
        if calculation == "bands":
            input_data["system"]["occupations"] = "fixed"
            input_data["system"].pop("smearing", None)
            input_data["system"].pop("degauss", None)
        if calculation == "nscf":
            input_data["system"]["occupations"] = "tetrahedra"
            input_data["system"].pop("smearing", None)
            input_data["system"].pop("degauss", None)

        # Write input file using ASE's espresso-in format
        write(
            str(input_filepath),
            atoms,
            format="espresso-in",
            input_data=input_data,
            pseudopotentials=pseudopotentials,
            kpts=tuple(kpts),
        )

        # Verify the file was written correctly
        if not input_filepath.exists():
            raise RuntimeError(f"Failed to write input file: {input_filepath}")

        # Create job metadata
        job_info = {
            "structure_file": str(structure_file),
            "input_file": str(input_filepath),
            "job_name": job_name,
            "calculation": calculation,
            "ecutwfc": ecutwfc,
            "ecutrho": ecutrho,
            "kpoints": kpts,
            "kspacing": kspacing,
            "occupations": occupations,
            "smearing": smearing,
            "degauss": degauss,
            "pseudopotentials": pseudopotentials,
            "pseudo_dir": pseudo_dir,
            "elements": elements,
            "nat": nat,
            "ntyp": ntyp,
            "conv_thr": input_data["electrons"]["conv_thr"],
            "input_dft": input_dft,
        }

        # Save metadata
        metadata_file = input_filepath.with_suffix(".json")
        with open(metadata_file, "w") as f:
            json.dump(job_info, f, indent=2)

        # Calculate estimated memory usage
        mem_estimate = nat * ntyp * ecutwfc * 0.1  # Very rough estimate in MB

        return (
            f"Generated QE {calculation} input successfully:\n"
            f"  Input file: {input_filepath}\n"
            f"  Structure: {nat} atoms ({atoms.get_chemical_formula()})\n"
            f"  Elements: {', '.join(elements)}\n"
            f"  K-points: {kpts[0]}x{kpts[1]}x{kpts[2]} = {np.prod(kpts)} total\n"
            f"  Cutoffs: wfc={ecutwfc} Ry, rho={ecutrho} Ry\n"
            f"  Occupation: {occupations}\n"
            f"  Estimated memory: ~{mem_estimate:.1f} MB\n"
            f"  Metadata: {metadata_file}"
        )

    except Exception as e:
        return f"Error generating QE input: {str(e)}"


@tool
def submit_local_job(
    input_files: Dict[str, str],
    executable: str = "pw.x",
    num_cores: int = 1,
    memory_limit: str = "2GB",
) -> str:
    """Submit a DFT calculation to local machine.
    To submit a job, provide a dictionary mapping calculation types
    to input file paths. Supports multiple calculations in one call.

    Args:
        input_files: Dict of calculation_type ('scf', 'relax', 'bands', ...) -> input_file_path
        executable: QE executable name
        num_cores: Number of CPU cores to use
        memory_limit: Memory limit for the job

    Returns:
        Job submission information and process IDs
    """
    try:
        job_results = {}

        for calc_type, input_file in input_files.items():
            # Validate calculation type
            valid_calculations = ["scf", "relax", "bands", "nscf", "vc-relax", "ensemble"]
            if calc_type not in valid_calculations:
                raise ValueError(f"Invalid calculation type '{calc_type}'")
            # Validate input file
            input_path = Path(input_file)
            if not input_path.exists():
                job_results[calc_type] = {
                    "input_file": input_file,
                    "error": "Input file not found",
                    "status": "failed_to_submit",
                }
                continue

            # Handle .pwi input files by creating a new script
            output_file = input_path.with_suffix(".pwo")

            # Prepare MPI command
            cmd = (
                f"mpirun -np {num_cores} {executable} < {input_path.name} > {output_file.name}"
                if num_cores > 1
                else f"{executable} < {input_path.name} > {output_file.name}"
            )

            # Create job directory
            job_dir = input_path.parent / "jobs"
            job_dir.mkdir(exist_ok=True)

            # Create job script
            job_script = job_dir / f"job_{calc_type}_{input_path.stem}.sh"
            with open(job_script, "w") as f:
                f.write("#!/bin/bash\n")
                f.write(f"# Local QE job for {calc_type} : {input_path.stem}\n\n")
                f.write(f"cd {input_path.parent}\n")
                f.write(f"echo 'Starting {calc_type} calculation at $(date)'\n")
                f.write(f"echo 'Using {num_cores} cores, memory limit {memory_limit}'\n")
                f.write(f"{cmd}\n")
                f.write(f"echo 'Finished {calc_type} calculation at $(date)'\n")

            job_script.chmod(0o755)

            # Submit job
            try:
                process = subprocess.Popen(
                    ["bash", str(job_script)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=job_script.parent,
                )

                job_results[calc_type] = {
                    "input_file": str(input_file),
                    "output_file": str(output_file),
                    "job_script": str(job_script),
                    "process_id": process.pid,
                    "status": "submitted",
                    "command": cmd,
                }

            except Exception as e:
                job_results[calc_type] = {
                    "input_file": str(input_file),
                    "error": str(e),
                    "status": "failed_to_submit",
                }

        # Save job metadata (use the parent directory of the last processed file)
        jobs_file = None
        if job_results:
            last_input = list(input_files.values())[-1]
            metadata_dir = Path(last_input).parent
            if Path(last_input).suffix != ".sh":
                metadata_dir = metadata_dir / "jobs"
                metadata_dir.mkdir(exist_ok=True)

            jobs_file = metadata_dir / "job_status.json"
            with open(jobs_file, "w") as f:
                json.dump(job_results, f, indent=2)

        # Summary
        summary = f"Submitted {len(input_files)} local jobs:\n"
        for calc_type, result in job_results.items():
            if "process_id" in result:
                summary += (
                    f"- {calc_type}: PID {result['process_id']} ({result['status']})\n"
                )
            else:
                summary += (
                    f"- {calc_type}: {result['status']} - {result.get('error', '')}\n"
                )

        if jobs_file:
            summary += f"\nJob status saved to: {jobs_file}"
        return summary

    except Exception as e:
        return f"Error submitting local job: {str(e)}"


@tool
def check_job_status(
    job_id: str,
    queue_system: str = "local",
    remote_host: Optional[str] = None,
) -> str:
    """Monitor job execution status.

    Args:
        job_id: Job ID or process ID
        queue_system: Queue system type ('local', 'slurm', 'pbs')
        remote_host: Remote host for SSH connection

    Returns:
        Job status information
    """
    try:
        system = queue_system.lower()

        if system == "local":
            try:
                pid = int(job_id)
                result = subprocess.run(
                    ["ps", "-p", str(pid)], capture_output=True, text=True
                )
                if result.returncode == 0:
                    return f"Job {job_id} is still running (PID: {pid})"
                return f"Job {job_id} has finished or does not exist"
            except ValueError:
                return f"Invalid job ID for local system: {job_id}"

        elif system == "slurm":
            cmd = ["squeue", "-j", job_id, "-h", "-o", "%T"]
            if remote_host:
                cmd = ["ssh", remote_host, *cmd]
            result = subprocess.run(cmd, capture_output=True, text=True)
            return (
                f"SLURM job {job_id} status: {result.stdout.strip()}"
                if result.returncode == 0
                else f"SLURM job {job_id} not found or completed"
            )

        elif system == "pbs":
            cmd = ["qstat", job_id]
            if remote_host:
                cmd = ["ssh", remote_host, *cmd]
            result = subprocess.run(cmd, capture_output=True, text=True)
            return (
                f"PBS job {job_id} status:\n{result.stdout}"
                if result.returncode == 0
                else f"PBS job {job_id} not found or completed"
            )

        else:
            return f"Unknown queue system: {queue_system}"

    except Exception as e:
        return f"Error checking job status: {str(e)}"


@tool
def read_output_file(
    output_file: str,
    code_type: str = "qe",
    extract_properties: Optional[List[str]] = None,
) -> str:
    """Parse DFT output files and extract properties.

    Args:
        output_file: Path to output file
        code_type: DFT code type ('qe', 'vasp')
        extract_properties: List of properties to extract

    Returns:
        Extracted properties and information
    """
    try:
        if extract_properties is None:
            extract_properties = ["energy", "forces", "stress", "convergence"]

        output_path = Path(output_file)
        if not output_path.exists():
            return f"Error: Output file not found: {output_file}"

        results = {
            "output_file": str(output_file),
            "code_type": code_type,
            "properties": {},
        }

        if code_type.lower() == "qe":
            with open(output_file, "r") as f:
                content = f.read()

            # Energy
            if "energy" in extract_properties:
                energy_lines = [
                    line
                    for line in content.splitlines()
                    if "!" in line and "total energy" in line
                ]
                if energy_lines:
                    try:
                        energy_val = float(
                            energy_lines[-1].split("=")[1].split("Ry")[0].strip()
                        )
                        results["properties"]["total_energy_ry"] = energy_val
                        results["properties"]["total_energy_ev"] = (
                            energy_val * 13.60569301
                        )
                    except Exception:
                        results["properties"]["energy_error"] = "Failed to parse energy"

            # Convergence
            if "convergence" in extract_properties:
                if "convergence has been achieved" in content.lower():
                    results["properties"]["converged"] = True
                elif "convergence NOT achieved" in content:
                    results["properties"]["converged"] = False
                else:
                    results["properties"]["converged"] = "unknown"

            # Forces
            if "forces" in extract_properties:
                results["properties"]["forces_available"] = (
                    "Forces acting on atoms" in content
                )

            # Stress
            if "stress" in extract_properties:
                results["properties"]["stress_available"] = "total   stress" in content

        # Save parsed results
        results_file = output_path.with_suffix(".parsed.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        # Summary
        summary = f"Parsed {code_type.upper()} output: {output_file}\n"
        for prop, value in results["properties"].items():
            if prop == "total_energy_ev":
                summary += f"Total Energy: {value:.6f} eV\n"
            elif prop == "converged":
                summary += f"Converged: {value}\n"
            elif prop.endswith("_available"):
                name = prop.replace("_available", "").title()
                summary += f"{name}: {'Yes' if value else 'No'}\n"

        summary += f"\nParsed results saved to: {results_file}"
        return summary

    except Exception as e:
        return f"Error parsing output file: {str(e)}"


@tool
def extract_energy(
    output_data: Dict[str, Any],
    energy_type: str = "total",
    units: str = "eV",
) -> str:
    """Extract total energy from calculation results.

    Args:
        output_data: Parsed output data dictionary
        energy_type: Type of energy to extract ('total', 'formation')
        units: Energy units ('eV', 'Ry', 'hartree')

    Returns:
        Energy value and information
    """
    try:
        if "properties" not in output_data:
            return "Error: No properties found in output data"

        props = output_data["properties"]
        energy_value = None

        if energy_type == "total":
            if units.lower() == "ev" and "total_energy_ev" in props:
                energy_value = props["total_energy_ev"]
            elif units.lower() == "ry" and "total_energy_ry" in props:
                energy_value = props["total_energy_ry"]
            elif "total_energy_ev" in props:
                ev_energy = props["total_energy_ev"]
                if units.lower() == "ry":
                    energy_value = ev_energy / 13.60569301
                elif units.lower() == "hartree":
                    energy_value = ev_energy / 27.2113834
                else:
                    energy_value = ev_energy

        if energy_value is not None:
            return f"Extracted {energy_type} energy: {energy_value:.6f} {units}"
        else:
            available = list(props.keys())
            return f"Could not extract {energy_type} energy in {units}. Available: {available}"

    except Exception as e:
        return f"Error extracting energy: {str(e)}"
