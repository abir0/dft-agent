"""
Quantum ESPRESSO Interface Tools

Tools for generating QE input files, submitting jobs, and parsing output.
"""

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

from backend.utils.workspace import get_subdir_path

try:
    from ase.io import read, write
    from ase.io.espresso import read_espresso_out, write_espresso_in

    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False


@tool
def generate_qe_input(
    structure_file: str,
    calculation: str = "scf",
    ecutwfc: float = 30.0,
    ecutrho: Optional[float] = None,
    kpts: Optional[List[int]] = None,
    occupations: str = "smearing",
    smearing: str = "gaussian",
    degauss: float = 0.02,
    pseudopotentials: Optional[Dict[str, str]] = None,
    job_name: str = "pwscf",
    _thread_id: Optional[str] = None,
) -> str:
    """Generate Quantum ESPRESSO input file (pw.x) from structure and parameters.

    Args:
        structure_file: Path to structure file
        calculation: Type of calculation ('scf', 'relax', 'bands', 'nscf', 'vc-relax')
        ecutwfc: Kinetic energy cutoff in Ry
        ecutrho: Charge density cutoff in Ry (default: 4*ecutwfc)
        kpts: K-point mesh (nx, ny, nz)
        occupations: Occupation method ('smearing', 'fixed', 'tetrahedra')
        smearing: Smearing type ('gaussian', 'methfessel-paxton', 'fermi-dirac')
        degauss: Smearing width in Ry
        pseudopotentials: Dict mapping elements to PP files
        job_name: Name for the calculation

    Returns:
        Path to generated input file and job info
    """
    try:
        if not ASE_AVAILABLE:
            return "Error: ASE not available. Please install with: pip install ase"

        # Set defaults
        if kpts is None:
            kpts = [6, 6, 6]

        # Read structure
        atoms = read(structure_file)

        # Get unique elements for pseudopotentials
        elements = list(set(atoms.get_chemical_symbols()))

        # Set default pseudopotentials if not provided
        if pseudopotentials is None:
            pseudopotentials = {}
            for el in elements:
                # Default PSL pseudopotentials
                if el in ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne"]:
                    pseudopotentials[el] = f"{el}.pbe-rrkjus_psl.1.0.0.UPF"
                else:
                    pseudopotentials[el] = f"{el}.pbe-spn-rrkjus_psl.1.0.0.UPF"

        # Set ecutrho default
        if ecutrho is None:
            ecutrho = 4.0 * ecutwfc

        # QE input parameters
        input_data = {
            "control": {
                "calculation": calculation,
                "restart_mode": "from_scratch",
                "prefix": job_name,
                "outdir": "./tmp",
                "pseudo_dir": "./pseudos",
                "verbosity": "high",
            },
            "system": {
                "ecutwfc": ecutwfc,
                "ecutrho": ecutrho,
                "occupations": occupations,
            },
            "electrons": {
                "conv_thr": 1e-8,
                "mixing_beta": 0.7,
                "mixing_mode": "plain",
                "diagonalization": "david",
            },
        }

        # Add smearing parameters if needed
        if occupations == "smearing":
            input_data["system"]["smearing"] = smearing
            input_data["system"]["degauss"] = degauss

        # Add relaxation parameters if needed
        if calculation in ["relax", "vc-relax"]:
            input_data["ions"] = {"ion_dynamics": "bfgs"}

        if calculation == "vc-relax":
            input_data["cell"] = {"cell_dynamics": "bfgs"}

        # Generate output directory and filename
        input_path = Path(structure_file)

        # Use workspace-specific directory if thread_id is available
        if _thread_id:
            output_dir = get_subdir_path(_thread_id, "calculations/qe_inputs")
        else:
            # Fallback to input file's parent directory
            output_dir = input_path.parent / "qe_inputs"
            output_dir.mkdir(exist_ok=True)

        input_filename = f"{job_name}_{calculation}.pwi"
        input_filepath = output_dir / input_filename

        # Write input file using ASE
        write_espresso_in(
            str(input_filepath),
            atoms,
            input_data=input_data,
            pseudopotentials=pseudopotentials,
            kpts=kpts,
        )

        # Create job metadata
        job_info = {
            "structure_file": structure_file,
            "input_file": str(input_filepath),
            "job_name": job_name,
            "calculation": calculation,
            "ecutwfc": ecutwfc,
            "ecutrho": ecutrho,
            "kpoints": kpts,
            "occupations": occupations,
            "smearing": smearing,
            "degauss": degauss,
            "pseudopotentials": pseudopotentials,
            "elements": elements,
        }

        metadata_file = input_filepath.with_suffix(".json")
        with open(metadata_file, "w") as f:
            json.dump(job_info, f, indent=2)

        # Also create a simple run script
        script_file = output_dir / f"run_{job_name}_{calculation}.sh"
        with open(script_file, "w") as f:
            f.write("#!/bin/bash\\n")
            f.write(f"# QE {calculation} calculation for {job_name}\\n")
            f.write(f"pw.x < {input_filename} > {job_name}_{calculation}.pwo\\n")

        script_file.chmod(0o755)  # Make executable

        return (
            f"Generated QE {calculation} input: {input_filepath}\\n"
            f"K-points: {kpts[0]}x{kpts[1]}x{kpts[2]}\\n"
            f"Cutoff: {ecutwfc} Ry (charge: {ecutrho} Ry)\\n"
            f"Elements: {', '.join(elements)}\\n"
            f"Run script: {script_file}\\n"
            f"Metadata: {metadata_file}"
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
    """Submit calculation to local machine.

    Args:
        input_files: Dict of calculation_type -> input_file_path
        executable: QE executable name
        num_cores: Number of CPU cores to use
        memory_limit: Memory limit for the job

    Returns:
        Job submission information and process IDs
    """
    try:
        job_results = {}

        for calc_type, input_file in input_files.items():
            input_path = Path(input_file)
            output_file = input_path.with_suffix(".pwo")

            # Prepare command
            if num_cores > 1:
                cmd = (
                    f"mpirun -np {num_cores} {executable} < {input_file} > {output_file}"
                )
            else:
                cmd = f"{executable} < {input_file} > {output_file}"

            # Create job directory
            job_dir = input_path.parent / "jobs"
            job_dir.mkdir(exist_ok=True)

            # Create job script
            job_script = job_dir / f"job_{calc_type}_{input_path.stem}.sh"
            with open(job_script, "w") as f:
                f.write("#!/bin/bash\\n")
                f.write(f"# Local QE job for {calc_type}\\n")
                f.write(f"cd {input_path.parent}\\n")
                f.write(f"echo 'Starting {calc_type} calculation at $(date)'\\n")
                f.write(f"{cmd}\\n")
                f.write(f"echo 'Finished {calc_type} calculation at $(date)'\\n")

            job_script.chmod(0o755)

            # Submit job (run in background)
            try:
                process = subprocess.Popen(
                    ["bash", str(job_script)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=input_path.parent,
                )

                job_results[calc_type] = {
                    "input_file": input_file,
                    "output_file": str(output_file),
                    "job_script": str(job_script),
                    "process_id": process.pid,
                    "status": "submitted",
                    "command": cmd,
                }

            except Exception as e:
                job_results[calc_type] = {
                    "input_file": input_file,
                    "error": str(e),
                    "status": "failed_to_submit",
                }

        # Save job information
        jobs_file = job_dir / "job_status.json"
        with open(jobs_file, "w") as f:
            json.dump(job_results, f, indent=2)

        # Create summary
        summary = f"Submitted {len(input_files)} local jobs:\\n"
        for calc_type, result in job_results.items():
            if "process_id" in result:
                summary += (
                    f"- {calc_type}: PID {result['process_id']} ({result['status']})\\n"
                )
            else:
                summary += (
                    f"- {calc_type}: {result['status']} - {result.get('error', '')}\\n"
                )

        summary += f"\\nJob status saved to: {jobs_file}"

        return summary

    except Exception as e:
        return f"Error submitting local job: {str(e)}"


@tool
def check_job_status(
    job_id: str, queue_system: str = "local", remote_host: Optional[str] = None
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
        if queue_system.lower() == "local":
            # Check if process is still running
            try:
                pid = int(job_id)
                result = subprocess.run(
                    ["ps", "-p", str(pid)], capture_output=True, text=True
                )

                if result.returncode == 0:
                    return f"Job {job_id} is still running (PID: {pid})"
                else:
                    return f"Job {job_id} has finished or does not exist"

            except ValueError:
                return f"Invalid job ID for local system: {job_id}"

        elif queue_system.lower() == "slurm":
            # Check SLURM job status
            cmd = ["squeue", "-j", job_id, "-h", "-o", "%T"]
            if remote_host:
                cmd = ["ssh", remote_host] + cmd

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                status = result.stdout.strip()
                return f"SLURM job {job_id} status: {status}"
            else:
                return f"SLURM job {job_id} not found or completed"

        elif queue_system.lower() == "pbs":
            # Check PBS job status
            cmd = ["qstat", job_id]
            if remote_host:
                cmd = ["ssh", remote_host] + cmd

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                return f"PBS job {job_id} status:\\n{result.stdout}"
            else:
                return f"PBS job {job_id} not found or completed"

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

        results = {"output_file": output_file, "code_type": code_type, "properties": {}}

        if code_type.lower() == "qe":
            # Parse QE output
            with open(output_file, "r") as f:
                content = f.read()

            # Extract total energy
            if "energy" in extract_properties:
                energy_lines = [
                    line
                    for line in content.split("\\n")
                    if "!" in line and "total energy" in line
                ]
                if energy_lines:
                    # Get the last energy value
                    last_energy_line = energy_lines[-1]
                    try:
                        energy_val = float(
                            last_energy_line.split("=")[1].split("Ry")[0].strip()
                        )
                        results["properties"]["total_energy_ry"] = energy_val
                        results["properties"]["total_energy_ev"] = (
                            energy_val * 13.60569301
                        )  # Ry to eV
                    except (IndexError, ValueError):
                        results["properties"]["energy_error"] = "Could not parse energy"

            # Check convergence
            if "convergence" in extract_properties:
                if "convergence achieved" in content.lower():
                    results["properties"]["converged"] = True
                elif "convergence NOT achieved" in content.lower():
                    results["properties"]["converged"] = False
                else:
                    results["properties"]["converged"] = "unknown"

            # Extract forces (simplified)
            if "forces" in extract_properties:
                force_lines = [
                    line
                    for line in content.split("\\n")
                    if "Forces acting on atoms" in line
                ]
                if force_lines:
                    results["properties"]["forces_available"] = True
                else:
                    results["properties"]["forces_available"] = False

            # Extract stress (simplified)
            if "stress" in extract_properties:
                stress_lines = [
                    line for line in content.split("\\n") if "total   stress" in line
                ]
                if stress_lines:
                    results["properties"]["stress_available"] = True
                else:
                    results["properties"]["stress_available"] = False

        # Save parsed results
        results_file = output_path.with_suffix(".parsed.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        # Create summary
        summary = f"Parsed {code_type.upper()} output: {output_file}\\n"

        for prop, value in results["properties"].items():
            if prop == "total_energy_ev":
                summary += f"Total Energy: {value:.6f} eV\\n"
            elif prop == "converged":
                summary += f"Converged: {value}\\n"
            elif prop.endswith("_available"):
                prop_name = prop.replace("_available", "").title()
                summary += f"{prop_name}: {'Yes' if value else 'No'}\\n"

        summary += f"\\nParsed results saved to: {results_file}"

        return summary

    except Exception as e:
        return f"Error parsing output file: {str(e)}"


@tool
def extract_energy(
    output_data: Dict[str, Any], energy_type: str = "total", units: str = "eV"
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

        # Look for energy values
        energy_value = None
        if energy_type == "total":
            if units.lower() == "ev" and "total_energy_ev" in props:
                energy_value = props["total_energy_ev"]
            elif units.lower() == "ry" and "total_energy_ry" in props:
                energy_value = props["total_energy_ry"]
            elif "total_energy_ev" in props:
                # Convert if needed
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
