import json
import os
import subprocess
from typing import Any, Dict, Optional

import paramiko
import seekpath
from ase import Atoms
from ase.io import read as ase_read
from langchain_core.tools import tool
from pymatgen.core import Structure
from pymatgen.ext.matproj import MPRester
from pymatgen.io.cif import CifParser
from pymatgen.io.qe import PWInput


# Materials Project lookup
@tool("materials_project_lookup", return_direct=True)
def materials_project_lookup(material_id: str) -> str:
    """
    Query Materials Project for structure and basic properties.
    Args:
        material_id (str): MP id (e.g. 'mp-149') or formula.
        api_key (str): API key if not set in environment.
    Returns:
        JSON string with 'structure' (pymatgen dict) and 'properties'.
    """
    try:
        api_key = os.getenv("MP_API_KEY")
        with MPRester(api_key) as mpr:
            struct = mpr.get_structure_by_material_id(material_id)
            props = mpr.query(
                material_id, ["band_gap", "density", "formation_energy_per_atom"]
            )[0]
        return json.dumps({"structure": struct.as_dict(), "properties": props})
    except Exception as e:
        raise RuntimeError(f"MP lookup failed: {e}")


# Generate structure from CIF or prototype
@tool("generate_structure_tool", return_direct=True)
def generate_structure_tool(cif_or_proto: str) -> str:
    """
    Generate or parse a structure.
    If input is a CIF filepath, parse CIF file. Otherwise, treat as prototype string.
    Returns CIF-format text.
    """
    try:
        if cif_or_proto.lower().endswith(".cif") and os.path.isfile(cif_or_proto):
            parser = CifParser(cif_or_proto)
            struct = parser.get_structures()[0]
        else:
            # simplistic: treat as spacegroup symbol and standardized coords
            struct = Structure.from_spacegroup(
                cif_or_proto, [[0, 0, 0], [1, 1, 0], [1, 0, 1]]
            )
        return struct.to(fmt="cif")
    except Exception as e:
        raise RuntimeError(f"Structure generation failed: {e}")


# Convergence testing
@tool("convergence_test_tool", return_direct=True)
def convergence_test_tool(base_qe_input: str) -> str:
    """
    Vary ecutwfc and kpoints to test convergence.
    Returns recommended parameters as JSON.
    """
    try:
        # stub: in practice would run multiple SCF jobs
        recommended = {"ecutwfc": 50, "ecutrho": 400, "kpoints": [6, 6, 6]}
        return json.dumps(recommended)
    except Exception as e:
        raise RuntimeError(f"Convergence test failed: {e}")


# Geometry optimization
@tool("optimize_geometry_tool", return_direct=True)
def optimize_geometry_tool(qe_input_text: str) -> str:
    """
    Run 'pw.x' in vc-relax mode and return optimized structure as JSON.
    """
    try:
        with open("opt.in", "w") as f:
            f.write(qe_input_text.replace("calculation='scf'", "calculation='vc-relax'"))
        subprocess.run(["pw.x", "-in", "opt.in"], check=True)
        struct = (
            ase_read("opt.in.save/aiida-relaxed.cif")
            if os.path.isdir("opt.in.save")
            else ase_read("opt.in")
        )
        return json.dumps(struct.to_dict())
    except Exception as e:
        raise RuntimeError(f"Geometry optimization failed: {e}")


# Band structure calculation
@tool("bands_calc_tool", return_direct=True)
def bands_calc_tool(qe_input_text: str, kpath_json: str) -> str:
    """
    Run NSCF calculation along k-path for band structure.
    Args:
        qe_input_text: base QE input content.
        kpath_json: JSON with 'points' and 'path'.
    """
    try:
        kdata = json.loads(kpath_json)
        pw_input = PWInput.from_string(qe_input_text)
        pw_input.kpoints = {"explicit": kdata["path"]}
        pw_input.control["calculation"] = "bands"
        pw_input.write_file("bands.in")
        subprocess.run(["pw.x", "-in", "bands.in"], check=True)
        return json.dumps({"status": "bands job completed", "kpath": kdata["path"]})
    except Exception as e:
        raise RuntimeError(f"Band structure calc failed: {e}")


# DOS calculation
@tool("dos_calc_tool", return_direct=True)
def dos_calc_tool(qe_input_text: str) -> str:
    """
    Compute total DOS via dos.x.
    """
    try:
        with open("nscf.in", "w") as f:
            f.write(qe_input_text.replace("calculation='scf'", "calculation='nscf'"))
        subprocess.run(["pw.x", "-in", "nscf.in"], check=True)
        subprocess.run("dos.x < dos.in", shell=True, check=True)
        return json.dumps({"status": "DOS completed"})
    except Exception as e:
        raise RuntimeError(f"DOS calc failed: {e}")


# pDOS calculation
@tool("pdos_calc_tool", return_direct=True)
def pdos_calc_tool(qe_input_text: str) -> str:
    """
    Compute projected DOS via projwfc.x.
    """
    try:
        with open("pw.in", "w") as f:
            f.write(qe_input_text)
        subprocess.run("projwfc.x < pw.in", shell=True, check=True)
        return json.dumps({"status": "pDOS completed"})
    except Exception as e:
        raise RuntimeError(f"pDOS calc failed: {e}")


# QE to ASE converter
@tool("qe_to_ase_tool", return_direct=True)
def qe_to_ase_tool(qe_input_text: str) -> Any:
    """
    Parse QE input to ASE Atoms.
    """
    try:
        pw_input = PWInput.from_string(qe_input_text)
        struct = pw_input.structure
        return Atoms(symbols=struct.species, positions=struct.cart_coords.tolist())
    except Exception as e:
        raise RuntimeError(f"QE->ASE conversion failed: {e}")


# Submit jobs backend
@tool("submit_job_tool", return_direct=True)
def submit_job_tool(config: Dict[str, Any]) -> str:
    """
    Submit a job via 'local', 'slurm', or 'ssh'.
    Config keys: mode, input_file, host, user, key, job_name.
    """
    mode = config.get("mode", "local")
    if mode == "local":
        proc = subprocess.Popen(["pw.x", "-in", config["input_file"]])
        return str(proc.pid)
    elif mode == "slurm":
        script = f"""#!/bin/bash
#SBATCH --job-name={config.get("job_name", "dft")}
#SBATCH --output={config.get("output", "slurm.out")}
pw.x < {config["input_file"]} > {config["input_file"]}.log
"""
        with open("job.slurm", "w") as f:
            f.write(script)
        out = subprocess.check_output(["sbatch", "job.slurm"])
        return out.decode().strip()
    elif mode == "ssh":
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(
            config["host"], username=config["user"], key_filename=config.get("key")
        )
        cmd = f"pw.x < {config['input_file']} > {config['input_file']}.log & echo $!"
        stdin, stdout, stderr = ssh.exec_command(cmd)
        return stdout.read().decode().strip()
    else:
        raise ValueError(f"Unknown mode {mode}")


# SeeK-path tool
@tool("get_k_path", return_direct=True)
def get_k_path(
    structure_json: Optional[dict] = None, cif_file: Optional[str] = None
) -> dict:
    """
    Compute the high-symmetry k-point path for a crystal structure using Seekpath.

    Accepts a pymatgen structure as a JSON dict or a CIF file path. Returns a dictionary
    with the primitive cell, k-point coordinates along the recommended path, and labels.

    Parameters:
        structure_json (dict, optional): JSON-like pymatgen Structure dict.
        cif_file (str, optional): Path to a CIF file.

    Returns:
        dict: A dictionary Seekpath output includes keys such as: 'point_coords', 'path',
        'primitive_lattice', 'primitive_positions', 'primitive_types', 'explicit_kpoints_rel',
        'explicit_kpoints_labels', 'explicit_segments', etc.

    Raises:
        ValueError: If neither `structure_json` nor `cif_file` is provided.
        RuntimeError: If Seekpath fails to compute the k-path for any reason.

    Notes:
    - Use `kpath_to_kpoints` to convert results to Quantum ESPRESSO k-point input.
    - Use `get_band_labels` to get band structure labels.
    """

    try:
        if structure_json is None and cif_file is None:
            raise ValueError("Either structure_json or cif_file must be provided.")

        if cif_file is not None:
            struct = Structure.from_file(cif_file)
        else:
            struct = Structure.from_dict(structure_json)
        cell = struct.lattice.matrix.tolist()
        positions = [site.frac_coords.tolist() for site in struct]
        numbers = [site.specie.number for site in struct]
        tuple_input = (cell, positions, numbers)

        details = seekpath.get_explicit_k_path(tuple_input)

        return details

    except Exception as e:
        raise RuntimeError(f"SeeK-path failed: {e}")


def kpath_to_kpoints(seekpath_data: dict, weight: float = 0.0) -> str:
    """
    Convert seekpath output to Quantum ESPRESSO band structure K_POINTS format.
    """
    kpoints = seekpath_data["explicit_kpoints_rel"]

    lines = []
    lines.append("K_POINTS crystal")
    lines.append(str(len(kpoints)))
    for kpt in kpoints:
        kx, ky, kz = kpt
        lines.append(f"{kx:.10f} {ky:.10f} {kz:.10f} {weight:.1f}")

    return "\n".join(lines)


def get_band_labels(seekpath_data):
    """
    Get band structure labels from seekpath output.
    """
    labels = seekpath_data["explicit_kpoints_labels"]
    linear_coords = seekpath_data["explicit_kpoints_linearcoord"]

    label_points = []
    for x, label in zip(linear_coords, labels, strict=True):
        if label:  # skip empty labels
            label_points.append((x, label.replace("GAMMA", "Γ")))
    return label_points


# Bilbao Crystallographic Server lookup
@tool("bilbao_crystal_tool", return_direct=True)
def bilbao_crystal_tool(prototype: str) -> str:
    """
    Fetch structure CIF from the Bilbao server (stub).
    """
    return f"# CIF for prototype {prototype} (retrieved from Bilbao)\n..."


# 12. QE input generator using pymatgen's PWInput
@tool("qe_input_generator_tool", return_direct=True)
def qe_input_generator_tool(structure_cif: str, params: Dict[str, Any]) -> str:
    """
    Generate Quantum ESPRESSO PW input using pymatgen.io.qe.
    Args:
        structure_cif: CIF-format string
        params: dict of control, system, electrons, kpoints_grid
    """
    try:
        struct = CifParser.from_string(structure_cif).get_structures()[0]
        inp = PWInput(
            structure=struct,
            control=params.get("control", {}),
            system=params.get("system", {}),
            electrons=params.get("electrons", {}),
            kpoints_mode="automatic",
            kpoints_grid=params.get("kpoints_grid", [6, 6, 6]),
        )
        return inp.get_string()
    except Exception as e:
        raise RuntimeError(f"QE input generation failed: {e}")
