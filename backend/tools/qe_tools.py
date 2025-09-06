# backend/tools/qe_tools.py
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    # Preferred decorator when LangChain is present
    from langchain_core.tools import tool
except Exception:  # pragma: no cover
    # Lightweight fallback so functions remain callable without LangChain
    def tool(func=None, *, return_direct: bool = False):
        def decorator(f):
            return f

        return decorator if func is None else decorator(func)


try:
    from ase import io as aseio
    from ase.build import surface as ase_surface, add_adsorbate
    from ase.lattice.cubic import FaceCenteredCubic, BodyCenteredCubic, SimpleCubic
except Exception as e:
    raise RuntimeError("ASE is required. Install with `pip install ase`.") from e


@dataclass
class QEParams:
    """Minimal QE control/ELECTRONS/SYSTEM parameters we may tweak."""

    control: Dict[str, Any]
    system: Dict[str, Any]
    electrons: Dict[str, Any]


@dataclass
class JobSpec:
    structure_path: str
    workdir: str
    pseudo_dir: str
    kmesh: Tuple[int, int, int] = (3, 3, 1)
    params: Optional[QEParams] = None
    code_command: str = "pw.x"  # if running locally; include mpirun as needed
    input_filename: str = "qe.pwi"
    output_filename: str = "qe.pwo"
    run_mode: str = "local"  # "local" or "aiida"
    max_minutes: int = 120


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _default_params() -> QEParams:
    # Sensible defaults for slab SCF
    return QEParams(
        control={
            "calculation": "scf",
            "verbosity": "low",
            "pseudo_dir": "./pseudos",
            "outdir": "./tmp",
            "tstress": True,
            "tprnfor": True,
        },
        system={
            "ecutwfc": 50.0,
            "ecutrho": 400.0,
            "occupations": "smearing",
            "smearing": "mp",
            "degauss": 0.02,
            "ibrav": 0,  # we will write CELL_PARAMETERS explicitly (ASE)
        },
        electrons={
            "diagonalization": "davidson",
            "conv_thr": 1e-7,
            "mixing_beta": 0.5,
            "electron_maxstep": 200,
        },
    )


# build bulk structure using ase.lattice.cubic
@tool
def build_bulk_structure(
    element: str,
    lattice_constant: float = 3.5,
    to_path: str = "bulk.traj",
) -> str:
    """
    Build a bulk cubic structure using ASE and save to `to_path`.
    Returns the path to the saved bulk structure.
    """
    element = element.capitalize()
    if element in ["Cu", "Ag", "Au", "Ni", "Pd", "Pt"]:
        bulk = FaceCenteredCubic(symbol=element, latticeconstant=lattice_constant)
    elif element in ["Fe", "Cr", "W"]:
        bulk = BodyCenteredCubic(symbol=element, latticeconstant=lattice_constant)
    else:
        bulk = SimpleCubic(symbol=element, latticeconstant=lattice_constant)

    aseio.write(to_path, bulk)
    return to_path


@tool
def build_slab_from_bulk(
    bulk_path: str,
    miller_hkl: Tuple[int, int, int] = (1, 1, 1),
    layers: int = 4,
    vacuum: float = 10.0,
    to_path: str = "slab.traj",
) -> str:
    """
    Build a surface slab from a bulk structure (CIF/POSCAR/… via ASE) and save to `to_path`.
    Returns the path to the saved slab.
    """
    bulk = aseio.read(bulk_path)
    h, k, l = miller_hkl
    slab = ase_surface(bulk, (h, k, l), layers, vacuum=vacuum)
    aseio.write(to_path, slab)
    return to_path


@tool
def add_adsorbate_to_slab(
    slab_path: str,
    adsorbate_path: str,
    height: float = 2.0,
    position: Tuple[float, float] = (0.5, 0.5),
    to_path: str = "ads_slab.traj",
) -> str:
    """
    Place an adsorbate onto a slab and save combined structure to `to_path`.
    """
    slab = aseio.read(slab_path)
    ads = aseio.read(adsorbate_path)
    add_adsorbate(slab, ads, height=height, position=position)
    aseio.write(to_path, slab)
    return to_path


def _write_qe_input_with_ase(
    atoms_path: str,
    job: JobSpec,
) -> str:
    """
    Write QE input using ASE I/O ('espresso-in') and return input filename.
    Note: ASE supports 'espresso-in' and will write CONTROL/SYSTEM/ELECTRONS and cards.
    """
    inpath = Path(job.workdir) / job.input_filename
    pseudo_dir_rel = Path(job.workdir) / "pseudos"
    _ensure_dir(Path(job.workdir))
    _ensure_dir(pseudo_dir_rel)
    _ensure_dir(Path(job.workdir) / "tmp")

    params = job.params or _default_params()
    # Ensure pseudo_dir is consistent with CONTROL card (ASE respects it)
    params.control["pseudo_dir"] = str(pseudo_dir_rel)

    # Build keyword dictionaries for ASE writer
    kw = {
        "input_data": {
            "control": params.control,
            "system": params.system,
            "electrons": params.electrons,
        },
        "kpts": list(job.kmesh),
        "pseudopotentials": {},  # Let QE find from pseudo_dir by element name
    }

    atoms = aseio.read(atoms_path)
    aseio.write(
        str(inpath), atoms, format="espresso-in", **kw
    )  # QE input writer in ASE
    return str(inpath)


def _run_local(job: JobSpec) -> Tuple[int, str]:
    """
    Run QE locally using subprocess. Returns (returncode, output_path).
    Will stream stdout to qe.pwo.
    """
    inp = Path(job.workdir) / job.input_filename
    outp = Path(job.workdir) / job.output_filename
    cmd = f"{job.code_command} -in {inp.name}"
    with open(outp, "w") as fh:
        proc = subprocess.Popen(
            cmd, cwd=job.workdir, shell=True, stdout=fh, stderr=subprocess.STDOUT
        )
        try:
            # crude watchdog by wall time
            t0 = time.time()
            while proc.poll() is None:
                if (time.time() - t0) / 60 > job.max_minutes:
                    proc.terminate()
                    return 124, str(outp)
                time.sleep(2.0)
        except KeyboardInterrupt:
            proc.terminate()
    return proc.returncode or 0, str(outp)


def _parse_scf(out_text: str) -> Dict[str, Any]:
    """
    Parse minimal SCF results and convergence flags.
    Looks for total energy lines and final convergence marker.
    """
    data: Dict[str, Any] = {
        "converged": False,
        "etot": None,
        "nbands": None,
        "warnings": [],
    }
    # QE "!" total energy line
    m = re.search(r"!\s*total energy\s*=\s*([-\d\.Ee+]+)\s*Ry", out_text)
    if m:
        data["etot"] = float(m.group(1))
    # typical convergence phrase
    if re.search(r"convergence has been achieved", out_text, re.I):
        data["converged"] = True
    # bands info (optional)
    m2 = re.search(r"number of Kohn-Sham states=\s*(\d+)", out_text)
    if m2:
        data["nbands"] = int(m2.group(1))
    # collect known warnings/errors
    for pat in [
        r"convergence NOT achieved",
        r"too many bands are not converged",
        r"Exit code:\s*\d+",
        r"error",
        r"WARNING",
    ]:
        for mm in re.finditer(pat, out_text, re.I):
            data["warnings"].append(mm.group(0))
    return data


def _tune_params_on_failure(params: QEParams, out_text: str) -> QEParams:
    """
    Heuristic auto-repair for common QE SCF failures.
    """
    txt = out_text.lower()
    new = QEParams(
        control=dict(params.control),
        system=dict(params.system),
        electrons=dict(params.electrons),
    )
    if "convergence not achieved" in txt or "error" in txt:
        # More mixing damping & more steps
        beta = float(new.electrons.get("mixing_beta", 0.5))
        new.electrons["mixing_beta"] = max(0.1, beta * 0.6)
        new.electrons["electron_maxstep"] = max(
            int(new.electrons.get("electron_maxstep", 100)) * 2, 200
        )
        # Try switching diagonalization if stuck
        diag = (new.electrons.get("diagonalization") or "davidson").lower()
        new.electrons["diagonalization"] = "cg" if diag == "davidson" else "davidson"
        # Slightly increase cutoff if needed
        ecutwfc = float(new.system.get("ecutwfc", 50.0))
        ecutrho = float(new.system.get("ecutrho", 8 * ecutwfc))
        new.system["ecutwfc"] = ecutwfc * 1.2
        new.system["ecutrho"] = ecutrho * 1.2
    return new


@tool
def prepare_qe_inputs(
    structure_path: str,
    workdir: str,
    pseudo_dir: str,
    kmesh: Tuple[int, int, int] = (3, 3, 1),
    params_json: Optional[str] = None,
    code_command: str = "pw.x",
    input_filename: str = "qe.pwi",
    output_filename: str = "qe.pwo",
) -> Dict[str, Any]:
    """
    Create QE input files into `workdir` (and copy pseudopotentials folder if provided).
    Returns a dict describing the job.
    """
    job = JobSpec(
        structure_path=structure_path,
        workdir=workdir,
        pseudo_dir=pseudo_dir,
        kmesh=kmesh,
        params=(
            _default_params()
            if not params_json
            else QEParams(**json.loads(params_json))
        ),
        code_command=code_command,
        input_filename=input_filename,
        output_filename=output_filename,
    )
    _ensure_dir(Path(workdir))
    if pseudo_dir:
        dest = Path(workdir) / "pseudos"
        _ensure_dir(dest)
        # Copy pseudo files (flat) if the source is a dir
        src = Path(pseudo_dir)
        if src.exists() and src.is_dir():
            for f in src.iterdir():
                if f.is_file():
                    shutil.copy2(f, dest / f.name)
    inpath = _write_qe_input_with_ase(structure_path, job)
    return {
        "job": asdict(job),
        "input_path": inpath,
    }


@tool
def run_qe_local(job_json: str) -> Dict[str, Any]:
    """
    Run QE locally. `job_json` is a JSON-serialized JobSpec; returns status and parsed outputs.
    """
    job = JobSpec(**json.loads(job_json))
    rc, outpath = _run_local(job)
    out_text = (
        Path(outpath).read_text(errors="ignore") if Path(outpath).exists() else ""
    )
    parsed = _parse_scf(out_text)
    return {
        "returncode": rc,
        "output_path": outpath,
        "parsed": parsed,
        "stdout_tail": out_text[-2000:],
    }


@tool
def suggest_qe_retry_params(params_json: str, stdout_tail: str) -> str:
    """
    Given prior QE params and a piece of stdout, return new QEParams JSON with safer settings.
    """
    params = QEParams(**json.loads(params_json))
    new_params = _tune_params_on_failure(params, stdout_tail)
    return json.dumps(asdict(new_params))


# ---- AiiDA submission (optional) ----------------------------------------------------


def _aiida_available() -> bool:
    try:
        import aiida  # noqa: F401

        return True
    except Exception:
        return False


@tool
def run_qe_aiida(job_json: str) -> Dict[str, Any]:
    """
    Submit QE via AiiDA if available; otherwise returns a stub response.
    Uses aiida-quantumespresso PwCalculation or PwBaseWorkChain as appropriate.
    """
    job = JobSpec(**json.loads(job_json))
    if not _aiida_available():
        return {
            "submitted": False,
            "reason": "AiiDA not available in this environment.",
        }

    # Minimal example closely follows AiiDA QE tutorial.
    # Users must have configured codes, computer, and pseudopotentials in the AiiDA profile.
    # See: https://aiida-quantumespresso.readthedocs.io/ and AiiDA QE tutorials.
    from aiida.engine import submit
    from aiida.orm import Code, Dict, StructureData, KpointsData, FolderData
    from aiida_quantumespresso.calculations.pw import PwCalculation

    struct_atoms = aseio.read(job.structure_path)
    structure = StructureData(ase=struct_atoms)

    kpoints = KpointsData()
    kpoints.set_kpoints_mesh(job.kmesh)

    inputs = {
        "code": Code.get_from_string(
            "qe-pw@localhost"
        ),  # user must configure this label
        "structure": structure,
        "kpoints": kpoints,
        "parameters": Dict(
            dict={
                "CONTROL": (
                    job.params.control if job.params else _default_params().control
                ),
                "SYSTEM": job.params.system if job.params else _default_params().system,
                "ELECTRONS": (
                    job.params.electrons if job.params else _default_params().electrons
                ),
            }
        ),
        "metadata": {"label": "dft-agent-pw"},
    }
    # Optional pseudopotentials and remote resources can be attached here.
    # For a full setup, see AiiDA QE tutorial docs.

    node = submit(PwCalculation, **inputs)
    return {"submitted": True, "node_uuid": str(node.uuid), "pk": node.pk}
