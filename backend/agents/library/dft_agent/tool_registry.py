from langchain_core.tools import Tool

from backend.agents.dft_tools.pmg_vasp import (
    write_vasp_scf, write_vasp_relax,
    parse_vasp_energy,
)
from backend.agents.dft_tools.pmg_qe import (
    write_qe_scf,
    parse_qe_energy
)
from backend.agents.dft_tools.struct_tools import (
    build_structure, make_supercell,
)
#from backend.agents.dft_tools.neb import neb_setup, neb_run, neb_parse_results
#from backend.agents.dft_tools.hpc import write_hpc_script_vasp, write_hpc_script_qe
from backend.agents.dft_tools.pmg_run import run_local

TOOL_REGISTRY: dict[str, Tool] = {
    "build_structure": build_structure,
    "make_supercell": make_supercell,
    "write_vasp_scf": write_vasp_scf,
    "write_vasp_relax": write_vasp_relax,
    "parse_vasp_energy": parse_vasp_energy,
    "write_qe_scf": write_qe_scf,
    "parse_qe_energy": parse_qe_energy,
    "run_local": run_local,
}
