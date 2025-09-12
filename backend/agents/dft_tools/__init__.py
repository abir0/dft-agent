"""
DFT Tools Package

This package contains tools for DFT calculations including:
- Structure generation and manipulation
- ASE and Pymatgen integration
- Quantum ESPRESSO and VASP interfaces
- Job management and output parsing
"""

# Import all tools from submodules
from .convergence_tools import (
    cutoff_convergence_test,
    kpoint_convergence_test,
    slab_thickness_convergence,
    vacuum_convergence_test,
)
from .database_tools import (
    create_calculations_database,
    export_results,
    query_calculations,
    search_similar_calculations,
    store_adsorption_energy,
    store_calculation,
    update_calculation_status,
)
from .pymatgen_tools import (
    analyze_crystal_structure,
    calculate_formation_energy,
    find_pseudopotentials,
    get_pseudopotential_recommendations,
    search_materials_project,
)
from .qe_tools import (
    check_job_status,
    extract_energy,
    generate_qe_input,
    read_output_file,
    submit_local_job,
)
from .structure_tools import (
    add_adsorbate,
    add_vacuum,
    create_supercell,
    generate_bulk,
    generate_slab,
)
from .tool_registry import (
    TOOL_REGISTRY,
    TOOL_CATEGORIES,
    get_tool_by_name,
    get_tools_by_category,
    list_all_tools,
    list_categories,
    get_tool_info,
)

__all__ = [  # noqa: RUF022
    # Structure tools
    "generate_bulk",
    "create_supercell",
    "generate_slab",
    "add_adsorbate",
    "add_vacuum",
    # Pymatgen tools
    "search_materials_project",
    "analyze_crystal_structure",
    "find_pseudopotentials",
    "get_pseudopotential_recommendations",
    "calculate_formation_energy",
    # QE tools
    "generate_qe_input",
    "submit_local_job",
    "check_job_status",
    "read_output_file",
    "extract_energy",
    # Convergence tools
    "kpoint_convergence_test",
    "cutoff_convergence_test",
    "slab_thickness_convergence",
    "vacuum_convergence_test",
    # Database tools
    "create_calculations_database",
    "store_calculation",
    "update_calculation_status",
    "store_adsorption_energy",
    "query_calculations",
    "export_results",
    "search_similar_calculations",
    # Tool registry
    "TOOL_REGISTRY",
    "TOOL_CATEGORIES",
    "get_tool_by_name",
    "get_tools_by_category",
    "list_all_tools",
    "list_categories",
    "get_tool_info",
]
