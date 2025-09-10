from langchain_core.tools import Tool

# Import all comprehensive DFT tools
from backend.agents.dft_tools import (
    add_adsorbate,
    add_vacuum,
    analyze_crystal_structure,
    calculate_formation_energy,
    check_job_status,
    create_calculations_database,
    create_supercell,
    cutoff_convergence_test,
    export_results,
    extract_energy,
    find_pseudopotentials,
    generate_bulk,
    generate_kpoint_mesh,
    generate_qe_input,
    generate_slab,
    geometry_optimization,
    get_kpath_bandstructure,
    kpoint_convergence_test,
    query_calculations,
    read_output_file,
    relax_bulk,
    relax_slab,
    search_materials_project,
    search_similar_calculations,
    slab_thickness_convergence,
    store_adsorption_energy,
    store_calculation,
    submit_local_job,
    update_calculation_status,
    vacuum_convergence_test,
)

TOOL_REGISTRY: dict[str, Tool] = {
    # Structure generation and manipulation
    "generate_bulk": generate_bulk,
    "create_supercell": create_supercell,
    "generate_slab": generate_slab,
    "add_adsorbate": add_adsorbate,
    "add_vacuum": add_vacuum,
    # ASE integration tools
    "geometry_optimization": geometry_optimization,
    "get_kpath_bandstructure": get_kpath_bandstructure,
    "generate_kpoint_mesh": generate_kpoint_mesh,
    "relax_bulk": relax_bulk,
    "relax_slab": relax_slab,
    # Pymatgen integration tools
    "search_materials_project": search_materials_project,
    "analyze_crystal_structure": analyze_crystal_structure,
    "find_pseudopotentials": find_pseudopotentials,
    "calculate_formation_energy": calculate_formation_energy,
    # Quantum ESPRESSO interface
    "generate_qe_input": generate_qe_input,
    "submit_local_job": submit_local_job,
    "check_job_status": check_job_status,
    "read_output_file": read_output_file,
    "extract_energy": extract_energy,
    # Convergence testing
    "kpoint_convergence_test": kpoint_convergence_test,
    "cutoff_convergence_test": cutoff_convergence_test,
    "slab_thickness_convergence": slab_thickness_convergence,
    "vacuum_convergence_test": vacuum_convergence_test,
    # Database management
    "create_calculations_database": create_calculations_database,
    "store_calculation": store_calculation,
    "update_calculation_status": update_calculation_status,
    "store_adsorption_energy": store_adsorption_energy,
    "query_calculations": query_calculations,
    "export_results": export_results,
    "search_similar_calculations": search_similar_calculations,
}
