"""
Tool Registry for DFT Agent

Central registry for all DFT tools to ensure consistent access and organization.
"""

from typing import Dict, Any
from langchain_core.tools import BaseTool

# Import all DFT tools
from .ase_tools import (
    generate_kpoint_mesh,
    get_kpath_bandstructure,
    relax_bulk,
    relax_slab,
)
from .dft_calculator import (
    run_dft_calculation,
    optimize_structure_dft,
    relax_slab_dft,
    test_hydrogen_atom,
)
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

# Tool registry mapping tool names to tool objects
TOOL_REGISTRY: Dict[str, BaseTool] = {
    # Structure tools
    "generate_bulk": generate_bulk,
    "create_supercell": create_supercell,
    "generate_slab": generate_slab,
    "add_adsorbate": add_adsorbate,
    "add_vacuum": add_vacuum,
    
    # DFT calculator tools
    "run_dft_calculation": run_dft_calculation,
    "optimize_structure_dft": optimize_structure_dft,
    "relax_slab_dft": relax_slab_dft,
    "test_hydrogen_atom": test_hydrogen_atom,
    
    # ASE tools
    "get_kpath_bandstructure": get_kpath_bandstructure,
    "generate_kpoint_mesh": generate_kpoint_mesh,
    "relax_bulk": relax_bulk,
    "relax_slab": relax_slab,
    
    # Pymatgen tools
    "search_materials_project": search_materials_project,
    "analyze_crystal_structure": analyze_crystal_structure,
    "find_pseudopotentials": find_pseudopotentials,
    "calculate_formation_energy": calculate_formation_energy,
    
    # QE tools
    "generate_qe_input": generate_qe_input,
    "submit_local_job": submit_local_job,
    "check_job_status": check_job_status,
    "read_output_file": read_output_file,
    "extract_energy": extract_energy,
    
    # Convergence tools
    "kpoint_convergence_test": kpoint_convergence_test,
    "cutoff_convergence_test": cutoff_convergence_test,
    "slab_thickness_convergence": slab_thickness_convergence,
    "vacuum_convergence_test": vacuum_convergence_test,
    
    # Database tools
    "create_calculations_database": create_calculations_database,
    "store_calculation": store_calculation,
    "update_calculation_status": update_calculation_status,
    "store_adsorption_energy": store_adsorption_energy,
    "query_calculations": query_calculations,
    "export_results": export_results,
    "search_similar_calculations": search_similar_calculations,
}

# Tool categories for organization
TOOL_CATEGORIES = {
    "structure_generation": [
        "generate_bulk",
        "create_supercell", 
        "generate_slab",
        "add_adsorbate",
        "add_vacuum",
    ],
    "dft_calculations": [
        "run_dft_calculation",
        "optimize_structure_dft",
        "relax_slab_dft",
        "test_hydrogen_atom",
    ],
    "structure_optimization": [
        "relax_bulk",
        "relax_slab",
    ],
    "kpoint_analysis": [
        "generate_kpoint_mesh",
        "get_kpath_bandstructure",
    ],
    "materials_database": [
        "search_materials_project",
        "analyze_crystal_structure",
        "find_pseudopotentials",
        "calculate_formation_energy",
    ],
    "quantum_espresso": [
        "generate_qe_input",
        "submit_local_job",
        "check_job_status",
        "read_output_file",
        "extract_energy",
    ],
    "convergence_testing": [
        "kpoint_convergence_test",
        "cutoff_convergence_test",
        "slab_thickness_convergence",
        "vacuum_convergence_test",
    ],
    "database_management": [
        "create_calculations_database",
        "store_calculation",
        "update_calculation_status",
        "store_adsorption_energy",
        "query_calculations",
        "export_results",
        "search_similar_calculations",
    ],
}

def get_tool_by_name(tool_name: str) -> BaseTool:
    """Get a tool by its name from the registry.
    
    Args:
        tool_name: Name of the tool to retrieve
        
    Returns:
        The tool object
        
    Raises:
        KeyError: If tool name is not found in registry
    """
    if tool_name not in TOOL_REGISTRY:
        available_tools = list(TOOL_REGISTRY.keys())
        raise KeyError(f"Tool '{tool_name}' not found in registry. Available tools: {available_tools}")
    
    return TOOL_REGISTRY[tool_name]

def get_tools_by_category(category: str) -> Dict[str, BaseTool]:
    """Get all tools in a specific category.
    
    Args:
        category: Category name
        
    Returns:
        Dictionary of tool names to tool objects
        
    Raises:
        KeyError: If category is not found
    """
    if category not in TOOL_CATEGORIES:
        available_categories = list(TOOL_CATEGORIES.keys())
        raise KeyError(f"Category '{category}' not found. Available categories: {available_categories}")
    
    tool_names = TOOL_CATEGORIES[category]
    return {name: TOOL_REGISTRY[name] for name in tool_names}

def list_all_tools() -> Dict[str, BaseTool]:
    """Get all tools in the registry.
    
    Returns:
        Dictionary of all tool names to tool objects
    """
    return TOOL_REGISTRY.copy()

def list_categories() -> list[str]:
    """List all available tool categories.
    
    Returns:
        List of category names
    """
    return list(TOOL_CATEGORIES.keys())

def get_tool_info(tool_name: str) -> Dict[str, Any]:
    """Get detailed information about a tool.
    
    Args:
        tool_name: Name of the tool
        
    Returns:
        Dictionary with tool information
    """
    tool = get_tool_by_name(tool_name)
    
    # Find which category this tool belongs to
    categories = []
    for category, tools in TOOL_CATEGORIES.items():
        if tool_name in tools:
            categories.append(category)
    
    return {
        "name": tool_name,
        "description": tool.description,
        "categories": categories,
        "args_schema": tool.args_schema.model_fields if hasattr(tool.args_schema, 'model_fields') else None,
    }
