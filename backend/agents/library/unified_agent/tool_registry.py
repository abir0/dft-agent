"""
Unified Tool Registry

Merges all tools from chatbot and DFT agent into a single comprehensive registry.
"""

from typing import List

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from langchain_core.tools import BaseTool, tool

# Import all DFT tools
from backend.agents.dft_tools import (
    add_adsorbate,
    add_vacuum,
    analyze_crystal_structure,
    calculate_formation_energy,
    check_job_status,
    # Database and results tools
    create_calculations_database,
    create_supercell,
    cutoff_convergence_test,
    export_results,
    extract_energy,
    find_pseudopotentials,
    # Structure tools
    generate_bulk,
    # Quantum ESPRESSO tools
    generate_qe_input,
    generate_slab,
    # Convergence tools
    kpoint_convergence_test,
    query_calculations,
    read_output_file,
    # Materials Project and analysis tools
    search_materials_project,
    search_similar_calculations,
    slab_thickness_convergence,
    store_adsorption_energy,
    store_calculation,
    submit_local_job,
    update_calculation_status,
    vacuum_convergence_test,
)

# Import tools from the tools module
from backend.agents.tools import calculator, python_repl


# Define web search tool (from chatbot)
@tool
def web_search(query: str) -> str:
    """Search the web using DuckDuckGo. Use this tool to find current information or answer questions that require up-to-date data."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
            if not results:
                return f"No results found for query: {query}"

            formatted_results = []
            for result in results:
                title = result.get("title", "No title")
                body = result.get("body", "No description")
                href = result.get("href", "")
                formatted_results.append(f"- {title}: {body}\n  URL: {href}")

            return "\n\n".join(formatted_results)
    except Exception as e:
        return f"Error searching the web: {str(e)}"


# Define literature search tool (from chatbot)
@tool
def search_literature(query: str) -> str:
    """Search for academic papers and literature on Google Scholar."""
    try:
        # Use Google Scholar search
        url = f"https://scholar.google.com/scholar?q={query}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return f"Failed to search Google Scholar. Status code: {response.status_code}"

        soup = BeautifulSoup(response.text, "html.parser")

        # Find paper results
        results = []
        for item in soup.select(".gs_ri")[:5]:  # Get top 5 results
            title_elem = item.select_one(".gs_rt a")
            if not title_elem:
                continue

            title = title_elem.get_text()
            link = title_elem.get("href", "")

            # Get authors and publication info
            authors_elem = item.select_one(".gs_a")
            authors_info = authors_elem.get_text() if authors_elem else "No author info"

            # Get snippet
            snippet_elem = item.select_one(".gs_rs")
            snippet = snippet_elem.get_text() if snippet_elem else "No abstract available"

            results.append(
                f"Title: {title}\n"
                f"Authors/Info: {authors_info}\n"
                f"Abstract: {snippet[:200]}...\n"
                f"URL: {link}"
            )

        if not results:
            return f"No literature found for query: {query}"

        return "\n\n---\n\n".join(results)
    except Exception as e:
        return f"Error searching literature: {str(e)}"


def get_unified_tool_registry() -> List[BaseTool]:
    """
    Get the complete unified tool registry.
    
    Returns:
        List of all available tools from both chatbot and DFT agent.
    """

    # General tools (from chatbot)
    general_tools = [
        web_search,
        calculator,
        python_repl,
        search_literature,
    ]

    # Structure generation tools
    structure_tools = [
        generate_bulk,
        create_supercell,
        generate_slab,
        add_adsorbate,
        add_vacuum,
    ]

    # Quantum ESPRESSO tools
    qe_tools = [
        generate_qe_input,
        submit_local_job,
        check_job_status,
        read_output_file,
        extract_energy,
    ]

    # Convergence testing tools
    convergence_tools = [
        kpoint_convergence_test,
        cutoff_convergence_test,
        slab_thickness_convergence,
        vacuum_convergence_test,
    ]

    # Materials database and analysis tools
    materials_tools = [
        search_materials_project,
        analyze_crystal_structure,
        find_pseudopotentials,
        calculate_formation_energy,
    ]

    # Database and results management tools
    database_tools = [
        create_calculations_database,
        store_calculation,
        update_calculation_status,
        store_adsorption_energy,
        query_calculations,
        export_results,
        search_similar_calculations,
    ]

    # Combine all tools into single registry
    all_tools = (
        general_tools +
        structure_tools +
        qe_tools +
        convergence_tools +
        materials_tools +
        database_tools
    )

    return all_tools


# Tool metadata for categorization (optional, for future use)
TOOL_CATEGORIES = {
    "general": ["web_search", "calculator", "python_repl", "search_literature"],
    "structure": ["generate_bulk", "create_supercell", "generate_slab",
                  "add_adsorbate", "add_vacuum"],
    "quantum_espresso": ["generate_qe_input", "submit_local_job", "check_job_status",
                         "read_output_file", "extract_energy"],
    "convergence": ["kpoint_convergence_test", "cutoff_convergence_test",
                    "slab_thickness_convergence", "vacuum_convergence_test"],
    "materials": ["search_materials_project", "analyze_crystal_structure",
                  "find_pseudopotentials", "calculate_formation_energy"],
    "database": ["create_calculations_database", "store_calculation",
                 "update_calculation_status", "store_adsorption_energy",
                 "query_calculations", "export_results", "search_similar_calculations"],
}
