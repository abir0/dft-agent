"""
Pymatgen-based Tools for Materials Analysis

Tools for materials database integration, crystal analysis, and electronic structure
analysis using Pymatgen and Materials Project API.
"""

import gzip
import json
import shutil
import tarfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import requests
from ase.io import read
from langchain_core.tools import tool
from mp_api.client import MPRester
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core import Structure, Molecule
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from backend.settings import settings

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:142.0) Gecko/20100101 Firefox/142.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


# Load parsed pseudopotential data
def load_pseudopotential_metadata():
    """Load comprehensive pseudopotential database with fallback for missing file."""
    import os
    from pathlib import Path
    
    # Try multiple possible locations for the database file
    possible_paths = [
        "data/inputs/pseudopotentials/pp_mapping_pslibrary.json",
        Path(__file__).parent.parent.parent.parent / "data" / "inputs" / "pseudopotentials" / "pp_mapping_pslibrary.json",
        Path.cwd() / "data" / "inputs" / "pseudopotentials" / "pp_mapping_pslibrary.json",
        "data/inputs/pseudopotentials/pslibrary_database.json",
        Path(__file__).parent.parent.parent.parent / "data" / "inputs" / "pseudopotentials" / "pslibrary_database.json",
        Path.cwd() / "data" / "inputs" / "pseudopotentials" / "pslibrary_database.json",
        "data/pseudos_metadata.json",
        Path(__file__).parent.parent.parent.parent / "data" / "pseudos_metadata.json",
        Path.cwd() / "data" / "pseudos_metadata.json",
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Silently load pseudopotential database
                    return data
            except Exception as e:
                print(f"Warning: Could not load pseudopotential database from {path}: {e}")
                continue
    
    # Return empty metadata if file not found - let the system handle missing elements dynamically
    return {}

def get_pseudopotential_for_element(element: str, xc_functional: str = "PBE", relativistic: bool = None) -> dict:
    """
    Get pseudopotential information for any element dynamically.
    
    Args:
        element: Element symbol (e.g., 'H', 'C', 'Pt', 'U')
        xc_functional: Exchange-correlation functional (default: 'PBE')
        relativistic: Whether to use relativistic pseudopotential (auto-detect if None)
    
    Returns:
        Dictionary with pseudopotential information
    """
    element = element.capitalize()
    xc_functional = xc_functional.lower()
    
    # Only PBE and PBEsol are available in the actual database
    if xc_functional not in ["pbe", "pbesol"]:
        xc_functional = "pbe"  # Default to PBE
    
    # Auto-detect relativistic for heavy elements (Z > 20)
    if relativistic is None:
        atomic_numbers = {
            'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
            'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
            'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
            'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
            'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
            'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,
            'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
            'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
            'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
            'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94
        }
        z = atomic_numbers.get(element, 1)
        relativistic = z > 20
    
    # Generate filename matching the actual database format
    # Different elements have different naming conventions in the actual database
    element_prefixes = {
        'H': '',  # H.pbe-kjpaw_psl.1.0.0.UPF
        'C': '-n',  # C.pbe-n-kjpaw_psl.1.0.0.UPF
        'O': '-n',  # O.pbe-n-kjpaw_psl.1.0.0.UPF
        'Mg': '-spn',  # Mg.pbe-spn-kjpaw_psl.1.0.0.UPF
        'Pt': '-spn',  # Pt.pbe-spn-kjpaw_psl.1.0.0.UPF
    }
    
    prefix = element_prefixes.get(element, '-n')  # Default to -n for most elements
    
    if relativistic:
        filename = f"{element}.rel-{xc_functional}{prefix}-kjpaw_psl.1.0.0.UPF"
    else:
        filename = f"{element}.{xc_functional}{prefix}-kjpaw_psl.1.0.0.UPF"
    
    return {
        "filename": filename,
        "functional": xc_functional,
        "type": "PAW",
        "quality": "high",
        "relativistic": relativistic,
        "cutoff_energy": {
            "ecutwfc": 60.0,
            "ecutrho": 300.0
        },
        "ase_name": filename,
        "source": "generated_pattern"
    }

def get_available_pseudopotentials_for_element(element: str) -> list:
    """
    Get list of available pseudopotentials for any element.
    
    Args:
        element: Element symbol
    
    Returns:
        List of available pseudopotential dictionaries
    """
    element = element.capitalize()
    
    # Only PBE and PBEsol are available in the actual database
    xc_functionals = ["pbe", "pbesol"]
    
    available_pps = []
    
    for xc in xc_functionals:
        # Non-relativistic version
        pp_info = get_pseudopotential_for_element(element, xc, relativistic=False)
        available_pps.append(pp_info)
        
        # Relativistic version (for heavy elements)
        pp_info_rel = get_pseudopotential_for_element(element, xc, relativistic=True)
        available_pps.append(pp_info_rel)
    
    return available_pps

PP_METADATA = load_pseudopotential_metadata()


@tool
def search_materials_project(
    formula: str,
    properties: Optional[List[str]] = None,
    limit: int = 10,
    api_key: Optional[str] = None,
) -> str:
    """Search Materials Project database for materials.

    Args:
        formula: Chemical formula (e.g., 'LiFePO4', 'TiO2', 'Cu')
        properties: List of properties to retrieve
        limit: Maximum number of results
        api_key: Materials Project API key (optional, uses settings if not provided)

    Returns:
        Search results with material properties
    """
    try:
        # Use API key from settings if not provided
        if api_key is None:
            from backend.settings import settings
            if settings.MP_API_KEY:
                api_key = settings.MP_API_KEY.get_secret_value()
            else:
                return "Error: Materials Project API key not configured. Please set MP_API_KEY in environment variables."
        
        if properties is None:
            properties = [
                "material_id",
                "formula_pretty",
                "structure",
                "formation_energy_per_atom",
                "band_gap",
                "density",
            ]

        with MPRester(api_key=api_key) as mpr:
            # Search without fields parameter (API doesn't support it anymore)
            docs = mpr.materials.summary.search(formula=formula)
            # Limit results after getting them
            docs = docs[:limit]

        if not docs:
            return f"No materials found for formula: {formula}"

        # Create output directory
        output_dir = Path("materials_project_data")
        output_dir.mkdir(exist_ok=True)

        results = []
        for i, doc in enumerate(docs):
            # Handle both dict and object responses
            if isinstance(doc, dict):
                material_id = doc.get("material_id", f"unknown_{i}")
                formula = doc.get("formula_pretty", "unknown")
                formation_energy = doc.get("formation_energy_per_atom")
                band_gap = doc.get("band_gap")
                density = doc.get("density")
                symmetry = doc.get("symmetry")
                structure = doc.get("structure")
            else:
                material_id = getattr(doc, "material_id", f"unknown_{i}")
                formula = getattr(doc, "formula_pretty", "unknown")
                formation_energy = getattr(doc, "formation_energy_per_atom", None)
                band_gap = getattr(doc, "band_gap", None)
                density = getattr(doc, "density", None)
                symmetry = getattr(doc, "symmetry", None)
                structure = getattr(doc, "structure", None)

            result = {
                "material_id": material_id,
                "formula": formula,
            }

            # Add available properties
            if formation_energy is not None:
                result["formation_energy_per_atom"] = float(formation_energy)

            if band_gap is not None:
                result["band_gap"] = float(band_gap)

            if density is not None:
                result["density"] = float(density)

            if symmetry:
                if isinstance(symmetry, dict):
                    result["spacegroup"] = symmetry.get("symbol", "unknown")
                    result["spacegroup_number"] = symmetry.get("number", "unknown")
                else:
                    result["spacegroup"] = getattr(symmetry, "symbol", "unknown")
                    result["spacegroup_number"] = getattr(symmetry, "number", "unknown")

            # Save structure if available
            if structure is not None:
                try:
                    structure_file = output_dir / f"{material_id}_{formula}.cif"
                    if hasattr(structure, 'to'):
                        # It's a Structure object
                        structure.to(filename=str(structure_file))
                    else:
                        # It's a dict, try to convert
                        from pymatgen.core import Structure
                        struct_obj = Structure.from_dict(structure)
                        struct_obj.to(filename=str(structure_file))
                    result["structure_file"] = str(structure_file)
                except Exception as e:
                    result["structure_error"] = str(e)

            results.append(result)

        # Save search results
        safe_formula = formula.replace(' ', '_') if formula else 'unknown'
        results_file = output_dir / f"search_{safe_formula}_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        summary = f"Found {len(results)} materials for {formula}:\\n"
        for result in results:
            summary += f"- {result['material_id']}: {result['formula']}"
            if "formation_energy_per_atom" in result:
                summary += f" (ΔHf: {result['formation_energy_per_atom']:.3f} eV/atom)"
            if "band_gap" in result:
                summary += f" (Eg: {result['band_gap']:.2f} eV)"
            summary += "\\n"

        summary += f"\\nResults saved to: {results_file}"
        return summary

    except Exception as e:
        return f"Error searching Materials Project: {str(e)}"


@tool
def analyze_crystal_structure(
    structure_file: str, tolerance: float = 0.01, angle_tolerance: float = 5.0
) -> str:
    """Analyze crystal structure using Pymatgen.

    Args:
        structure_file: Path to structure file
        tolerance: Tolerance for symmetry analysis
        angle_tolerance: Angle tolerance for symmetry analysis

    Returns:
        String with crystal structure analysis
    """
    try:
        # Read structure using ASE then convert to Pymatgen
        atoms = read(structure_file)
        adaptor = AseAtomsAdaptor()
        structure = adaptor.get_structure(atoms)

        # Perform symmetry analysis
        sga = SpacegroupAnalyzer(
            structure, symprec=tolerance, angle_tolerance=angle_tolerance
        )

        # Get crystal system and space group
        crystal_system = sga.get_crystal_system()
        space_group = sga.get_space_group_symbol()
        space_group_number = sga.get_space_group_number()

        # Get conventional structure
        conventional_structure = sga.get_conventional_standard_structure()

        # Analyze local environment
        try:
            cnn = CrystalNN()
            coordination_envs = []
            for i, site in enumerate(structure):
                cn_info = cnn.get_cn(structure, i, use_weights=True)
                coordination_envs.append(
                    {
                        "site_index": i,
                        "element": str(site.specie),
                        "coordination_number": cn_info,
                    }
                )
        except Exception:
            coordination_envs = "Could not analyze coordination environments"

        # Generate output
        input_path = Path(structure_file)
        output_dir = input_path.parent / "analysis"
        output_dir.mkdir(exist_ok=True)

        # Save conventional structure
        conv_file = output_dir / f"{input_path.stem}_conventional.cif"
        conventional_structure.to(filename=str(conv_file))

        # Create analysis report
        analysis = {
            "original_file": structure_file,
            "formula": structure.formula,
            "crystal_system": crystal_system,
            "space_group_symbol": space_group,
            "space_group_number": space_group_number,
            "lattice_parameters": {
                "a": float(structure.lattice.a),
                "b": float(structure.lattice.b),
                "c": float(structure.lattice.c),
                "alpha": float(structure.lattice.alpha),
                "beta": float(structure.lattice.beta),
                "gamma": float(structure.lattice.gamma),
            },
            "volume": float(structure.volume),
            "density": float(structure.density),
            "num_sites": len(structure),
            "coordination_environments": coordination_envs,
            "conventional_structure_file": str(conv_file),
        }

        analysis_file = output_dir / f"{input_path.stem}_analysis.json"
        with open(analysis_file, "w") as f:
            json.dump(analysis, f, indent=2, default=str)

        # Create summary string
        summary = f"Crystal Structure Analysis for {structure.formula}:\\n"
        summary += f"Crystal System: {crystal_system}\\n"
        summary += f"Space Group: {space_group} (#{space_group_number})\\n"
        summary += f"Lattice Parameters: a={structure.lattice.a:.3f}, b={structure.lattice.b:.3f}, c={structure.lattice.c:.3f} Å\\n"
        summary += f"Angles: α={structure.lattice.alpha:.1f}°, β={structure.lattice.beta:.1f}°, γ={structure.lattice.gamma:.1f}°\\n"
        summary += f"Volume: {structure.volume:.2f} Å³\\n"
        summary += f"Density: {structure.density:.2f} g/cm³\\n"
        summary += f"Number of sites: {len(structure)}\\n"
        summary += f"\\nAnalysis saved to: {analysis_file}"
        summary += f"\\nConventional structure saved to: {conv_file}"

        return summary

    except Exception as e:
        return f"Error analyzing crystal structure: {str(e)}"


@tool
def find_pseudopotentials(
    elements: List[str],
    pp_type: str = "PAW",
    pp_library: str = "pslibrary",
    functional: str = "PBE",
    quality: str = "high",
    relativistic: bool = None,
) -> dict:
    """Find, download, and extract pseudopotentials from comprehensive PSLibrary database.

    Args:
        elements: List of chemical elements (symbols, e.g. ["Si", "O"])
        pp_type: Pseudopotential type ("PAW", "US", "ONCV")
        pp_library: Pseudopotential library ("pslibrary", "gbrv", "sg15", "dojo", "SSSP_Efficiency")
        functional: Exchange-correlation functional ("PBE", "PBEsol", "PZ")
        quality: Quality level ("high", "medium", "low")
        relativistic: Whether to use relativistic pseudopotentials (auto-detect for heavy elements if None)

    Returns:
        dict with working_dir, found pseudopotentials (local paths), and parameters
    """
    try:
        output_dir = Path(f"{settings.ROOT_PATH}/WORKSPACE/pseudos")
        output_dir.mkdir(exist_ok=True)

        found_pps = {}
        missing_pps = []
        pp_parameters = {}

        for elem in elements:
            elem = elem.capitalize()
            
            # First try to find in the actual database file
            available_pps = []
            pp_db_path = Path("data/inputs/pseudopotentials/pp_mapping_pslibrary.json")
            if pp_db_path.exists():
                try:
                    with open(pp_db_path, 'r') as f:
                        pp_database = json.load(f)
                    if elem in pp_database:
                        available_pps = pp_database[elem]
                except Exception as e:
                    print(f"Warning: Could not load pseudopotential database: {e}")
            
            # If not found in database, generate dynamically
            if not available_pps:
                available_pps = get_available_pseudopotentials_for_element(elem)
            
            # Find best matching pseudopotential
            best_match = None
            
            for pp in available_pps:
                # Check if this pseudopotential matches our criteria
                if (pp.get("type", "").upper() == pp_type.upper() and
                    pp.get("functional", "").lower() == functional.lower()):
                    
                    # Handle relativistic selection
                    if relativistic is None:
                        # Auto-detect: use relativistic for heavy elements (Z > 20)
                        atomic_numbers = {
                            'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
                            'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
                            'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
                            'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
                            'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
                            'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,
                            'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
                            'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
                            'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
                            'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94
                        }
                        z = atomic_numbers.get(elem, 1)
                        should_use_rel = z > 20
                    else:
                        should_use_rel = relativistic
                    
                    # Check relativistic preference (if specified in pp)
                    if "relativistic" not in pp or pp.get("relativistic", False) == should_use_rel:
                        best_match = pp
                        break
            
            if not best_match:
                missing_pps.append(elem)
                continue
            
            # Download the pseudopotential
            filename = best_match["filename"]  # Use "filename" key from database
            local_file = output_dir / filename
            
            if not local_file.exists():
                # Try to download from PSLibrary
                base_url = "https://github.com/dalcorso/pslibrary/raw/master/upf_files/"
                url = base_url + filename
                
                try:
                    response = requests.get(url, headers=headers, timeout=30)
                    if response.status_code == 200:
                        with open(local_file, "w") as f:
                            f.write(response.text)
                    else:
                        # Fallback: create a placeholder file
                        print(f"Warning: Could not download {filename}, creating placeholder")
                        with open(local_file, "w") as f:
                            f.write(f"# Placeholder for {filename}\n")
                except Exception as e:
                    print(f"Warning: Error downloading {filename}: {e}")
                    # Create placeholder
                    with open(local_file, "w") as f:
                        f.write(f"# Placeholder for {filename}\n")
            
            found_pps[elem] = str(local_file)
            pp_parameters[elem] = {
                "filename": filename,
                "type": best_match.get("type", "PAW"),
                "functional": best_match.get("functional", "pbe"),
                "quality": best_match.get("quality", "high"),
                "relativistic": best_match.get("relativistic", False),
                "cutoff_energy": best_match.get("cutoff_energy", {"ecutwfc": 60.0, "ecutrho": 300.0}),
                "description": best_match.get("description", f"Pseudopotential for {elem}")
            }

        return {
            "working_dir": str(output_dir.resolve()),
            "found_pseudopotentials": found_pps,
            "missing": missing_pps,
            "parameters": pp_parameters,
            "database_info": PP_METADATA.get("metadata", {})
        }

    except Exception as e:
        return {"error": str(e)}


@tool
def get_pseudopotential_recommendations(
    elements: List[str],
    calculation_type: str = "scf",
    functional: str = "PBE",
    quality: str = "high"
) -> str:
    """Get pseudopotential recommendations with optimal parameters for calculations.
    
    Args:
        elements: List of chemical elements
        calculation_type: Type of calculation (scf, relax, vc-relax, etc.)
        functional: Exchange-correlation functional (PBE, PBEsol, PZ)
        quality: Quality level (high, medium, low)
        
    Returns:
        Detailed recommendations with parameters
    """
    try:
        # Get pseudopotentials
        pp_result = find_pseudopotentials.invoke({
            'elements': elements,
            'pp_type': 'PAW',
            'functional': functional,
            'quality': quality
        })
        
        if 'error' in pp_result:
            return f"Error getting pseudopotential recommendations: {pp_result['error']}"
        
        recommendations = f"**Pseudopotential Recommendations for {', '.join(elements)}**\n\n"
        
        if pp_result.get('parameters'):
            recommendations += "**Selected Pseudopotentials:**\n"
            for elem, params in pp_result['parameters'].items():
                recommendations += f"• **{elem}**: {params['filename']}\n"
                recommendations += f"  - Type: {params['type']}\n"
                recommendations += f"  - Functional: {params['functional']}\n"
                recommendations += f"  - Quality: {params['quality']}\n"
                recommendations += f"  - Relativistic: {params['relativistic']}\n"
                recommendations += f"  - Recommended cutoff: {params['cutoff_energy']['ecutwfc']} Ry (wave), {params['cutoff_energy']['ecutrho']} Ry (density)\n\n"
        
        # Add calculation-specific recommendations
        recommendations += "**Calculation Parameters:**\n"
        recommendations += f"• Calculation type: {calculation_type}\n"
        recommendations += f"• Functional: {functional}\n"
        recommendations += f"• Quality: {quality}\n\n"
        
        # Add general recommendations
        recommendations += "**General Recommendations:**\n"
        recommendations += "• Use PAW pseudopotentials for most calculations\n"
        recommendations += "• PBE functional is recommended for general use\n"
        recommendations += "• PBEsol is better for solid-state calculations\n"
        recommendations += "• Use relativistic pseudopotentials for heavy elements (Z > 20)\n"
        recommendations += "• Start with recommended cutoffs and test convergence\n"
        
        if pp_result.get('missing'):
            recommendations += f"\n**Missing pseudopotentials:** {', '.join(pp_result['missing'])}\n"
        
        return recommendations
        
    except Exception as e:
        return f"Error generating pseudopotential recommendations: {str(e)}"

@tool
def calculate_formation_energy(
    structure_file: str, reference_energies: Dict[str, float], total_energy: float
) -> str:
    """Calculate formation energy of a compound.

    Args:
        structure_file: Path to structure file
        reference_energies: Dict of element -> energy per atom (eV)
        total_energy: Total energy of the compound (eV)

    Returns:
        Formation energy calculation results
    """
    try:
        # Read structure
        atoms = read(structure_file)

        # Get composition
        symbols = atoms.get_chemical_symbols()
        composition = {}
        for symbol in symbols:
            composition[symbol] = composition.get(symbol, 0) + 1

        # Calculate formation energy
        # ΔHf = E_compound - Σ(n_i * E_i^ref)
        reference_sum = 0.0
        missing_refs = []

        for element, count in composition.items():
            if element in reference_energies:
                reference_sum += count * reference_energies[element]
            else:
                missing_refs.append(element)

        if missing_refs:
            return f"Error: Missing reference energies for elements: {', '.join(missing_refs)}"

        formation_energy = total_energy - reference_sum
        formation_energy_per_atom = formation_energy / len(atoms)

        # Generate output
        input_path = Path(structure_file)
        output_dir = input_path.parent / "formation_energies"
        output_dir.mkdir(exist_ok=True)

        # Create formation energy data
        fe_data = {
            "structure_file": structure_file,
            "formula": atoms.get_chemical_formula(),
            "composition": composition,
            "total_energy": total_energy,
            "reference_energies": reference_energies,
            "reference_sum": reference_sum,
            "formation_energy": formation_energy,
            "formation_energy_per_atom": formation_energy_per_atom,
            "num_atoms": len(atoms),
        }

        fe_file = output_dir / f"{input_path.stem}_formation_energy.json"
        with open(fe_file, "w") as f:
            json.dump(fe_data, f, indent=2)

        # Create summary
        summary = f"Formation Energy Calculation for {atoms.get_chemical_formula()}:\\n"
        summary += f"Total Energy: {total_energy:.4f} eV\\n"
        summary += f"Reference Sum: {reference_sum:.4f} eV\\n"
        summary += f"Formation Energy: {formation_energy:.4f} eV\\n"
        summary += (
            f"Formation Energy per atom: {formation_energy_per_atom:.4f} eV/atom\\n"
        )
        summary += f"\\nComposition: {composition}\\n"
        summary += f"Results saved to: {fe_file}"

        return summary

    except Exception as e:
        return f"Error calculating formation energy: {str(e)}"


@tool
def add_adsorbate_pymatgen(
    slab_file: str,
    adsorbate_formula: str,
    site_position: Optional[List[float]] = None,
    height: float = 2.0,
    adsorption_site: str = "top",
    _thread_id: Optional[str] = None,
) -> str:
    """Add adsorbate to surface slab using Pymatgen for robust structure manipulation.

    Args:
        slab_file: Path to slab structure file
        adsorbate_formula: Adsorbate formula/name (e.g., 'CO', 'H', 'O', 'CH4')
        site_position: Two fractional surface coordinates [x, y] each between 0 and 1
        height: Height above surface in Angstrom (distance from surface to bottom of adsorbate)
        adsorption_site: Type of adsorption site ('top', 'bridge', 'hollow', etc.)
        _thread_id: Workspace identifier used to isolate output directories

    Returns:
        String with adsorbate information and file path
    """
    try:
        from backend.utils.workspace import get_subdir_path
        
        # Default center of surface if not provided
        if site_position is None:
            site_position = [0.5, 0.5]

        # Validate site_position
        if not isinstance(site_position, (list, tuple)):
            return "Error: site_position must be a list like [x, y]."
        if len(site_position) != 2:
            return "Error: site_position must have exactly two values [x, y]."
        try:
            sx, sy = float(site_position[0]), float(site_position[1])
        except Exception:
            return "Error: site_position values must be numeric."
        if not (0.0 <= sx <= 1.0 and 0.0 <= sy <= 1.0):
            return "Error: site_position values must be within [0, 1]."

        site_position = [sx, sy]

        # Load slab structure using pymatgen
        slab = Structure.from_file(slab_file)
        
        # Create adsorbate molecule using pymatgen
        if adsorbate_formula in ["H", "O", "N", "C", "S"]:
            # Single atom adsorbates
            adsorbate = Molecule([adsorbate_formula], [[0, 0, 0]])
        elif adsorbate_formula == "CO":
            # CO molecule with proper geometry
            adsorbate = Molecule(["C", "O"], [[0, 0, 0], [0, 0, 1.15]])
        elif adsorbate_formula == "H2":
            adsorbate = Molecule(["H", "H"], [[0, 0, 0], [0, 0, 0.74]])
        elif adsorbate_formula == "O2":
            adsorbate = Molecule(["O", "O"], [[0, 0, 0], [0, 0, 1.21]])
        elif adsorbate_formula == "N2":
            adsorbate = Molecule(["N", "N"], [[0, 0, 0], [0, 0, 1.10]])
        elif adsorbate_formula == "H2O":
            # Water molecule with proper geometry
            adsorbate = Molecule(
                ["O", "H", "H"], 
                [[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]]
            )
        elif adsorbate_formula == "CH4":
            # Methane molecule with proper tetrahedral geometry
            adsorbate = Molecule(
                ["C", "H", "H", "H", "H"],
                [
                    [0, 0, 0],
                    [1.09, 1.09, 1.09],
                    [-1.09, -1.09, 1.09],
                    [-1.09, 1.09, -1.09],
                    [1.09, -1.09, -1.09]
                ]
            )
        else:
            # Try to create as a simple molecule
            try:
                # For unknown molecules, create as single atoms
                adsorbate = Molecule([adsorbate_formula], [[0, 0, 0]])
            except Exception:
                return f"Error: Cannot create adsorbate molecule for {adsorbate_formula}"

        # Find surface atoms (highest z-coordinates in Cartesian)
        surface_atoms = []
        for i, site in enumerate(slab):
            cart_coords = slab.lattice.get_cartesian_coords(site.frac_coords)
            surface_atoms.append((i, cart_coords[2]))
        
        # Sort by z-coordinate and get the highest (surface) atoms
        surface_atoms.sort(key=lambda x: x[1], reverse=True)
        surface_height_cart = surface_atoms[0][1]  # Highest z-coordinate in Cartesian
        
        # Calculate target position in Cartesian coordinates
        target_frac = [site_position[0], site_position[1], 0.5]  # Use middle z for initial target
        target_cart = slab.lattice.get_cartesian_coords(target_frac)
        
        # Find the surface atom closest to the target position
        min_dist = float('inf')
        best_surface_idx = surface_atoms[0][0]
        
        for atom_idx, z_coord in surface_atoms:
            # Only consider atoms near the surface (within 1.0 Å of highest)
            if z_coord >= surface_height_cart - 1.0:
                atom_frac = slab[atom_idx].frac_coords
                atom_cart = slab.lattice.get_cartesian_coords(atom_frac)
                dist = np.linalg.norm(atom_cart[:2] - target_cart[:2])  # Only x,y distance
                if dist < min_dist:
                    min_dist = dist
                    best_surface_idx = atom_idx
        
        # Get the surface atom position
        surface_atom_frac = slab[best_surface_idx].frac_coords
        surface_atom_cart = slab.lattice.get_cartesian_coords(surface_atom_frac)
        
        # For top site adsorption, place the adsorbate directly above this atom
        if adsorption_site == "top":
            # Override the target position to be exactly at the surface atom's x,y coordinates
            target_cart[0] = surface_atom_cart[0]
            target_cart[1] = surface_atom_cart[1]
        
        # Calculate the adsorbate center
        adsorbate_center = np.mean(adsorbate.cart_coords, axis=0)
        
        # For the translation, we want the bottom atom of the adsorbate to be at height above the surface
        # The bottom atom is the one with the lowest z-coordinate
        adsorbate_bottom = np.min(adsorbate.cart_coords[:, 2])
        
        # Calculate the translation needed
        # We want: surface_atom_cart[2] + height = adsorbate_bottom + translation_z
        # So: translation_z = surface_atom_cart[2] + height - adsorbate_bottom
        translation_z = surface_atom_cart[2] + height - adsorbate_bottom
        
        # Translate adsorbate so that the bottom atom is at the correct position
        # Find the bottom atom (lowest z-coordinate)
        bottom_atom_idx = np.argmin(adsorbate.cart_coords[:, 2])
        bottom_atom_pos = adsorbate.cart_coords[bottom_atom_idx]
        
        # Calculate translation to place bottom atom at target position
        translation_vector = [
            target_cart[0] - bottom_atom_pos[0],
            target_cart[1] - bottom_atom_pos[1],
            (surface_atom_cart[2] + height) - bottom_atom_pos[2]
        ]
        
        adsorbate.translate_sites(list(range(len(adsorbate))), translation_vector)
        
        # Add adsorbate to slab
        for site in adsorbate:
            slab.append(site.specie, site.coords, coords_are_cartesian=True)
        
        # Save the structure
        input_path = Path(slab_file)
        output_dir = get_subdir_path(_thread_id, "structures/with_adsorbates")
        
        pos_str = f"x{site_position[0]:.2f}y{site_position[1]:.2f}"
        stem = f"{input_path.stem}_{adsorbate_formula}_{pos_str}_h{height:.1f}_{adsorption_site}"
        output_path = output_dir / f"{stem}.cif"
        
        slab.to(str(output_path), fmt="cif")
        
        # Save metadata
        metadata = {
            "slab_file": slab_file,
            "adsorbate": adsorbate_formula,
            "site_position": site_position,
            "height": height,
            "adsorption_site": adsorption_site,
            "total_atoms": len(slab),
            "adsorbate_atoms": len(adsorbate),
            "formula": slab.composition.formula,
            "files": {"cif": str(output_path)},
        }
        
        metadata_file = output_path.with_suffix(".json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        return (
            f"Added {adsorbate_formula} adsorbate at {adsorption_site} site {site_position}, "
            f"height {height} Å. Total atoms: {len(slab)}. "
            f"Saved to {output_path}"
        )

    except Exception as e:
        return f"Error adding adsorbate: {str(e)}"
