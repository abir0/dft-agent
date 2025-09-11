"""
Pymatgen-based Tools for Materials Analysis

Tools for materials database integration, crystal analysis, and electronic structure
analysis using Pymatgen and Materials Project API.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from langchain_core.tools import tool

try:
    from mp_api.client import MPRester
    from pymatgen.analysis.local_env import CrystalNN
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    MP_API_AVAILABLE = True
except ImportError:
    MP_API_AVAILABLE = False

try:
    from ase.io import read, write

    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False


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
        api_key: Materials Project API key

    Returns:
        Search results with material properties
    """
    try:
        if not MP_API_AVAILABLE:
            return "Error: Materials Project API not available. Please install with: pip install mp-api"

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
            docs = mpr.materials.summary.search(formula=formula, fields=properties)[
                :limit
            ]

        if not docs:
            return f"No materials found for formula: {formula}"

        # Create output directory
        output_dir = Path("materials_project_data")
        output_dir.mkdir(exist_ok=True)

        results = []
        for i, doc in enumerate(docs):
            result = {
                "material_id": doc.material_id,
                "formula": doc.formula_pretty,
            }

            # Add available properties
            if (
                hasattr(doc, "formation_energy_per_atom")
                and doc.formation_energy_per_atom is not None
            ):
                result["formation_energy_per_atom"] = float(doc.formation_energy_per_atom)

            if hasattr(doc, "band_gap") and doc.band_gap is not None:
                result["band_gap"] = float(doc.band_gap)

            if hasattr(doc, "density") and doc.density is not None:
                result["density"] = float(doc.density)

            if hasattr(doc, "symmetry"):
                result["spacegroup"] = doc.symmetry.symbol
                result["spacegroup_number"] = doc.symmetry.number

            # Save structure if available
            if hasattr(doc, "structure") and doc.structure is not None:
                structure_file = (
                    output_dir / f"{doc.material_id}_{doc.formula_pretty}.cif"
                )
                doc.structure.to(filename=str(structure_file))
                result["structure_file"] = str(structure_file)

            results.append(result)

        # Save search results
        results_file = output_dir / f"search_{formula.replace(' ', '_')}_results.json"
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
        if not MP_API_AVAILABLE:
            return (
                "Error: Pymatgen not available. Please install with: pip install pymatgen"
            )

        if not ASE_AVAILABLE:
            return "Error: ASE not available. Please install with: pip install ase"

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
    pp_type: str = "paw",
    pp_library: str = "psl",
    functional: str = "pbe",
) -> str:
    """Find and validate pseudopotentials for elements using pslibrary database.

    Args:
        elements: List of chemical elements
        pp_type: Pseudopotential type ('paw', 'nc', 'us')
        pp_library: Pseudopotential library ('psl', 'gbrv', 'sg15')
        functional: Exchange-correlation functional ('pbe', 'lda', 'pbesol')

    Returns:
        Information about available pseudopotentials
    """
    try:
        # Load pslibrary database
        pp_db_path = Path("data/inputs/pseudopotentials/pslibrary_database.json")
        if not pp_db_path.exists():
            return "Error: pslibrary database not found. Please ensure the database is created."
        
        with open(pp_db_path, 'r') as f:
            pp_database = json.load(f)
        
        # Find pseudopotentials for requested elements
        found_pps = {}
        missing_pps = []
        pp_details = {}
        
        for element in elements:
            element = element.capitalize()
            
            if element in pp_database['pseudopotentials']:
                element_pps = pp_database['pseudopotentials'][element]['available_pseudopotentials']
                
                # Find matching pseudopotential
                matching_pp = None
                for pp in element_pps:
                    if (pp['functional'] == functional and 
                        pp['type'].lower() == pp_type.lower() and
                        pp['recommended']):
                        matching_pp = pp
                        break
                
                if matching_pp:
                    found_pps[element] = matching_pp['filename']
                    pp_details[element] = {
                        'filename': matching_pp['filename'],
                        'type': matching_pp['type'],
                        'functional': matching_pp['functional'],
                        'quality': matching_pp['quality'],
                        'relativistic': matching_pp['relativistic'],
                        'cutoff_energy': matching_pp['cutoff_energy'],
                        'description': matching_pp['description']
                    }
                else:
                    missing_pps.append(element)
            else:
                missing_pps.append(element)
        
        # Create output directory and save PP mapping
        output_dir = Path("data/outputs/pseudopotentials")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        pp_data = {
            "elements": elements,
            "pp_type": pp_type,
            "pp_library": pp_library,
            "functional": functional,
            "pseudopotentials": found_pps,
            "pseudopotential_details": pp_details,
            "missing_elements": missing_pps,
            "source": "pslibrary",
            "notes": "Pseudopotentials from pslibrary database"
        }
        
        pp_file = output_dir / f"pp_mapping_{functional}_{pp_library}.json"
        with open(pp_file, "w") as f:
            json.dump(pp_data, f, indent=2)
        
        summary = f"Found pseudopotentials for {len(found_pps)} elements:\n"
        for element, pp_file in found_pps.items():
            details = pp_details[element]
            summary += f"  {element}: {pp_file}\n"
            summary += f"    Type: {details['type']}, Quality: {details['quality']}\n"
            summary += f"    Cutoff: {details['cutoff_energy']['ecutwfc']} Ry\n"
            if details['relativistic']:
                summary += f"    Relativistic: Yes\n"
            summary += "\n"
        
        summary += f"Mapping saved to: {pp_file}"
        
        if missing_pps:
            summary += f"\n\nMissing pseudopotentials for: {', '.join(missing_pps)}"
        
        return summary
        
    except Exception as e:
        return f"Error finding pseudopotentials: {str(e)}"


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
        if not ASE_AVAILABLE:
            return "Error: ASE not available. Please install with: pip install ase"

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
