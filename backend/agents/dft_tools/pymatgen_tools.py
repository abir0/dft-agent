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
    """Find and validate pseudopotentials for elements.

    Args:
        elements: List of chemical elements
        pp_type: Pseudopotential type ('paw', 'nc', 'us')
        pp_library: Pseudopotential library ('psl', 'gbrv', 'sg15')
        functional: Exchange-correlation functional ('pbe', 'lda', 'pbesol')

    Returns:
        Information about available pseudopotentials
    """
    try:
        # Common pseudopotential naming conventions
        pp_mappings = {
            "psl": {
                "pbe": {
                    "H": "H.pbe-rrkjus_psl.1.0.0.UPF",
                    "He": "He.pbe-n-rrkjus_psl.1.0.0.UPF",
                    "Li": "Li.pbe-s-rrkjus_psl.1.0.0.UPF",
                    "Be": "Be.pbe-n-rrkjus_psl.1.0.0.UPF",
                    "B": "B.pbe-n-rrkjus_psl.1.0.0.UPF",
                    "C": "C.pbe-n-rrkjus_psl.1.0.0.UPF",
                    "N": "N.pbe-n-rrkjus_psl.1.0.0.UPF",
                    "O": "O.pbe-n-rrkjus_psl.1.0.0.UPF",
                    "F": "F.pbe-n-rrkjus_psl.1.0.0.UPF",
                    "Ne": "Ne.pbe-n-rrkjus_psl.1.0.0.UPF",
                    "Na": "Na.pbe-spn-rrkjus_psl.1.0.0.UPF",
                    "Mg": "Mg.pbe-spn-rrkjus_psl.1.0.0.UPF",
                    "Al": "Al.pbe-n-rrkjus_psl.1.0.0.UPF",
                    "Si": "Si.pbe-n-rrkjus_psl.1.0.0.UPF",
                    "P": "P.pbe-n-rrkjus_psl.1.0.0.UPF",
                    "S": "S.pbe-n-rrkjus_psl.1.0.0.UPF",
                    "Cl": "Cl.pbe-n-rrkjus_psl.1.0.0.UPF",
                    "Ar": "Ar.pbe-n-rrkjus_psl.1.0.0.UPF",
                    "K": "K.pbe-spn-rrkjus_psl.1.0.0.UPF",
                    "Ca": "Ca.pbe-spn-rrkjus_psl.1.0.0.UPF",
                    "Ti": "Ti.pbe-spn-rrkjus_psl.1.0.0.UPF",
                    "V": "V.pbe-spn-rrkjus_psl.1.0.0.UPF",
                    "Cr": "Cr.pbe-spn-rrkjus_psl.1.0.0.UPF",
                    "Mn": "Mn.pbe-spn-rrkjus_psl.1.0.0.UPF",
                    "Fe": "Fe.pbe-spn-rrkjus_psl.1.0.0.UPF",
                    "Co": "Co.pbe-spn-rrkjus_psl.1.0.0.UPF",
                    "Ni": "Ni.pbe-spn-rrkjus_psl.1.0.0.UPF",
                    "Cu": "Cu.pbe-spn-rrkjus_psl.1.0.0.UPF",
                    "Zn": "Zn.pbe-spn-rrkjus_psl.1.0.0.UPF",
                    "Ga": "Ga.pbe-spn-rrkjus_psl.1.0.0.UPF",
                    "Ge": "Ge.pbe-n-rrkjus_psl.1.0.0.UPF",
                    "As": "As.pbe-n-rrkjus_psl.1.0.0.UPF",
                    "Se": "Se.pbe-n-rrkjus_psl.1.0.0.UPF",
                    "Br": "Br.pbe-n-rrkjus_psl.1.0.0.UPF",
                    "Kr": "Kr.pbe-n-rrkjus_psl.1.0.0.UPF",
                    "Pt": "Pt.pbe-spfn-rrkjus_psl.1.0.0.UPF",
                    "Au": "Au.pbe-spfn-rrkjus_psl.1.0.0.UPF",
                }
            }
        }

        # Find pseudopotentials for requested elements
        found_pps = {}
        missing_pps = []

        for element in elements:
            if (
                pp_library in pp_mappings
                and functional in pp_mappings[pp_library]
                and element in pp_mappings[pp_library][functional]
            ):
                found_pps[element] = pp_mappings[pp_library][functional][element]
            else:
                # Generate a generic filename
                if pp_library == "psl":
                    if element in ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne"]:
                        suffix = "rrkjus_psl.1.0.0.UPF"
                    else:
                        suffix = "spn-rrkjus_psl.1.0.0.UPF"
                    pp_filename = f"{element}.{functional}-{suffix}"
                else:
                    pp_filename = f"{element}.{functional}-{pp_type}.UPF"

                found_pps[element] = pp_filename
                missing_pps.append(element)

        # Create output directory and save PP mapping
        output_dir = Path("pseudopotentials")
        output_dir.mkdir(exist_ok=True)

        pp_data = {
            "elements": elements,
            "pp_type": pp_type,
            "pp_library": pp_library,
            "functional": functional,
            "pseudopotentials": found_pps,
            "missing_or_generic": missing_pps,
            "notes": "Verify pseudopotential files exist in your PP directory",
        }

        pp_file = output_dir / f"pp_mapping_{functional}_{pp_library}.json"
        with open(pp_file, "w") as f:
            json.dump(pp_data, f, indent=2)

        # Create summary
        summary = f"Pseudopotential mapping for {len(elements)} elements:\\n"
        summary += f"Functional: {functional.upper()}, Library: {pp_library.upper()}, Type: {pp_type.upper()}\\n\\n"

        for element in elements:
            summary += f"{element}: {found_pps[element]}"
            if element in missing_pps:
                summary += " (verify availability)"
            summary += "\\n"

        summary += f"\\nMapping saved to: {pp_file}"

        if missing_pps:
            summary += f"\\n\\nNote: Please verify availability of pseudopotentials for: {', '.join(missing_pps)}"

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
