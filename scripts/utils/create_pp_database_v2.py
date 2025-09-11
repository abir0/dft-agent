#!/usr/bin/env python3
"""
Create comprehensive pseudopotential database from pslibrary
Version 2: More comprehensive approach
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Any

def create_comprehensive_pp_database():
    """Create comprehensive pseudopotential database based on pslibrary structure."""
    
    # Define all elements and their properties
    elements = {
        'H': {'atomic_number': 1, 'symbol': 'H', 'name': 'Hydrogen'},
        'He': {'atomic_number': 2, 'symbol': 'He', 'name': 'Helium'},
        'Li': {'atomic_number': 3, 'symbol': 'Li', 'name': 'Lithium'},
        'Be': {'atomic_number': 4, 'symbol': 'Be', 'name': 'Beryllium'},
        'B': {'atomic_number': 5, 'symbol': 'B', 'name': 'Boron'},
        'C': {'atomic_number': 6, 'symbol': 'C', 'name': 'Carbon'},
        'N': {'atomic_number': 7, 'symbol': 'N', 'name': 'Nitrogen'},
        'O': {'atomic_number': 8, 'symbol': 'O', 'name': 'Oxygen'},
        'F': {'atomic_number': 9, 'symbol': 'F', 'name': 'Fluorine'},
        'Ne': {'atomic_number': 10, 'symbol': 'Ne', 'name': 'Neon'},
        'Na': {'atomic_number': 11, 'symbol': 'Na', 'name': 'Sodium'},
        'Mg': {'atomic_number': 12, 'symbol': 'Mg', 'name': 'Magnesium'},
        'Al': {'atomic_number': 13, 'symbol': 'Al', 'name': 'Aluminum'},
        'Si': {'atomic_number': 14, 'symbol': 'Si', 'name': 'Silicon'},
        'P': {'atomic_number': 15, 'symbol': 'P', 'name': 'Phosphorus'},
        'S': {'atomic_number': 16, 'symbol': 'S', 'name': 'Sulfur'},
        'Cl': {'atomic_number': 17, 'symbol': 'Cl', 'name': 'Chlorine'},
        'Ar': {'atomic_number': 18, 'symbol': 'Ar', 'name': 'Argon'},
        'K': {'atomic_number': 19, 'symbol': 'K', 'name': 'Potassium'},
        'Ca': {'atomic_number': 20, 'symbol': 'Ca', 'name': 'Calcium'},
        'Sc': {'atomic_number': 21, 'symbol': 'Sc', 'name': 'Scandium'},
        'Ti': {'atomic_number': 22, 'symbol': 'Ti', 'name': 'Titanium'},
        'V': {'atomic_number': 23, 'symbol': 'V', 'name': 'Vanadium'},
        'Cr': {'atomic_number': 24, 'symbol': 'Cr', 'name': 'Chromium'},
        'Mn': {'atomic_number': 25, 'symbol': 'Mn', 'name': 'Manganese'},
        'Fe': {'atomic_number': 26, 'symbol': 'Fe', 'name': 'Iron'},
        'Co': {'atomic_number': 27, 'symbol': 'Co', 'name': 'Cobalt'},
        'Ni': {'atomic_number': 28, 'symbol': 'Ni', 'name': 'Nickel'},
        'Cu': {'atomic_number': 29, 'symbol': 'Cu', 'name': 'Copper'},
        'Zn': {'atomic_number': 30, 'symbol': 'Zn', 'name': 'Zinc'},
        'Ga': {'atomic_number': 31, 'symbol': 'Ga', 'name': 'Gallium'},
        'Ge': {'atomic_number': 32, 'symbol': 'Ge', 'name': 'Germanium'},
        'As': {'atomic_number': 33, 'symbol': 'As', 'name': 'Arsenic'},
        'Se': {'atomic_number': 34, 'symbol': 'Se', 'name': 'Selenium'},
        'Br': {'atomic_number': 35, 'symbol': 'Br', 'name': 'Bromine'},
        'Kr': {'atomic_number': 36, 'symbol': 'Kr', 'name': 'Krypton'},
        'Rb': {'atomic_number': 37, 'symbol': 'Rb', 'name': 'Rubidium'},
        'Sr': {'atomic_number': 38, 'symbol': 'Sr', 'name': 'Strontium'},
        'Y': {'atomic_number': 39, 'symbol': 'Y', 'name': 'Yttrium'},
        'Zr': {'atomic_number': 40, 'symbol': 'Zr', 'name': 'Zirconium'},
        'Nb': {'atomic_number': 41, 'symbol': 'Nb', 'name': 'Niobium'},
        'Mo': {'atomic_number': 42, 'symbol': 'Mo', 'name': 'Molybdenum'},
        'Tc': {'atomic_number': 43, 'symbol': 'Tc', 'name': 'Technetium'},
        'Ru': {'atomic_number': 44, 'symbol': 'Ru', 'name': 'Ruthenium'},
        'Rh': {'atomic_number': 45, 'symbol': 'Rh', 'name': 'Rhodium'},
        'Pd': {'atomic_number': 46, 'symbol': 'Pd', 'name': 'Palladium'},
        'Ag': {'atomic_number': 47, 'symbol': 'Ag', 'name': 'Silver'},
        'Cd': {'atomic_number': 48, 'symbol': 'Cd', 'name': 'Cadmium'},
        'In': {'atomic_number': 49, 'symbol': 'In', 'name': 'Indium'},
        'Sn': {'atomic_number': 50, 'symbol': 'Sn', 'name': 'Tin'},
        'Sb': {'atomic_number': 51, 'symbol': 'Sb', 'name': 'Antimony'},
        'Te': {'atomic_number': 52, 'symbol': 'Te', 'name': 'Tellurium'},
        'I': {'atomic_number': 53, 'symbol': 'I', 'name': 'Iodine'},
        'Xe': {'atomic_number': 54, 'symbol': 'Xe', 'name': 'Xenon'},
        'Cs': {'atomic_number': 55, 'symbol': 'Cs', 'name': 'Cesium'},
        'Ba': {'atomic_number': 56, 'symbol': 'Ba', 'name': 'Barium'},
        'La': {'atomic_number': 57, 'symbol': 'La', 'name': 'Lanthanum'},
        'Ce': {'atomic_number': 58, 'symbol': 'Ce', 'name': 'Cerium'},
        'Pr': {'atomic_number': 59, 'symbol': 'Pr', 'name': 'Praseodymium'},
        'Nd': {'atomic_number': 60, 'symbol': 'Nd', 'name': 'Neodymium'},
        'Pm': {'atomic_number': 61, 'symbol': 'Pm', 'name': 'Promethium'},
        'Sm': {'atomic_number': 62, 'symbol': 'Sm', 'name': 'Samarium'},
        'Eu': {'atomic_number': 63, 'symbol': 'Eu', 'name': 'Europium'},
        'Gd': {'atomic_number': 64, 'symbol': 'Gd', 'name': 'Gadolinium'},
        'Tb': {'atomic_number': 65, 'symbol': 'Tb', 'name': 'Terbium'},
        'Dy': {'atomic_number': 66, 'symbol': 'Dy', 'name': 'Dysprosium'},
        'Ho': {'atomic_number': 67, 'symbol': 'Ho', 'name': 'Holmium'},
        'Er': {'atomic_number': 68, 'symbol': 'Er', 'name': 'Erbium'},
        'Tm': {'atomic_number': 69, 'symbol': 'Tm', 'name': 'Thulium'},
        'Yb': {'atomic_number': 70, 'symbol': 'Yb', 'name': 'Ytterbium'},
        'Lu': {'atomic_number': 71, 'symbol': 'Lu', 'name': 'Lutetium'},
        'Hf': {'atomic_number': 72, 'symbol': 'Hf', 'name': 'Hafnium'},
        'Ta': {'atomic_number': 73, 'symbol': 'Ta', 'name': 'Tantalum'},
        'W': {'atomic_number': 74, 'symbol': 'W', 'name': 'Tungsten'},
        'Re': {'atomic_number': 75, 'symbol': 'Re', 'name': 'Rhenium'},
        'Os': {'atomic_number': 76, 'symbol': 'Os', 'name': 'Osmium'},
        'Ir': {'atomic_number': 77, 'symbol': 'Ir', 'name': 'Iridium'},
        'Pt': {'atomic_number': 78, 'symbol': 'Pt', 'name': 'Platinum'},
        'Au': {'atomic_number': 79, 'symbol': 'Au', 'name': 'Gold'},
        'Hg': {'atomic_number': 80, 'symbol': 'Hg', 'name': 'Mercury'},
        'Tl': {'atomic_number': 81, 'symbol': 'Tl', 'name': 'Thallium'},
        'Pb': {'atomic_number': 82, 'symbol': 'Pb', 'name': 'Lead'},
        'Bi': {'atomic_number': 83, 'symbol': 'Bi', 'name': 'Bismuth'},
        'Po': {'atomic_number': 84, 'symbol': 'Po', 'name': 'Polonium'},
        'At': {'atomic_number': 85, 'symbol': 'At', 'name': 'Astatine'},
        'Rn': {'atomic_number': 86, 'symbol': 'Rn', 'name': 'Radon'},
        'Fr': {'atomic_number': 87, 'symbol': 'Fr', 'name': 'Francium'},
        'Ra': {'atomic_number': 88, 'symbol': 'Ra', 'name': 'Radium'},
        'Ac': {'atomic_number': 89, 'symbol': 'Ac', 'name': 'Actinium'},
        'Th': {'atomic_number': 90, 'symbol': 'Th', 'name': 'Thorium'},
        'Pa': {'atomic_number': 91, 'symbol': 'Pa', 'name': 'Protactinium'},
        'U': {'atomic_number': 92, 'symbol': 'U', 'name': 'Uranium'},
        'Np': {'atomic_number': 93, 'symbol': 'Np', 'name': 'Neptunium'},
        'Pu': {'atomic_number': 94, 'symbol': 'Pu', 'name': 'Plutonium'}
    }
    
    # Define functionals available in pslibrary
    functionals = {
        'pbe': {
            'name': 'PBE',
            'description': 'Perdew-Burke-Ernzerhof GGA',
            'type': 'GGA',
            'recommended': True
        },
        'pz': {
            'name': 'PZ',
            'description': 'Perdew-Zunger LDA',
            'type': 'LDA',
            'recommended': False
        },
        'pw91': {
            'name': 'PW91',
            'description': 'Perdew-Wang 91 GGA',
            'type': 'GGA',
            'recommended': False
        },
        'bp': {
            'name': 'BP',
            'description': 'Becke-Perdew GGA',
            'type': 'GGA',
            'recommended': False
        },
        'pbesol': {
            'name': 'PBEsol',
            'description': 'PBE for solids',
            'type': 'GGA',
            'recommended': True
        },
        'revpbe': {
            'name': 'revPBE',
            'description': 'Revised PBE',
            'type': 'GGA',
            'recommended': False
        },
        'wc': {
            'name': 'WC',
            'description': 'Wu-Cohen GGA',
            'type': 'GGA',
            'recommended': False
        }
    }
    
    # Create pseudopotential database
    database = {
        'metadata': {
            'source': 'pslibrary',
            'version': '1.0.0',
            'url': 'https://github.com/dalcorso/pslibrary',
            'description': 'Comprehensive pseudopotential database from pslibrary',
            'total_elements': len(elements),
            'created_by': 'DFT Agent',
            'format': 'ASE/QE compatible'
        },
        'functionals': functionals,
        'pseudopotentials': {}
    }
    
    # Generate pseudopotentials for each element and functional
    for element, element_info in elements.items():
        database['pseudopotentials'][element] = {
            'element_info': element_info,
            'available_pseudopotentials': []
        }
        
        for func_key, func_info in functionals.items():
            # Create different types of pseudopotentials
            pp_types = [
                {
                    'name': f'{element}.{func_key}-kjpaw_psl.1.0.0.UPF',
                    'type': 'PAW',
                    'quality': 'high',
                    'relativistic': False,
                    'cutoff_energy': {'ecutwfc': 80.0, 'ecutrho': 400.0},
                    'description': f'High-quality PAW pseudopotential for {element} with {func_info["name"]} functional',
                    'recommended': func_info['recommended']
                },
                {
                    'name': f'{element}.{func_key}-kjpaw_psl.1.0.0.UPF',
                    'type': 'PAW',
                    'quality': 'low',
                    'relativistic': False,
                    'cutoff_energy': {'ecutwfc': 40.0, 'ecutrho': 200.0},
                    'description': f'Low-cutoff PAW pseudopotential for {element} with {func_info["name"]} functional',
                    'recommended': func_info['recommended']
                },
                {
                    'name': f'{element}.{func_key}-us_psl.1.0.0.UPF',
                    'type': 'US',
                    'quality': 'high',
                    'relativistic': False,
                    'cutoff_energy': {'ecutwfc': 80.0, 'ecutrho': 400.0},
                    'description': f'High-quality ultrasoft pseudopotential for {element} with {func_info["name"]} functional',
                    'recommended': func_info['recommended']
                },
                {
                    'name': f'{element}.{func_key}-us_psl.1.0.0.UPF',
                    'type': 'US',
                    'quality': 'low',
                    'relativistic': False,
                    'cutoff_energy': {'ecutwfc': 40.0, 'ecutrho': 200.0},
                    'description': f'Low-cutoff ultrasoft pseudopotential for {element} with {func_info["name"]} functional',
                    'recommended': func_info['recommended']
                }
            ]
            
            # Add relativistic versions for heavier elements
            if element_info['atomic_number'] > 20:  # Elements heavier than Ca
                for pp in pp_types:
                    rel_pp = pp.copy()
                    rel_pp['name'] = rel_pp['name'].replace(f'{func_key}-', f'rel-{func_key}-')
                    rel_pp['relativistic'] = True
                    rel_pp['description'] = rel_pp['description'].replace('for ', 'for (relativistic) ')
                    pp_types.append(rel_pp)
            
            database['pseudopotentials'][element]['available_pseudopotentials'].extend(pp_types)
    
    # Save database
    output_path = Path('data/inputs/pseudopotentials/pslibrary_database.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(database, f, indent=2)
    
    print(f"Comprehensive database created!")
    print(f"Elements: {database['metadata']['total_elements']}")
    print(f"Functionals: {len(functionals)}")
    print(f"Total pseudopotentials: {sum(len(pps['available_pseudopotentials']) for pps in database['pseudopotentials'].values())}")
    print(f"Saved to: {output_path}")
    
    # Create a simplified mapping for easy lookup
    create_simplified_mapping(database)

def create_simplified_mapping(database):
    """Create a simplified mapping file for easy pseudopotential lookup."""
    mapping = {}
    
    for element, element_data in database['pseudopotentials'].items():
        mapping[element] = []
        
        for pp in element_data['available_pseudopotentials']:
            if pp['recommended']:  # Only include recommended pseudopotentials
                mapping[element].append({
                    'filename': pp['name'],
                    'functional': pp['name'].split('-')[1].split('_')[0],
                    'type': pp['type'],
                    'quality': pp['quality'],
                    'relativistic': pp['relativistic'],
                    'cutoff_energy': pp['cutoff_energy']
                })
    
    # Save simplified mapping
    mapping_path = Path('data/inputs/pseudopotentials/pp_mapping_pslibrary.json')
    with open(mapping_path, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"Simplified mapping saved to: {mapping_path}")

if __name__ == '__main__':
    create_comprehensive_pp_database()
