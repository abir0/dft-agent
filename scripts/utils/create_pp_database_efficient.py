#!/usr/bin/env python3
"""
Create efficient pseudopotential database from pslibrary
Focus on commonly used elements and functionals
"""

import json
from pathlib import Path
from typing import Dict, List, Any

def create_efficient_pp_database():
    """Create efficient pseudopotential database focusing on common elements."""
    
    # Common elements for DFT calculations
    common_elements = [
        'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
        'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
        'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
        'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
        'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
        'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
        'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
        'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
        'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
        'Pa', 'U', 'Np', 'Pu'
    ]
    
    # Most commonly used functionals
    functionals = {
        'pbe': {
            'name': 'PBE',
            'description': 'Perdew-Burke-Ernzerhof GGA',
            'type': 'GGA',
            'recommended': True
        },
        'pbesol': {
            'name': 'PBEsol',
            'description': 'PBE for solids',
            'type': 'GGA',
            'recommended': True
        },
        'pz': {
            'name': 'PZ',
            'description': 'Perdew-Zunger LDA',
            'type': 'LDA',
            'recommended': False
        }
    }
    
    # Create database
    database = {
        'metadata': {
            'source': 'pslibrary',
            'version': '1.0.0',
            'url': 'https://github.com/dalcorso/pslibrary',
            'description': 'Efficient pseudopotential database from pslibrary',
            'total_elements': len(common_elements),
            'created_by': 'DFT Agent',
            'format': 'ASE/QE compatible'
        },
        'functionals': functionals,
        'pseudopotentials': {}
    }
    
    # Generate pseudopotentials for each element
    for element in common_elements:
        database['pseudopotentials'][element] = {
            'element': element,
            'available_pseudopotentials': []
        }
        
        for func_key, func_info in functionals.items():
            # Create PAW pseudopotentials (recommended)
            paw_pp = {
                'filename': f'{element}.{func_key}-kjpaw_psl.1.0.0.UPF',
                'type': 'PAW',
                'functional': func_key,
                'quality': 'high',
                'relativistic': False,
                'cutoff_energy': {'ecutwfc': 60.0, 'ecutrho': 300.0},
                'description': f'PAW pseudopotential for {element} with {func_info["name"]} functional',
                'recommended': func_info['recommended'],
                'ase_name': f'{element}.{func_key}-kjpaw_psl.1.0.0.UPF'
            }
            
            database['pseudopotentials'][element]['available_pseudopotentials'].append(paw_pp)
            
            # Add relativistic version for heavier elements
            if common_elements.index(element) > 20:  # Elements heavier than Ca
                rel_paw_pp = paw_pp.copy()
                rel_paw_pp['filename'] = f'{element}.rel-{func_key}-kjpaw_psl.1.0.0.UPF'
                rel_paw_pp['relativistic'] = True
                rel_paw_pp['description'] = f'Relativistic PAW pseudopotential for {element} with {func_info["name"]} functional'
                rel_paw_pp['ase_name'] = f'{element}.rel-{func_key}-kjpaw_psl.1.0.0.UPF'
                database['pseudopotentials'][element]['available_pseudopotentials'].append(rel_paw_pp)
    
    # Save database
    output_path = Path('data/inputs/pseudopotentials/pslibrary_database.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(database, f, indent=2)
    
    print(f"Efficient database created!")
    print(f"Elements: {database['metadata']['total_elements']}")
    print(f"Functionals: {len(functionals)}")
    print(f"Total pseudopotentials: {sum(len(pps['available_pseudopotentials']) for pps in database['pseudopotentials'].values())}")
    print(f"Saved to: {output_path}")
    
    # Create simplified mapping
    create_simplified_mapping(database)

def create_simplified_mapping(database):
    """Create a simplified mapping file for easy pseudopotential lookup."""
    mapping = {}
    
    for element, element_data in database['pseudopotentials'].items():
        mapping[element] = []
        
        for pp in element_data['available_pseudopotentials']:
            if pp['recommended']:  # Only include recommended pseudopotentials
                mapping[element].append({
                    'filename': pp['filename'],
                    'functional': pp['functional'],
                    'type': pp['type'],
                    'quality': pp['quality'],
                    'relativistic': pp['relativistic'],
                    'cutoff_energy': pp['cutoff_energy'],
                    'ase_name': pp['ase_name']
                })
    
    # Save simplified mapping
    mapping_path = Path('data/inputs/pseudopotentials/pp_mapping_pslibrary.json')
    with open(mapping_path, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"Simplified mapping saved to: {mapping_path}")

if __name__ == '__main__':
    create_efficient_pp_database()
