#!/usr/bin/env python3
"""
Create comprehensive pseudopotential database from pslibrary
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Any

def parse_job_file(job_file: Path) -> List[Dict[str, Any]]:
    """Parse a pslibrary job file to extract pseudopotential information."""
    pseudopotentials = []
    
    with open(job_file, 'r') as f:
        content = f.read()
    
    # Find all pseudopotential definitions
    # Pattern: cat > Element.functional-type.UPF.in << EOF
    pattern = r'cat > ([A-Za-z]+)\.([^\.]+)\.in << EOF'
    matches = re.findall(pattern, content)
    
    for element, pp_name in matches:
        # Extract the configuration block
        start_pattern = f'cat > {element}\\.{pp_name}\\.in << EOF'
        end_pattern = 'EOF'
        
        start_idx = content.find(start_pattern)
        if start_idx == -1:
            continue
            
        # Find the end of this pseudopotential definition
        end_idx = content.find(end_pattern, start_idx + len(start_pattern))
        if end_idx == -1:
            continue
            
        pp_block = content[start_idx:end_idx]
        
        # Parse the pseudopotential parameters
        pp_info = {
            'element': element,
            'name': pp_name,
            'filename': f'{element}.{pp_name}.UPF',
            'type': 'PAW' if 'paw' in pp_name.lower() else 'US',
            'functional': extract_functional(pp_name),
            'relativistic': 'rel' in pp_name.lower(),
            'valence_config': extract_valence_config(pp_block),
            'cutoff_energy': estimate_cutoff_energy(pp_name),
            'notes': extract_notes(pp_name)
        }
        
        pseudopotentials.append(pp_info)
    
    return pseudopotentials

def extract_functional(pp_name: str) -> str:
    """Extract functional from pseudopotential name."""
    functionals = {
        'pbe': 'PBE',
        'pz': 'PZ',
        'pw91': 'PW91',
        'bp': 'BP',
        'pbesol': 'PBEsol',
        'revpbe': 'revPBE',
        'wc': 'WC'
    }
    
    for func_key, func_name in functionals.items():
        if func_key in pp_name.lower():
            return func_name
    
    return 'Unknown'

def extract_valence_config(pp_block: str) -> str:
    """Extract valence configuration from pseudopotential block."""
    # Look for config line
    config_match = re.search(r"config='([^']+)'", pp_block)
    if config_match:
        return config_match.group(1)
    return 'Unknown'

def estimate_cutoff_energy(pp_name: str) -> Dict[str, float]:
    """Estimate cutoff energy based on pseudopotential name."""
    if 'low' in pp_name.lower():
        return {'ecutwfc': 40.0, 'ecutrho': 200.0}
    elif 'high' in pp_name.lower():
        return {'ecutwfc': 80.0, 'ecutrho': 400.0}
    else:
        return {'ecutwfc': 60.0, 'ecutrho': 300.0}

def extract_notes(pp_name: str) -> str:
    """Extract notes about the pseudopotential."""
    notes = []
    
    if 'low' in pp_name.lower():
        notes.append('Low cutoff optimized')
    if 'high' in pp_name.lower():
        notes.append('High accuracy')
    if 'rel' in pp_name.lower():
        notes.append('Relativistic')
    if 'paw' in pp_name.lower():
        notes.append('PAW dataset')
    if 'us' in pp_name.lower():
        notes.append('Ultrasoft')
    
    return '; '.join(notes) if notes else 'Standard'

def create_pp_database():
    """Create comprehensive pseudopotential database."""
    pslibrary_path = Path('data/inputs/pseudopotentials/pslibrary')
    
    if not pslibrary_path.exists():
        print(f"Error: pslibrary not found at {pslibrary_path}")
        return
    
    all_pseudopotentials = {}
    
    # Process all job files
    job_files = [
        'paw_ps_high.job',
        'paw_ps_low.job', 
        'us_ps_high.job',
        'us_ps_low.job'
    ]
    
    for job_file in job_files:
        job_path = pslibrary_path / job_file
        if job_path.exists():
            print(f"Processing {job_file}...")
            pps = parse_job_file(job_path)
            
            for pp in pps:
                element = pp['element']
                if element not in all_pseudopotentials:
                    all_pseudopotentials[element] = []
                all_pseudopotentials[element].append(pp)
    
    # Create organized database
    database = {
        'metadata': {
            'source': 'pslibrary',
            'version': '1.0.0',
            'url': 'https://github.com/dalcorso/pslibrary',
            'description': 'Comprehensive pseudopotential database from pslibrary',
            'total_elements': len(all_pseudopotentials),
            'total_pseudopotentials': sum(len(pps) for pps in all_pseudopotentials.values())
        },
        'functionals': {
            'PBE': 'Perdew-Burke-Ernzerhof GGA',
            'PZ': 'Perdew-Zunger LDA',
            'PW91': 'Perdew-Wang 91 GGA',
            'BP': 'Becke-Perdew GGA',
            'PBEsol': 'PBE for solids',
            'revPBE': 'Revised PBE',
            'WC': 'Wu-Cohen GGA'
        },
        'pseudopotentials': all_pseudopotentials
    }
    
    # Save database
    output_path = Path('data/inputs/pseudopotentials/pslibrary_database.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(database, f, indent=2)
    
    print(f"Database created with {database['metadata']['total_elements']} elements")
    print(f"Total pseudopotentials: {database['metadata']['total_pseudopotentials']}")
    print(f"Saved to: {output_path}")

if __name__ == '__main__':
    create_pp_database()
