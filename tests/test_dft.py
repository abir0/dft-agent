#!/usr/bin/env python3
"""
Test script for DFT calculations with ASE and Quantum ESPRESSO
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_dft_calculator():
    """Test the DFT calculator with a hydrogen atom."""
    try:
        from backend.agents.dft_tools.dft_calculator import test_hydrogen_atom
        
        print("Testing DFT calculator with hydrogen atom...")
        print("=" * 50)
        
        # Test with different QE command paths
        qe_commands = [
            "/Users/o-viejayzorilla/miniconda3/bin/pw.x",
            "pw.x",
            "/usr/local/bin/pw.x",
        ]
        
        for qe_cmd in qe_commands:
            print(f"\nTrying QE command: {qe_cmd}")
            try:
                result = test_hydrogen_atom.func(qe_command=qe_cmd)
                print("SUCCESS!")
                print(result)
                return True
            except Exception as e:
                print(f"Failed with {qe_cmd}: {str(e)}")
                continue
        
        print("\nAll QE commands failed. Let's try with EMT calculator instead...")
        
        # Test with EMT calculator as fallback
        from backend.agents.dft_tools.ase_tools import geometry_optimization
        
        # Create a simple hydrogen structure
        from ase import Atoms
        from ase.io import write
        
        atoms = Atoms('H', positions=[(0, 0, 0)], cell=[10, 10, 10], pbc=True)
        test_file = "data/outputs/calculations/test_results/hydrogen_test.cif"
        write(test_file, atoms)
        
        result = geometry_optimization.func(
            structure_file=test_file,
            calculator="emt",
            relax_type="positions"
        )
        
        print("SUCCESS with EMT calculator!")
        print(result)
        return True
        
    except Exception as e:
        print(f"Error in DFT test: {str(e)}")
        return False

def test_structure_generation():
    """Test structure generation tools."""
    try:
        from backend.agents.dft_tools.structure_tools import generate_bulk
        
        print("\nTesting structure generation...")
        print("=" * 50)
        
        result = generate_bulk.func(
            element="H",
            crystal="diamond",
            a=3.0
        )
        
        print("SUCCESS!")
        print(result)
        return True
        
    except Exception as e:
        print(f"Error in structure generation test: {str(e)}")
        return False

def test_pseudopotential_lookup():
    """Test pseudopotential lookup."""
    try:
        from backend.agents.dft_tools.pymatgen_tools import find_pseudopotentials
        
        print("\nTesting pseudopotential lookup...")
        print("=" * 50)
        
        result = find_pseudopotentials.func(['H'], 'paw', 'psl', 'pbe')
        
        print("SUCCESS!")
        print(result)
        return True
        
    except Exception as e:
        print(f"Error in pseudopotential test: {str(e)}")
        return False

if __name__ == "__main__":
    print("DFT Agent Test Suite")
    print("=" * 50)
    
    # Test pseudopotential lookup first
    test_pseudopotential_lookup()
    
    # Test structure generation
    test_structure_generation()
    
    # Test DFT calculator
    test_dft_calculator()
    
    print("\nTest suite completed!")
