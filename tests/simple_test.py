#!/usr/bin/env python3
"""
Simple test for DFT calculations with EMT calculator
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_emt_calculator():
    """Test DFT calculation with EMT calculator."""
    try:
        from backend.agents.dft_tools.ase_tools import geometry_optimization
        
        print("Testing EMT calculator with hydrogen atom...")
        print("=" * 50)
        
        # Create a simple hydrogen structure
        from ase import Atoms
        from ase.io import write
        
        atoms = Atoms('H', positions=[(0, 0, 0)], cell=[10, 10, 10], pbc=True)
        test_file = "data/outputs/calculations/test_results/hydrogen_emt_test.cif"
        write(test_file, atoms)
        
        result = geometry_optimization.func(
            structure_file=test_file,
            calculator="emt",
            relax_type="positions",
            force_tolerance=0.01
        )
        
        print("SUCCESS with EMT calculator!")
        print(result)
        return True
        
    except Exception as e:
        print(f"Error in EMT test: {str(e)}")
        return False

def test_structure_workflow():
    """Test complete structure generation and optimization workflow."""
    try:
        from backend.agents.dft_tools.structure_tools import generate_bulk
        from backend.agents.dft_tools.ase_tools import geometry_optimization
        
        print("\nTesting complete workflow...")
        print("=" * 50)
        
        # Generate hydrogen bulk structure
        print("1. Generating hydrogen bulk structure...")
        bulk_result = generate_bulk.func(
            element="H",
            crystal="diamond",
            a=3.0
        )
        print(bulk_result)
        
        # Find the generated file
        import re
        bulk_file = re.search(r'Saved as (.+)', bulk_result).group(1)
        print(f"Generated file: {bulk_file}")
        
        # Optimize the structure
        print("\n2. Optimizing structure with EMT...")
        opt_result = geometry_optimization.func(
            structure_file=bulk_file,
            calculator="emt",
            relax_type="positions",
            force_tolerance=0.01
        )
        print(opt_result)
        
        print("\nSUCCESS! Complete workflow tested!")
        return True
        
    except Exception as e:
        print(f"Error in workflow test: {str(e)}")
        return False

if __name__ == "__main__":
    print("Simple DFT Agent Test")
    print("=" * 50)
    
    # Test EMT calculator
    test_emt_calculator()
    
    # Test complete workflow
    test_structure_workflow()
    
    print("\nSimple test completed!")
