#!/usr/bin/env python3
"""
Comprehensive Test Suite for DFT Tools

This script tests all DFT tools in the system to ensure they are working properly.
It includes tests for structure generation, DFT calculations, ASE tools, pymatgen tools,
Quantum ESPRESSO tools, convergence testing, and database management.

Usage:
    python test_dft_tools_comprehensive.py

Requirements:
    - Environment must be activated (source .venv/bin/activate)
    - API keys must be configured in .env file
    - Test structures must be available in WORKSPACE/test_structures/
"""

import os
import sys
import json
import time
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import all DFT tools
from backend.agents.dft_tools import (
    TOOL_REGISTRY,
    TOOL_CATEGORIES,
    get_tool_by_name,
    get_tools_by_category,
    list_all_tools,
    list_categories,
    get_tool_info,
)

# Import specific tool modules for testing
from backend.agents.dft_tools.structure_tools import (
    generate_bulk,
    create_supercell,
    generate_slab,
    add_adsorbate,
    add_vacuum,
)
from backend.agents.dft_tools.dft_calculator import (
    run_dft_calculation,
    optimize_structure_dft,
    relax_slab_dft,
    test_hydrogen_atom,
)
from backend.agents.dft_tools.ase_tools import (
    get_kpath_bandstructure,
    generate_kpoint_mesh,
    relax_bulk,
    relax_slab,
)
from backend.agents.dft_tools.pymatgen_tools import (
    search_materials_project,
    analyze_crystal_structure,
    find_pseudopotentials,
    calculate_formation_energy,
)
from backend.agents.dft_tools.qe_tools import (
    generate_qe_input,
    submit_local_job,
    check_job_status,
    read_output_file,
    extract_energy,
)
from backend.agents.dft_tools.convergence_tools import (
    kpoint_convergence_test,
    cutoff_convergence_test,
    slab_thickness_convergence,
    vacuum_convergence_test,
)
from backend.agents.dft_tools.database_tools import (
    create_calculations_database,
    store_calculation,
    update_calculation_status,
    store_adsorption_energy,
    query_calculations,
    export_results,
    search_similar_calculations,
)

# Test configuration
TEST_CONFIG = {
    "workspace_root": "WORKSPACE",
    "test_structures_dir": "WORKSPACE/test_structures",
    "test_outputs_dir": "WORKSPACE/test_outputs",
    "test_calculations_dir": "WORKSPACE/test_calculations",
    "test_database": "WORKSPACE/test_calculations.db",
    "timeout": 300,  # 5 minutes per test
    "cleanup_after_test": True,
}

# Test results storage
test_results = {
    "timestamp": datetime.now().isoformat(),
    "total_tests": 0,
    "passed_tests": 0,
    "failed_tests": 0,
    "skipped_tests": 0,
    "test_details": {},
    "summary": {},
}


class TestRunner:
    """Main test runner class for DFT tools."""
    
    def __init__(self):
        self.setup_test_environment()
        
    def setup_test_environment(self):
        """Set up test environment and directories."""
        print("🔧 Setting up test environment...")
        
        # Create test directories
        for dir_name in ["test_structures", "test_outputs", "test_calculations"]:
            dir_path = project_root / TEST_CONFIG["workspace_root"] / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # Set environment variables
        os.environ["WORKSPACE_ROOT"] = str(project_root / TEST_CONFIG["workspace_root"])
        
        print("✅ Test environment setup complete")
        
    def create_test_structures(self):
        """Create test atomic structures for testing."""
        print("🏗️  Creating test structures...")
        
        structures_dir = project_root / TEST_CONFIG["test_structures_dir"]
        
        # Silicon bulk structure
        silicon_bulk = """8
Si bulk structure
Si 0.0 0.0 0.0
Si 1.35 1.35 0.0
Si 1.35 0.0 1.35
Si 0.0 1.35 1.35
Si 2.7 2.7 2.7
Si 4.05 4.05 2.7
Si 4.05 2.7 4.05
Si 2.7 4.05 4.05
"""
        
        with open(structures_dir / "silicon_bulk.xyz", "w") as f:
            f.write(silicon_bulk)
            
        # Graphene slab structure
        graphene_slab = """8
Graphene slab structure
C 0.0 0.0 0.0
C 1.23 0.0 0.0
C 0.615 1.065 0.0
C 1.845 1.065 0.0
C 0.0 0.0 3.35
C 1.23 0.0 3.35
C 0.615 1.065 3.35
C 1.845 1.065 3.35
"""
        
        with open(structures_dir / "graphene_slab.xyz", "w") as f:
            f.write(graphene_slab)
            
        # Hydrogen atom for simple tests
        hydrogen_atom = """1
H atom
H 0.0 0.0 0.0
"""
        
        with open(structures_dir / "hydrogen_atom.xyz", "w") as f:
            f.write(hydrogen_atom)
            
        print("✅ Test structures created")
        
    def run_test(self, test_name: str, test_func, *args, **kwargs) -> Dict[str, Any]:
        """Run a single test and record results."""
        print(f"🧪 Running test: {test_name}")
        
        start_time = time.time()
        test_result = {
            "name": test_name,
            "status": "unknown",
            "duration": 0,
            "error": None,
            "output": None,
            "args": args,
            "kwargs": kwargs,
        }
        
        try:
            # Run the test function
            result = test_func(*args, **kwargs)
            test_result["output"] = result
            test_result["status"] = "passed"
            test_result["duration"] = time.time() - start_time
            
            print(f"✅ {test_name} - PASSED ({test_result['duration']:.2f}s)")
            test_results["passed_tests"] += 1
            
        except Exception as e:
            test_result["error"] = str(e)
            test_result["status"] = "failed"
            test_result["duration"] = time.time() - start_time
            
            print(f"❌ {test_name} - FAILED ({test_result['duration']:.2f}s)")
            print(f"   Error: {e}")
            test_results["failed_tests"] += 1
            
        test_results["total_tests"] += 1
        test_results["test_details"][test_name] = test_result
        
        return test_result
        
    def test_tool_registry(self):
        """Test the tool registry functionality."""
        print("\n📋 Testing Tool Registry...")
        
        # Test registry access
        self.run_test("tool_registry_count", lambda: len(TOOL_REGISTRY))
        self.run_test("tool_categories_count", lambda: len(TOOL_CATEGORIES))
        self.run_test("list_all_tools", list_all_tools)
        self.run_test("list_categories", list_categories)
        
        # Test tool retrieval
        self.run_test("get_tool_by_name", get_tool_by_name, "generate_bulk")
        self.run_test("get_tools_by_category", get_tools_by_category, "Structure Generation & Manipulation")
        self.run_test("get_tool_info", get_tool_info, "generate_bulk")
        
    def test_structure_tools(self):
        """Test structure generation and manipulation tools."""
        print("\n🏗️  Testing Structure Tools...")
        
        # Test bulk generation
        self.run_test(
            "generate_bulk_silicon",
            generate_bulk,
            "Si",
            "diamond",
            lattice_parameter=5.43,
            output_file="WORKSPACE/test_structures/silicon_generated.xyz"
        )
        
        # Test supercell creation
        self.run_test(
            "create_supercell",
            create_supercell,
            "WORKSPACE/test_structures/silicon_bulk.xyz",
            [2, 2, 2],
            output_file="WORKSPACE/test_structures/silicon_supercell.xyz"
        )
        
        # Test slab generation
        self.run_test(
            "generate_slab",
            generate_slab,
            "WORKSPACE/test_structures/silicon_bulk.xyz",
            [1, 1, 0],
            layers=5,
            vacuum=10.0,
            output_file="WORKSPACE/test_structures/silicon_slab.xyz"
        )
        
        # Test adsorbate addition
        self.run_test(
            "add_adsorbate",
            add_adsorbate,
            "WORKSPACE/test_structures/graphene_slab.xyz",
            "H",
            [0.0, 0.0, 2.0],
            output_file="WORKSPACE/test_structures/graphene_with_h.xyz"
        )
        
        # Test vacuum addition
        self.run_test(
            "add_vacuum",
            add_vacuum,
            "WORKSPACE/test_structures/silicon_slab.xyz",
            15.0,
            output_file="WORKSPACE/test_structures/silicon_slab_vacuum.xyz"
        )
        
    def test_dft_calculator_tools(self):
        """Test DFT calculation tools."""
        print("\n⚛️  Testing DFT Calculator Tools...")
        
        # Test hydrogen atom calculation (simple test)
        self.run_test(
            "test_hydrogen_atom",
            test_hydrogen_atom,
            output_dir="WORKSPACE/test_calculations/h_atom_test"
        )
        
        # Test DFT calculation (if QE is available)
        self.run_test(
            "run_dft_calculation",
            run_dft_calculation,
            "WORKSPACE/test_structures/hydrogen_atom.xyz",
            "WORKSPACE/test_calculations/h_dft_test",
            ecutwfc=20.0,
            kpts=[1, 1, 1]
        )
        
        # Test structure optimization
        self.run_test(
            "optimize_structure_dft",
            optimize_structure_dft,
            "WORKSPACE/test_structures/hydrogen_atom.xyz",
            "WORKSPACE/test_calculations/h_opt_test",
            ecutwfc=20.0,
            kpts=[1, 1, 1]
        )
        
        # Test DFT slab relaxation with layer fixing
        self.run_test(
            "relax_slab_dft",
            relax_slab_dft,
            "WORKSPACE/test_structures/graphene_slab.xyz",
            "WORKSPACE/test_calculations/slab_relax_test",
            fixed_layers=1,
            ecutwfc=20.0,
            kpts=[2, 2, 1]
        )
        
    def test_ase_tools(self):
        """Test ASE-based tools."""
        print("\n🔬 Testing ASE Tools...")
        
        # Test k-point mesh generation
        self.run_test(
            "generate_kpoint_mesh",
            generate_kpoint_mesh,
            "WORKSPACE/test_structures/silicon_bulk.xyz",
            [4, 4, 4],
            output_file="WORKSPACE/test_outputs/silicon_kpoints.txt"
        )
        
        # Test band structure k-path
        self.run_test(
            "get_kpath_bandstructure",
            get_kpath_bandstructure,
            "WORKSPACE/test_structures/silicon_bulk.xyz",
            output_file="WORKSPACE/test_outputs/silicon_kpath.txt"
        )
        
        # Test bulk relaxation
        self.run_test(
            "relax_bulk",
            relax_bulk,
            "WORKSPACE/test_structures/silicon_bulk.xyz",
            ecutwfc=20.0,
            kpts=[2, 2, 2]
        )
        
        # Test slab relaxation
        self.run_test(
            "relax_slab",
            relax_slab,
            "WORKSPACE/test_structures/graphene_slab.xyz",
            fixed_layers=1,
            ecutwfc=20.0,
            kpts=[2, 2, 1]
        )
        
    def test_pymatgen_tools(self):
        """Test pymatgen-based tools."""
        print("\n🧪 Testing Pymatgen Tools...")
        
        # Test materials project search
        self.run_test(
            "search_materials_project",
            search_materials_project,
            "silicon",
            max_results=5
        )
        
        # Test crystal structure analysis
        self.run_test(
            "analyze_crystal_structure",
            analyze_crystal_structure,
            "WORKSPACE/test_structures/silicon_bulk.xyz"
        )
        
        # Test pseudopotential search
        self.run_test(
            "find_pseudopotentials",
            find_pseudopotentials,
            "Si",
            "pbe"
        )
        
        # Test formation energy calculation
        self.run_test(
            "calculate_formation_energy",
            calculate_formation_energy,
            "WORKSPACE/test_structures/silicon_bulk.xyz",
            "WORKSPACE/test_structures/hydrogen_atom.xyz"
        )
        
    def test_qe_tools(self):
        """Test Quantum ESPRESSO tools."""
        print("\n⚡ Testing Quantum ESPRESSO Tools...")
        
        # Test QE input generation
        self.run_test(
            "generate_qe_input",
            generate_qe_input,
            "WORKSPACE/test_structures/hydrogen_atom.xyz",
            "WORKSPACE/test_outputs/h_atom.in",
            ecutwfc=20.0,
            kpts=[1, 1, 1]
        )
        
        # Test job submission (local)
        self.run_test(
            "submit_local_job",
            submit_local_job,
            "WORKSPACE/test_outputs/h_atom.in",
            "WORKSPACE/test_calculations/h_job_test"
        )
        
        # Test job status checking
        self.run_test(
            "check_job_status",
            check_job_status,
            "1",
            "local"
        )
        
        # Test output file reading
        self.run_test(
            "read_output_file",
            read_output_file,
            "WORKSPACE/test_calculations/h_atom_test/output.pwo",
            "qe"
        )
        
        # Test energy extraction
        self.run_test(
            "extract_energy",
            extract_energy,
            "WORKSPACE/test_calculations/h_atom_test/output.pwo",
            "total"
        )
        
    def test_convergence_tools(self):
        """Test convergence testing tools."""
        print("\n📊 Testing Convergence Tools...")
        
        # Test k-point convergence
        self.run_test(
            "kpoint_convergence_test",
            kpoint_convergence_test,
            "WORKSPACE/test_structures/silicon_bulk.xyz",
            "WORKSPACE/test_calculations/kpoint_conv_test",
            kpoint_range=[2, 4, 6, 8],
            ecutwfc=20.0
        )
        
        # Test cutoff convergence
        self.run_test(
            "cutoff_convergence_test",
            cutoff_convergence_test,
            "WORKSPACE/test_structures/silicon_bulk.xyz",
            "WORKSPACE/test_calculations/cutoff_conv_test",
            cutoff_range=[20, 30, 40, 50],
            kpts=[4, 4, 4]
        )
        
        # Test slab thickness convergence
        self.run_test(
            "slab_thickness_convergence",
            slab_thickness_convergence,
            "WORKSPACE/test_structures/silicon_bulk.xyz",
            [1, 1, 0],
            "WORKSPACE/test_calculations/slab_conv_test",
            layer_range=[3, 5, 7, 9],
            vacuum=10.0
        )
        
        # Test vacuum convergence
        self.run_test(
            "vacuum_convergence_test",
            vacuum_convergence_test,
            "WORKSPACE/test_structures/silicon_slab.xyz",
            "WORKSPACE/test_calculations/vacuum_conv_test",
            vacuum_range=[5, 10, 15, 20],
            ecutwfc=20.0,
            kpts=[2, 2, 1]
        )
        
    def test_database_tools(self):
        """Test database management tools."""
        print("\n🗄️  Testing Database Tools...")
        
        # Test database creation
        self.run_test(
            "create_calculations_database",
            create_calculations_database,
            TEST_CONFIG["test_database"]
        )
        
        # Test calculation storage
        self.run_test(
            "store_calculation",
            store_calculation,
            TEST_CONFIG["test_database"],
            "test_calc_1",
            "WORKSPACE/test_structures/silicon_bulk.xyz",
            "dft_optimization",
            {"ecutwfc": 30.0, "kpts": [4, 4, 4]},
            "completed",
            -15.1234
        )
        
        # Test status update
        self.run_test(
            "update_calculation_status",
            update_calculation_status,
            TEST_CONFIG["test_database"],
            "test_calc_1",
            "completed"
        )
        
        # Test adsorption energy storage
        self.run_test(
            "store_adsorption_energy",
            store_adsorption_energy,
            TEST_CONFIG["test_database"],
            "test_ads_1",
            "WORKSPACE/test_structures/graphene_slab.xyz",
            "H",
            -0.5,
            {"site": "top", "height": 2.0}
        )
        
        # Test querying calculations
        self.run_test(
            "query_calculations",
            query_calculations,
            TEST_CONFIG["test_database"],
            calculation_type="dft_optimization"
        )
        
        # Test results export
        self.run_test(
            "export_results",
            export_results,
            TEST_CONFIG["test_database"],
            "WORKSPACE/test_outputs/exported_results.json"
        )
        
        # Test similar calculations search
        self.run_test(
            "search_similar_calculations",
            search_similar_calculations,
            TEST_CONFIG["test_database"],
            "WORKSPACE/test_structures/silicon_bulk.xyz",
            max_results=5
        )
        
    def generate_test_report(self):
        """Generate comprehensive test report."""
        print("\n📊 Generating Test Report...")
        
        # Calculate summary statistics
        total = test_results["total_tests"]
        passed = test_results["passed_tests"]
        failed = test_results["failed_tests"]
        skipped = test_results["skipped_tests"]
        
        success_rate = (passed / total * 100) if total > 0 else 0
        
        test_results["summary"] = {
            "total_tests": total,
            "passed_tests": passed,
            "failed_tests": failed,
            "skipped_tests": skipped,
            "success_rate": success_rate,
            "timestamp": test_results["timestamp"]
        }
        
        # Save detailed results
        report_file = project_root / TEST_CONFIG["test_outputs_dir"] / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, "w") as f:
            json.dump(test_results, f, indent=2)
            
        # Print summary
        print("\n" + "="*60)
        print("🧪 DFT TOOLS TEST SUMMARY")
        print("="*60)
        print(f"Total Tests: {total}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"⏭️  Skipped: {skipped}")
        print(f"📈 Success Rate: {success_rate:.1f}%")
        print(f"📄 Detailed Report: {report_file}")
        print("="*60)
        
        # Print failed tests
        if failed > 0:
            print("\n❌ FAILED TESTS:")
            for test_name, details in test_results["test_details"].items():
                if details["status"] == "failed":
                    print(f"  - {test_name}: {details['error']}")
                    
        return test_results
        
    def run_all_tests(self):
        """Run all DFT tool tests."""
        print("🚀 Starting Comprehensive DFT Tools Testing")
        print("="*60)
        
        # Create test structures
        self.create_test_structures()
        
        # Run all test categories
        self.test_tool_registry()
        self.test_structure_tools()
        self.test_dft_calculator_tools()
        self.test_ase_tools()
        self.test_pymatgen_tools()
        self.test_qe_tools()
        self.test_convergence_tools()
        self.test_database_tools()
        
        # Generate final report
        return self.generate_test_report()


def main():
    """Main function to run all tests."""
    try:
        runner = TestRunner()
        results = runner.run_all_tests()
        
        # Exit with appropriate code
        if results["summary"]["failed_tests"] > 0:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n⚠️  Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error during testing: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
