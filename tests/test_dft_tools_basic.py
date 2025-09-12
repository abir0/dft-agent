#!/usr/bin/env python3
"""
Basic DFT Tools Test Suite

This script performs basic functionality tests for DFT tools without requiring
full DFT calculations or external dependencies. It focuses on testing:
- Tool registry functionality
- Basic structure operations
- File I/O operations
- Database operations
- Configuration validation

Usage:
    python test_dft_tools_basic.py

Requirements:
    - Basic Python environment
    - Project dependencies installed
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Test results storage
test_results = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "total_tests": 0,
    "passed_tests": 0,
    "failed_tests": 0,
    "test_details": {},
}


def run_test(test_name: str, test_func, *args, **kwargs) -> Dict[str, Any]:
    """Run a single test and record results."""
    print(f"🧪 Testing: {test_name}")
    
    start_time = time.time()
    test_result = {
        "name": test_name,
        "status": "unknown",
        "duration": 0,
        "error": None,
        "output": None,
    }
    
    try:
        result = test_func(*args, **kwargs)
        test_result["output"] = str(result)[:200] + "..." if len(str(result)) > 200 else str(result)
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


def test_imports():
    """Test that all modules can be imported."""
    print("\n📦 Testing Module Imports...")
    
    # Test core imports
    run_test("import_tool_registry", lambda: __import__("backend.agents.dft_tools.tool_registry"))
    run_test("import_structure_tools", lambda: __import__("backend.agents.dft_tools.structure_tools"))
    run_test("import_dft_calculator", lambda: __import__("backend.agents.dft_tools.dft_calculator"))
    run_test("import_ase_tools", lambda: __import__("backend.agents.dft_tools.ase_tools"))
    run_test("import_pymatgen_tools", lambda: __import__("backend.agents.dft_tools.pymatgen_tools"))
    run_test("import_qe_tools", lambda: __import__("backend.agents.dft_tools.qe_tools"))
    run_test("import_convergence_tools", lambda: __import__("backend.agents.dft_tools.convergence_tools"))
    run_test("import_database_tools", lambda: __import__("backend.agents.dft_tools.database_tools"))


def test_tool_registry():
    """Test tool registry functionality."""
    print("\n📋 Testing Tool Registry...")
    
    try:
        from backend.agents.dft_tools import (
            TOOL_REGISTRY,
            TOOL_CATEGORIES,
            get_tool_by_name,
            get_tools_by_category,
            list_all_tools,
            list_categories,
            get_tool_info,
        )
        
        # Test registry access
        run_test("tool_registry_not_empty", lambda: len(TOOL_REGISTRY) > 0)
        run_test("tool_categories_not_empty", lambda: len(TOOL_CATEGORIES) > 0)
        run_test("list_all_tools", list_all_tools)
        run_test("list_categories", list_categories)
        
        # Test specific tool retrieval
        run_test("get_tool_by_name_valid", lambda: get_tool_by_name("generate_bulk") is not None)
        run_test("get_tool_by_name_invalid", lambda: "Error" in str(get_tool_by_name("nonexistent_tool")))
        run_test("get_tools_by_category", lambda: len(get_tools_by_category("structure_generation")) > 0)
        run_test("get_tool_info", lambda: get_tool_info("generate_bulk") is not None)
        
    except ImportError as e:
        print(f"❌ Failed to import tool registry: {e}")


def test_structure_operations():
    """Test basic structure operations."""
    print("\n🏗️  Testing Structure Operations...")
    
    try:
        from backend.agents.dft_tools.structure_tools import generate_bulk
        
        # Create test directory
        test_dir = project_root / "WORKSPACE" / "test_structures"
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Test bulk generation
        run_test(
            "generate_bulk_silicon",
            generate_bulk.invoke,
            {
                "element": "Si",
                "crystal_structure": "diamond",
                "lattice_parameter": 5.43,
                "output_file": str(test_dir / "silicon_test.xyz")
            }
        )
        
        # Test file creation
        run_test(
            "test_file_creation",
            lambda: (test_dir / "silicon_test.xyz").exists()
        )
        
    except ImportError as e:
        print(f"❌ Failed to import structure tools: {e}")


def test_database_operations():
    """Test database operations."""
    print("\n🗄️  Testing Database Operations...")
    
    try:
        from backend.agents.dft_tools.database_tools import create_calculations_database
        
        # Create test database
        test_db = project_root / "WORKSPACE" / "test_basic.db"
        test_db.parent.mkdir(parents=True, exist_ok=True)
        
        run_test(
            "create_test_database",
            create_calculations_database,
            str(test_db)
        )
        
        # Test database file creation
        run_test(
            "test_database_file_creation",
            lambda: test_db.exists()
        )
        
    except ImportError as e:
        print(f"❌ Failed to import database tools: {e}")


def test_configuration():
    """Test configuration and settings."""
    print("\n⚙️  Testing Configuration...")
    
    try:
        from backend.settings import settings
        from backend.core.models import OpenAIModelName
        
        # Test settings import
        run_test("import_settings", lambda: settings is not None)
        
        # Test model configuration
        run_test("check_default_model", lambda: settings.DEFAULT_MODEL is not None)
        run_test("check_gpt5_available", lambda: OpenAIModelName.GPT_5 in OpenAIModelName)
        
    except ImportError as e:
        print(f"❌ Failed to import settings: {e}")


def test_file_operations():
    """Test file I/O operations."""
    print("\n📁 Testing File Operations...")
    
    # Create test workspace
    workspace = project_root / "WORKSPACE"
    workspace.mkdir(exist_ok=True)
    
    # Test directory creation
    run_test(
        "create_test_directories",
        lambda: all([
            (workspace / "test_structures").mkdir(exist_ok=True),
            (workspace / "test_outputs").mkdir(exist_ok=True),
            (workspace / "test_calculations").mkdir(exist_ok=True),
        ])
    )
    
    # Test file writing
    test_file = workspace / "test_outputs" / "test_file.txt"
    run_test(
        "write_test_file",
        lambda: test_file.write_text("Test content")
    )
    
    # Test file reading
    run_test(
        "read_test_file",
        lambda: test_file.read_text() == "Test content"
    )


def test_environment_setup():
    """Test environment setup."""
    print("\n🌍 Testing Environment Setup...")
    
    # Test Python path
    run_test(
        "check_python_path",
        lambda: str(project_root) in sys.path
    )
    
    # Test workspace environment variable
    os.environ["WORKSPACE_ROOT"] = str(project_root / "WORKSPACE")
    run_test(
        "check_workspace_env_var",
        lambda: "WORKSPACE_ROOT" in os.environ
    )
    
    # Test project structure
    run_test(
        "check_project_structure",
        lambda: all([
            (project_root / "backend").exists(),
            (project_root / "frontend").exists(),
            (project_root / "pyproject.toml").exists(),
        ])
    )


def generate_report():
    """Generate test report."""
    print("\n📊 Generating Test Report...")
    
    total = test_results["total_tests"]
    passed = test_results["passed_tests"]
    failed = test_results["failed_tests"]
    success_rate = (passed / total * 100) if total > 0 else 0
    
    # Save results
    report_file = project_root / "WORKSPACE" / "test_outputs" / "basic_test_report.json"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_file, "w") as f:
        json.dump(test_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*50)
    print("🧪 BASIC DFT TOOLS TEST SUMMARY")
    print("="*50)
    print(f"Total Tests: {total}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {success_rate:.1f}%")
    print(f"📄 Report: {report_file}")
    print("="*50)
    
    if failed > 0:
        print("\n❌ FAILED TESTS:")
        for test_name, details in test_results["test_details"].items():
            if details["status"] == "failed":
                print(f"  - {test_name}: {details['error']}")
    
    return success_rate


def main():
    """Main function to run basic tests."""
    print("🚀 Starting Basic DFT Tools Testing")
    print("="*50)
    
    # Run all test categories
    test_imports()
    test_tool_registry()
    test_structure_operations()
    test_database_operations()
    test_configuration()
    test_file_operations()
    test_environment_setup()
    
    # Generate report
    success_rate = generate_report()
    
    # Exit with appropriate code
    if success_rate >= 80:
        print("\n🎉 Basic tests completed successfully!")
        sys.exit(0)
    else:
        print("\n⚠️  Some basic tests failed. Check the report for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
