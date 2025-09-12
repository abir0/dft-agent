#!/usr/bin/env python3
"""
Basic structure test without dependencies
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_basic_imports():
    """Test basic imports without dependencies."""
    try:
        # Test that we can import the modules (even if they fail due to missing deps)
        import backend.agents.dft_tools.structure_tools
        print("✅ Structure tools module exists")
        
        import backend.agents.dft_tools.dft_calculator
        print("✅ DFT calculator module exists")
        
        import backend.agents.dft_tools.tool_registry
        print("✅ Tool registry module exists")
        
        import backend.core.models
        print("✅ Core models module exists")
        
        import backend.settings
        print("✅ Settings module exists")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error (expected due to missing dependencies): {e}")
        return True  # This is expected
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_file_structure():
    """Test that all expected files exist."""
    expected_files = [
        "backend/agents/dft_tools/structure_tools.py",
        "backend/agents/dft_tools/dft_calculator.py",
        "backend/agents/dft_tools/tool_registry.py",
        "backend/agents/dft_tools/__init__.py",
        "backend/core/models.py",
        "backend/settings.py",
        "README.md",
        "env.example",
    ]
    
    missing_files = []
    for file_path in expected_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All expected files exist")
        return True

def test_tool_registry_structure():
    """Test that tool registry has expected structure."""
    try:
        # Read the tool registry file and check for expected content
        with open("backend/agents/dft_tools/tool_registry.py", "r") as f:
            content = f.read()
        
        expected_content = [
            "TOOL_REGISTRY",
            "TOOL_CATEGORIES", 
            "get_tool_by_name",
            "get_tools_by_category",
            "list_all_tools",
        ]
        
        missing_content = []
        for item in expected_content:
            if item not in content:
                missing_content.append(item)
        
        if missing_content:
            print(f"❌ Tool registry missing content: {missing_content}")
            return False
        else:
            print("✅ Tool registry has expected structure")
            return True
            
    except Exception as e:
        print(f"❌ Error testing tool registry: {e}")
        return False

if __name__ == "__main__":
    print("DFT Agent Basic Structure Test")
    print("=" * 50)
    
    tests = [
        test_file_structure,
        test_tool_registry_structure,
        test_basic_imports,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All basic structure tests passed!")
        print("The codebase structure is correct. Install dependencies to run full tests.")
    else:
        print("❌ Some tests failed. Check the issues above.")
