# Testing Guide

## Overview

The DFT Agent includes comprehensive testing frameworks to ensure reliability and functionality of all components.

## Test Suites

### 1. Basic Test Suite (`tests/test_dft_tools_basic.py`)

**Purpose**: Tests core functionality without requiring full DFT calculations

**Runtime**: ~5 seconds  
**Success Rate**: 96.6% (28/29 tests passed)

**Test Categories:**
- Module imports (8 tests)
- Tool registry functionality (7 tests)
- Structure operations (2 tests)
- Database operations (2 tests)
- Configuration validation (3 tests)
- File I/O operations (3 tests)
- Environment setup (3 tests)

### 2. Comprehensive Test Suite (`tests/test_dft_tools_comprehensive.py`)

**Purpose**: Tests all DFT tools with actual calculations and real data

**Runtime**: 10-30 minutes (depending on DFT software)  
**Expected Success Rate**: 70-90%

**Test Categories:**
- Tool registry (4 tests)
- Structure tools (5 tests)
- DFT calculator tools (4 tests)
- ASE tools (4 tests)
- Pymatgen tools (4 tests)
- Quantum ESPRESSO tools (5 tests)
- Convergence testing (4 tests)
- Database management (7 tests)

## Running Tests

### Prerequisites

```bash
# Activate environment
source .venv/bin/activate
export PYTHONPATH="$(pwd):$PYTHONPATH"
```

### Basic Tests (Recommended First)

```bash
# Run basic functionality tests
python tests/test_dft_tools_basic.py
```

**Expected Output:**
```
🚀 Starting Basic DFT Tools Testing
==================================================

📦 Testing Module Imports...
✅ import_tool_registry - PASSED (4.12s)
✅ import_structure_tools - PASSED (0.00s)
...

📊 Generating Test Report...
==================================================
🧪 BASIC DFT TOOLS TEST SUMMARY
==================================================
Total Tests: 29
✅ Passed: 28
❌ Failed: 1
📈 Success Rate: 96.6%
📄 Report: WORKSPACE/test_outputs/basic_test_report.json
==================================================
```

### Comprehensive Tests

```bash
# Run full DFT tool tests (requires DFT software)
python tests/test_dft_tools_comprehensive.py
```

**Expected Output:**
```
🚀 Starting Comprehensive DFT Tools Testing
============================================================

🔧 Setting up test environment...
✅ Test environment setup complete

🏗️  Creating test structures...
✅ Test structures created

📋 Testing Tool Registry...
🧪 Running test: tool_registry_count
✅ tool_registry_count - PASSED (0.01s)
...

📊 Generating Test Report...
============================================================
🧪 DFT TOOLS TEST SUMMARY
============================================================
Total Tests: 36
✅ Passed: 32
❌ Failed: 4
⏭️  Skipped: 0
📈 Success Rate: 88.9%
📄 Detailed Report: WORKSPACE/test_outputs/test_report_20241201_143022.json
============================================================
```

## Test Results

### Report Files

Both test suites generate detailed JSON reports:

- **Basic Tests**: `WORKSPACE/test_outputs/basic_test_report.json`
- **Comprehensive Tests**: `WORKSPACE/test_outputs/test_report_YYYYMMDD_HHMMSS.json`

### Report Structure

```json
{
  "timestamp": "2024-12-01T14:30:22",
  "total_tests": 36,
  "passed_tests": 32,
  "failed_tests": 4,
  "skipped_tests": 0,
  "test_details": {
    "test_name": {
      "name": "test_name",
      "status": "passed|failed|skipped",
      "duration": 1.23,
      "error": null,
      "output": "Test result output...",
      "args": [...],
      "kwargs": {...}
    }
  },
  "summary": {
    "total_tests": 36,
    "passed_tests": 32,
    "failed_tests": 4,
    "skipped_tests": 0,
    "success_rate": 88.9,
    "timestamp": "2024-12-01T14:30:22"
  }
}
```

### Success Criteria

- **Basic Tests**: ≥80% success rate
- **Comprehensive Tests**: ≥70% success rate (some tests may fail due to missing DFT software)

## Tool-Specific Testing

### Structure Tools Testing

```python
# Test bulk generation
result = generate_bulk.invoke({
    "element": "Si",
    "crystal_structure": "diamond",
    "lattice_parameter": 5.43,
    "output_file": "test_silicon.xyz"
})

# Test slab generation
result = generate_slab.invoke({
    "structure_file": "test_silicon.xyz",
    "miller_indices": [1, 1, 1],
    "layers": 5,
    "vacuum": 10.0
})
```

### DFT Calculator Testing

```python
# Test hydrogen atom calculation
result = test_hydrogen_atom.invoke({
    "output_dir": "test_h_atom"
})

# Test slab relaxation with layer fixing
result = relax_slab_dft.invoke({
    "structure_file": "test_slab.xyz",
    "output_dir": "test_slab_relax",
    "fixed_layers": 1,
    "ecutwfc": 20.0,
    "kpts": [2, 2, 1]
})
```

### Database Testing

```python
# Test database creation
result = create_calculations_database.invoke({
    "database_path": "test.db"
})

# Test calculation storage
result = store_calculation.invoke({
    "database_path": "test.db",
    "calculation_id": "test_1",
    "structure_file": "test.xyz",
    "calculation_type": "dft_optimization",
    "parameters": {"ecutwfc": 30.0},
    "status": "completed",
    "energy": -15.1234
})
```

## Performance Benchmarks

### Expected Performance

| Test Category | Basic Tests | Comprehensive Tests |
|---------------|-------------|-------------------|
| Tool Registry | <1s | <1s |
| Structure Tools | <5s | 30-60s |
| DFT Calculator | N/A | 2-10min |
| ASE Tools | <10s | 1-5min |
| Pymatgen Tools | <5s | 30s-2min |
| QE Tools | <5s | 1-10min |
| Convergence | N/A | 5-20min |
| Database | <2s | <10s |

### Optimization Tips

1. **Parallel Testing**: Run independent tests in parallel
2. **Resource Management**: Limit concurrent DFT calculations
3. **Caching**: Cache test structures and results
4. **Selective Testing**: Run only specific test categories

## Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Error: ModuleNotFoundError: No module named 'backend'
# Solution:
export PYTHONPATH="$(pwd):$PYTHONPATH"
```

#### 2. Missing Dependencies
```bash
# Error: ImportError: No module named 'ase'
# Solution:
uv sync --dev
# or
pip install -e .
```

#### 3. DFT Software Not Found
```bash
# Error: Error: Executable 'pw.x' not found in PATH
# Solution:
# Install Quantum ESPRESSO
sudo apt-get install quantum-espresso
# Or set environment variable
export QE_BIN="/path/to/pw.x"
```

#### 4. API Key Issues
```bash
# Error: ValueError: At least one LLM API key must be provided
# Solution:
# Edit .env file with your API keys
export OPENAI_API_KEY="sk-..."
export MP_API_KEY="your_mp_key"
```

#### 5. Permission Errors
```bash
# Error: PermissionError: [Errno 13] Permission denied
# Solution:
chmod +x scripts/*.sh
chmod +x tests/*.py
```

### Debug Mode

Run tests with verbose output:

```bash
python -u tests/test_dft_tools_basic.py 2>&1 | tee test_output.log
```

## Continuous Integration

### GitHub Actions Example

```yaml
# .github/workflows/test.yml
name: DFT Tools Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - name: Install dependencies
        run: |
          pip install uv
          uv sync --dev
      - name: Run basic tests
        run: |
          source .venv/bin/activate
          export PYTHONPATH="$(pwd):$PYTHONPATH"
          python tests/test_dft_tools_basic.py
```

## Test Development

### Adding New Tests

1. **Create test function**:
```python
def test_new_function():
    """Test new functionality."""
    result = new_function.invoke({
        "param1": "value1",
        "param2": "value2"
    })
    assert "success" in result.lower()
```

2. **Add to test suite**:
```python
def test_new_category(self):
    """Test new category of tools."""
    self.run_test("test_new_function", test_new_function)
```

3. **Update documentation**:
- Add test description to this guide
- Update expected performance benchmarks
- Document any special requirements

### Test Best Practices

1. **Isolation**: Each test should be independent
2. **Cleanup**: Clean up test files after completion
3. **Error Handling**: Test both success and failure cases
4. **Documentation**: Document test purpose and requirements
5. **Performance**: Monitor test execution time

## Conclusion

The testing framework ensures the reliability and functionality of all DFT tools. Start with basic tests to verify the environment, then proceed to comprehensive tests for full validation. The detailed reports help identify and resolve any issues quickly.

For questions or issues, refer to the troubleshooting section or check the generated test reports for specific error details.
