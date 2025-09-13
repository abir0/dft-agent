import math
import re
import json
from typing import Annotated, Dict, Any, Optional, List
from pathlib import Path

import numexpr
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL

from backend.agents.llm import settings


@tool
def calculator(expression: str) -> str:
    """Calculates a math expression using numexpr.

    Useful for when you need to answer questions about math using numexpr.
    This tool is only for math questions and nothing else. Only input
    math expressions.

    Args:
        expression (str): A valid numexpr formatted math expression.

    Returns:
        str: The result of the math expression.
    """

    try:
        local_dict = {"pi": math.pi, "e": math.e}
        output = str(
            numexpr.evaluate(
                expression.strip(),
                global_dict={},  # restrict access to globals
                local_dict=local_dict,  # add common mathematical functions
            )
        )
        return re.sub(r"^\[|\]$", "", output)
    except Exception as e:
        raise ValueError(
            f'calculator("{expression}") raised error: {e}.'
            " Please try again with a valid numerical expression"
        )


@tool
def python_repl(code: Annotated[str, "Python code or filename to read the code from"]):
    """Use this tool to execute python code. Make sure that you input the code correctly.
    Either input actual code or filename of the code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user.
    """

    try:
        result = PythonREPL().run(code)
        print("RESULT CODE EXECUTION:", result)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return f"Executed:\n```python\n{code}\n```\nStdout: {result}"


@tool
def load_local_adsorption_data(dataset_name: str = "CPD_H") -> Dict[str, Any]:
    """
    Load adsorption data from local JSON datasets.
    
    Args:
        dataset_name: Name of the dataset to load. Options: "CPD_H", "OC2020_H", "jp4c06194_SI"
    
    Returns:
        Dictionary with dataset info and all adsorption records
    """
    datasets_dir = Path(settings.ROOT_PATH) / "data" / "raw_data"
    
    valid_datasets = ["CPD_H", "OC2020_H", "jp4c06194_SI"]
    if dataset_name not in valid_datasets:
        return {"error": f"Invalid dataset. Choose from: {valid_datasets}"}
    
    dataset_path = datasets_dir / dataset_name / "adsorption_data.json"
    
    if not dataset_path.exists():
        return {"error": f"Dataset file not found: {dataset_path}"}
    
    try:
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        # Extract all records from the nested structure
        all_records = []
        for entry in data:
            if isinstance(entry, dict) and 'data' in entry:
                all_records.extend(entry['data'])
        
        return {
            "dataset_name": dataset_name,
            "dataset_path": str(dataset_path),
            "total_records": len(all_records),
            "records": all_records
        }
    
    except Exception as e:
        return {"error": f"Failed to load dataset: {str(e)}"}


"""
Adsorption Database Search Tools

These tools enable benchmarking of DFT calculations against literature databases.
Supports searching across multiple datasets with different data formats for validation
of calculated adsorption energies.
"""
@tool
def search_adsorption_data(
    adsorbate: Optional[str] = None,
    adsorbent_composition: Optional[str] = None,
    surface: Optional[str] = None,
    dataset_name: str = "CPD_H"
) -> Dict[str, Any]:
    """
    Search for specific adsorption data in local datasets.
    
    Args:
        adsorbate: Adsorbate formula to search for (e.g., "H", "OH", "O")
        adsorbent_composition: Metal composition to search for (e.g., "Pt", "Cu", "Ni")
        surface: Surface type to search for (e.g., "fcc111", "fcc100")
        dataset_name: Dataset to search in. Options: "CPD_H", "OC2020_H", "jp4c06194_SI"
    
    Returns:
        Dictionary with matching records and summary statistics
    """
    # Load the dataset
    dataset_result = load_local_adsorption_data(dataset_name)
    
    if "error" in dataset_result:
        return dataset_result
    
    records = dataset_result["records"]
    filtered_records = []
    
    # Apply filters
    for record in records:
        # Check adsorbate
        if adsorbate:
            adsorbate_data = record.get('adsorbate', '')
            # Handle both object format and string format across datasets
            if isinstance(adsorbate_data, dict):
                formula = adsorbate_data.get('formula', '')
            else:
                formula = str(adsorbate_data)
            if not formula or adsorbate not in formula:
                continue
            
        # Check adsorbent composition
        if adsorbent_composition:
            composition = record.get('adsorbent', {}).get('composition', '')
            if not composition or adsorbent_composition not in composition:
                continue
        
        # Check surface
        if surface:
            record_surface = record.get('adsorbent', {}).get('surface', '')
            if not record_surface or surface.lower() not in record_surface.lower():
                continue
        
        filtered_records.append(record)
    
    # Calculate statistics
    if filtered_records:
        energies = [r.get('adsorption_energy', 0) for r in filtered_records if 'adsorption_energy' in r]
        avg_energy = sum(energies) / len(energies) if energies else None
        min_energy = min(energies) if energies else None
        max_energy = max(energies) if energies else None
    else:
        avg_energy = min_energy = max_energy = None
    
    return {
        "dataset_name": dataset_name,
        "search_criteria": {
            "adsorbate": adsorbate,
            "adsorbent_composition": adsorbent_composition,
            "surface": surface
        },
        "total_matches": len(filtered_records),
        "average_adsorption_energy": avg_energy,
        "min_adsorption_energy": min_energy,
        "max_adsorption_energy": max_energy,
        "matching_records": filtered_records[:20] if len(filtered_records) > 20 else filtered_records  # Limit to first 20
    }


@tool
def list_available_datasets() -> Dict[str, Any]:
    """
    List all available local adsorption datasets.
    
    Returns:
        Information about available datasets and their contents
    """
    datasets_dir = Path(settings.ROOT_PATH) / "data" / "raw_data"
    available_datasets = []
    
    for dataset_name in ["CPD_H", "OC2020_H", "jp4c06194_SI"]:
        dataset_path = datasets_dir / dataset_name / "adsorption_data.json"
        
        if dataset_path.exists():
            try:
                with open(dataset_path, 'r') as f:
                    data = json.load(f)
                
                # Count records
                total_records = 0
                for entry in data:
                    if isinstance(entry, dict) and 'data' in entry:
                        total_records += len(entry['data'])
                
                available_datasets.append({
                    "name": dataset_name,
                    "path": str(dataset_path),
                    "total_records": total_records,
                    "description": f"{dataset_name} dataset with hydrogen species adsorption data"
                })
            except Exception as e:
                available_datasets.append({
                    "name": dataset_name,
                    "path": str(dataset_path),
                    "error": f"Failed to read: {str(e)}"
                })
    
    return {
        "available_datasets": available_datasets,
        "total_datasets": len(available_datasets)
    }
