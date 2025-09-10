from typing import Optional, Dict, Any, List
from pymatgen.ext.matproj import MPRester
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import json

def _choose_best(docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Prefer stable, ordered, experimental-like
    docs = [d for d in docs if d.get("is_ordered", True)]
    docs.sort(key=lambda d: (d.get("energy_above_hull", 1e9), not d.get("experimental", False)))
    return docs[0]



def mp_fetch_best_structure(
    api_key: str,
    formula: str,
    spacegroup_number: Optional[int] = None,
    require_stable: bool = True,
) -> Dict[str, Any]:
    """
    Returns {'mp_id': str, 'structure': <Structure as dict>, 'sg': int, 'ehull': float}
    """
    with MPRester(api_key) as mpr:
        # fields minimal for selection
        fields = ["material_id","structure","symmetry","energy_above_hull","is_ordered","theoretical"]
        docs = mpr.summary.search(formula=formula, fields=fields)  # MP v2 summary API
    if require_stable:
        docs = [d for d in docs if (d.get("energy_above_hull") or 0.0) < 0.05] or docs
    if spacegroup_number:
        docs = [d for d in docs if (d.get("symmetry", {}).get("space_group_number") == spacegroup_number)] or docs
    doc = _choose_best(docs)
    s: Structure = doc["structure"]
    # normalize to conventional for readability/consistency
    s_conv = SpacegroupAnalyzer(s, symprec=1e-3, angle_tolerance=0.5).get_conventional_standard_structure()
    return {
        "mp_id": doc["material_id"],
        "structure": s_conv.as_dict(),
        "sg": doc.get("symmetry", {}).get("space_group_number"),
        "ehull": doc.get("energy_above_hull"),
    }