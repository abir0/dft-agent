# Raw Data Documentation

This directory contains three adsorption energy datasets from different sources, each with its own schema and data structure. This document provides comprehensive information for AI agents to understand and work with these datasets.

## Dataset Overview

| Dataset | Source | Entries | Size | Description |
|---------|--------|---------|------|-------------|
| CPD_H | Catalysis-hub | ~2,500+ | 3.6MB | Large collection of adsorption data from catalysis-hub database |
| OC2020_H | Open Catalyst Project 2020 | ~130 | 250KB | Hydrogen adsorption energies on various surfaces |
| jp4c06194_SI | Literature | 281 | 290KB | Adsorption data from 2024 research publication |

## Dataset Schemas

### 1. CPD_H Dataset (`CPD_H/adsorption_data.json`)

**Structure**: Array of objects, each containing a `data` array with adsorption entries.

```json
[
  {
    "data": [
      {
        "source": "catalysis-hub",
        "entry_id": null,
        "adsorbate": {
          "formula": "O",
          "fractional_coverage": "1/4"
        },
        "adsorbent": {
          "composition": "Cu",
          "surface": "111",
          "miller_indices": "(111)",
          "adsorption_site": "fcc",
          "space_group": "Fm3m",
          "is_most_stable_site": true
        },
        "adsorption_energy": -4.29,
        "units": "eV/f.u.",
        "computational_settings": {
          "kpoint_mesh": null,
          "xc_functional": "PW91",
          "energy_cutoff": null,
          "force_cutoff": null,
          "convergence_criteria": null
        },
        "experimental_conditions": {
          "temperature": null,
          "pressure": null
        },
        "doi": "10.1016/S0039-6028(01)01464-9",
        "contributors": {
          "email": "Tuong.Bui@nrel.gov",
          "last_name": "Bui",
          "first_name": "Tuong",
          "affiliation": "NREL"
        },
        "year": null
      }
    ]
  }
]
```

**Key Features:**
- Nested structure with multiple entries per top-level object
- Detailed adsorbate information with coverage fractions
- Complete adsorbent characterization including space groups
- Contributor information and DOI references
- Mix of theoretical and experimental data points

### 2. OC2020_H Dataset (`OC2020_H/adsorption_data.json`)

**Structure**: Single object with a `data` array containing all entries.

```json
{
  "data": [
    {
      "source": "open_catalyst_project_2020",
      "entry_id": "61690980998491c1d0883984",
      "adsorbate": "*H",
      "adsorbent": {
        "composition": "Ag10Sb10Se20",
        "surface": "212",
        "miller_indices": [2, 1, 2],
        "adsorption_site": [[-0.52, 10.11, 25.25]]
      },
      "adsorption_energy": 0.422505,
      "units": "eV/f.u.",
      "computational_settings": {
        "kpoint_mesh": null,
        "xc_functional": "RPBE",
        "energy_cutoff": "350",
        "force": "0.03",
        "convergence_criteria": "10e-4"
      },
      "experimental_conditions": {
        "temperature": null,
        "pressure": null
      },
      "metadata": {
        "doi": "10.1021/acscatal.0c04525",
        "authors": "Chanussot*, Lowik and Das*, Abhishek...",
        "year": "2021"
      }
    }
  ]
}
```

**Key Features:**
- Focus on hydrogen adsorption (*H)
- Unique MongoDB-style entry IDs
- Miller indices as arrays [h, k, l]
- 3D coordinates for adsorption sites
- Detailed computational parameters
- Standardized metadata structure

### 3. jp4c06194_SI Dataset (`jp4c06194_SI/adsorption_data.json`)

**Structure**: Array of objects, each containing a `data` array with adsorption entries.

```json
[
  {
    "data": [
      {
        "source": "literature",
        "entry_id": null,
        "adsorbate": "H",
        "adsorbent": {
          "composition": "V",
          "surface": "001",
          "miller_indices": "(001)",
          "adsorption_site": "top",
          "space_group": null,
          "is_most_stable_site": null
        },
        "adsorption_energy": -0.02,
        "units": "eV/f.u.",
        "computational_settings": {
          "software": "VASP",
          "kpoint_mesh": [3, 3, 1],
          "xc_functional": "PBE",
          "energy_cutoff": "1e-6",
          "force_cutoff": null,
          "convergence_criteria": "1e-5"
        },
        "experimental_conditions": {
          "temperature": null,
          "pressure": null
        },
        "metadata": {
          "doi": "10.1021/acs.jpcc.4c06194",
          "authors": ["Allés, M., Meng, L., Beltrán, I., Fernández, F., & Viñes, F."],
          "year": 2024
        }
      }
    ]
  }
]
```

**Key Features:**
- Recent literature data (2024)
- Focus on hydrogen adsorption
- Explicit software specification (VASP)
- Detailed k-point mesh information
- Authors as array structure

## Additional Files

### jp4c06194_SI/jp4c06194_si_001.pdf
- **Type**: PDF document (10 pages, 360KB)
- **Content**: Supporting information for the research publication
- **Purpose**: Contains detailed methodology and supplementary data

## Schema Field Definitions

### Common Fields Across All Datasets

| Field | Type | Description |
|-------|------|-------------|
| `source` | string | Data source identifier |
| `entry_id` | string/null | Unique identifier (when available) |
| `adsorbate` | string/object | Adsorbed species information |
| `adsorbent` | object | Surface/catalyst information |
| `adsorption_energy` | number | Binding energy value |
| `units` | string | Energy units (typically "eV/f.u.") |
| `computational_settings` | object | DFT calculation parameters |
| `experimental_conditions` | object | Temperature/pressure conditions |

### Dataset-Specific Variations

#### CPD_H Specific:
- `adsorbate.fractional_coverage`: Coverage information
- `contributors`: Detailed contributor information
- `doi`: Direct DOI string

#### OC2020_H Specific:
- `miller_indices`: Array format [h, k, l]
- `adsorption_site`: 3D coordinates
- `metadata`: Structured metadata object

#### jp4c06194_SI Specific:
- `computational_settings.software`: Explicit software name
- `metadata.authors`: Array of author strings
- `kpoint_mesh`: Array format

## Usage Guidelines for AI Agents

### Data Access Patterns
1. **CPD_H**: Iterate through top-level array, then through each object's `data` array
2. **OC2020_H**: Access single `data` array directly
3. **jp4c06194_SI**: Similar to CPD_H structure

### Key Considerations
- Handle null values gracefully across all datasets
- Energy units are consistently "eV/f.u." but verify per entry
- Miller indices format varies (string vs array)
- Adsorption site representation differs significantly
- Computational settings completeness varies

### Common Analysis Tasks
- Filter by adsorbate type (H, O, O2, etc.)
- Group by surface composition/orientation
- Compare computational settings across sources
- Analyze energy distributions by material type

## Data Quality Notes

- Some JSON files may have formatting issues requiring robust parsing
- Missing values are typically represented as `null`
- Computational settings completeness varies by source
- Entry counts: jp4c06194_SI (281), others estimated from file sizes

## References

- **CPD_H**: Catalysis-hub database entries
- **OC2020_H**: Chanussot et al., ACS Catalysis, 2021, DOI: 10.1021/acscatal.0c04525
- **jp4c06194_SI**: Allés et al., J. Phys. Chem. C, 2024, DOI: 10.1021/acs.jpcc.4c06194