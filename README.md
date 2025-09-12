# dft-agent

An LLM Agent that can perform materials research and DFT calculations autonomously and share the results.

## Dataset Location

**Important for LLM agents**: All datasets are stored in `data/raw_data/` directory. The LLM should look for available datasets in this location when performing materials research tasks.

Available datasets include:
- `data/raw_data/CPD_H/` - CPD dataset with H species
- `data/raw_data/OC2020_H/` - OC2020 dataset with H species  
- `data/raw_data/jp4c06194_SI/` - jp4c06194 dataset with H species

### Working with Adsorption Data

Use this tested code pattern to load and analyze adsorption data:

```python
import json

# Load adsorption data
with open('data/raw_data/CPD_H/adsorption_data.json', 'r') as f:
    data = json.load(f)

# Extract all records (data structure: [{\"data\": [records...]}])
all_records = []
for entry in data:
    if isinstance(entry, dict) and 'data' in entry:
        all_records.extend(entry['data'])

# Example: Find hydrogen-containing adsorbates
hydrogen_records = []
for record in all_records:
    formula = record['adsorbate']['formula']
    if 'H' in formula:
        hydrogen_records.append({
            'formula': formula,
            'energy': record['adsorption_energy'],
            'metal': record['adsorbent']['composition'],
            'surface': record['adsorbent']['surface']
        })

print(f'Found {len(hydrogen_records)} hydrogen entries')
```
