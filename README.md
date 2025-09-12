# dft-agent

An LLM Agent that can perform materials research and DFT calculations autonomously and share the results.

## Dataset Location

**Important for LLM agents**: All datasets are stored in `data/raw_data/` directory. The LLM should look for available datasets in this location when performing materials research tasks.

Available datasets include:
- `data/raw_data/CPD_H/` - CPD dataset with H species
- `data/raw_data/OC2020_H/` - OC2020 dataset with H species  
- `data/raw_data/jp4c06194_SI/` - Supporting information dataset
