# dft-agent

An LLM Agent that can perform materials research and DFT calculations autonomously and share the results.

For github issue 3:
```python
python -m backend.graph.qe_workflow \
 --structure_path data/slab.traj \
 --pseudo_dir ./pseudos/PBE \
 --workdir runs/pt111_co \
 --run_mode local \
 --thread_id pt111_co
```
