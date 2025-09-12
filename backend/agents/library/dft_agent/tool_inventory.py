def tools_text_block() -> str:
    """Return the EXACT list of tools you actually expose at runtime.
    Keep these names in sync with your registry/executor."""
    return """
AVAILABLE TOOLS (exact names; call only these)

IMPORTANT: Any tool argument named 'struct_json' expects the JSON string representation of a crystal structure, often from the output of a previous step. Do not pass file paths to 'struct_json'.

STRUCTURE
- build_structure(source: str, file_content?: str, file_format?: str, spacegroup?: str, species?: [str], lattice_constants?: {a,b,c,alpha,beta,gamma}, frac_coords?: [[f,f,f]], mp_id?: str, api_key?: str, lattice_dict?: {type, params})
- make_supercell(struct_json: str, mult: [int,int,int])
- read_structure_from_file(file_path: str)

QE (SCF/RELAX + parsers)
- write_qe_scf(struct_json: str, workdir: str, kpts: str, ecutwfc: int, ecutrho?: int, occupations?: str, smearing?: str, degauss?: float, spin_polarized?: bool)
- write_qe_relax(struct_json: str, workdir: str, kpts: str, ecutwfc: int, ecutrho?: int, conv_thr?: float, press?: float, cell_dofree?: str)
- parse_qe_energy(workdir: str)
- parse_qe_relaxed_structure(workdir: str)

SCF / RELAX (VASP)
- write_vasp_singlepoint(struct_json: str, workdir: str, kpts: str, encut: int, sigma?: float, ismear?: int, ispin?: int, ediff?: float)
- write_vasp_relax(struct_json: str, workdir: str, kpts: str, encut: int, isif?: int, ibrion?: int, nsw?: int, ediff?: float)
- parse_vasp_energy(workdir: str)
- parse_vasp_relaxed_structure(workdir: str)

SURFACES / ADSORPTION
- make_slab(struct_json: str, miller: [int,int,int], min_slab?: float, min_vac?: float)
- place_adsorbate_on_slab(slab_json: str, ads: str, site: str, height?: float)

NEB / DIFFUSION
- neb_setup(images: [str], workdir: str, spring?: float, nimages?: int)
- neb_run(workdir: str, code_params?: dict)
- neb_parse_results(workdir: str)

Convergence Tools
- converge_encut(struct_json: str, workdir: str, encut_list: [int], kpts: str, tol_mev?: float)
- converge_kpoints(struct_json: str, workdir: str, kpt_list: [str], encut: int, tol_mev?: float)

EXECUTION
- run_local(cmd: str, workdir: str)
""".strip()