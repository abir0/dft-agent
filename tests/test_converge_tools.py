# tests/test_vasp_converge.py
from pathlib import Path
import pytest
import backend.agents.dft_tools.vasp_convergence as conv_mod  # adjust import to your path

class StubInvokeTool:
    def __init__(self, returns=None, seq=None):
        self.calls = []
        self._returns = returns
        self._seq = iter(seq) if seq is not None else None
    def invoke(self, args):
        self.calls.append(args)
        if self._seq is not None:
            return next(self._seq)
        return self._returns

def _poscar_text():
    return """Si
1.0
5.430000 0.000000 0.000000
0.000000 5.430000 0.000000
0.000000 0.000000 5.430000
Si
2
Direct
0.000000 0.000000 0.000000
0.250000 0.250000 0.250000
"""

@pytest.fixture
def tmp_struct(tmp_path: Path):
    p = tmp_path / "POSCAR_Si"
    p.write_text(_poscar_text())
    return p

def test_converge_encut_path(monkeypatch, tmp_path, tmp_struct):
    # stubs
    scf_stub   = StubInvokeTool(returns=str(tmp_path / "encut_conv/ecut_XXX"))  # value unused by code
    run_stub   = StubInvokeTool(returns=None)
    energy_stub= StubInvokeTool(seq=[-10.000, -10.003, -10.005])  # eV sequence

    # swap the whole tool objects in the module namespace
    monkeypatch.setattr(conv_mod, "write_vasp_scf", scf_stub, raising=False)
    monkeypatch.setattr(conv_mod, "run_local", run_stub, raising=False)
    monkeypatch.setattr(conv_mod, "parse_vasp_energy", energy_stub, raising=False)

    res = conv_mod.converge_encut.invoke({
        "struct": str(tmp_struct),
        "workdir": str(tmp_path / "encut_conv"),
        "encut_list": [500, 400, 400, 300],
        "kpts": "2 2 2",
        "natoms": 2,
        "tol_mev": 1.0,
        "ismear": 0,
        "sigma": 0.05,
    })

    assert res["grid"] == [300, 400, 500]
    assert res["energies_ev"] == pytest.approx([-10.000, -10.003, -10.005])
    assert res["deltas_mev"] == pytest.approx([1.5, 1.0])
    assert res["best_index"] == 2 and res["best_value"] == 500

    # sanity on calls
    assert [Path(c["workdir"]).name for c in scf_stub.calls] == ["ecut_300", "ecut_400", "ecut_500"]
    assert all(c["kpts"] == "2x2x2" for c in scf_stub.calls)
    assert all(c["cmd"] == "vasp_std" for c in run_stub.calls)

def test_converge_kpoints_path(monkeypatch, tmp_path, tmp_struct):
    scf_stub    = StubInvokeTool(returns=str(tmp_path / "k_conv/k_XXX"))
    run_stub    = StubInvokeTool(returns=None)
    energy_stub = StubInvokeTool(seq=[-20.000, -20.001, -20.0014])

    monkeypatch.setattr(conv_mod, "write_vasp_scf", scf_stub, raising=False)
    monkeypatch.setattr(conv_mod, "run_local", run_stub, raising=False)
    monkeypatch.setattr(conv_mod, "parse_vasp_energy", energy_stub, raising=False)

    res = conv_mod.converge_kpoints.invoke({
        "struct": str(tmp_struct),
        "workdir": str(tmp_path / "k_conv"),
        "kpt_list": ["2 2 2", [3,3,3], "4x4x4"],
        "encut": 450,
        "natoms": 4,
        "tol_mev": 0.5,
        "ismear": 0,
        "sigma": 0.05,
    })

    assert res["grid"] == ["2x2x2", "3x3x3", "4x4x4"]
    assert res["energies_ev"] == pytest.approx([-20.000, -20.001, -20.0014])
    assert res["deltas_mev"] == pytest.approx([0.25, 0.10])
    assert res["best_index"] == 1 and res["best_value"] == "3x3x3"

    assert [Path(c["workdir"]).name for c in scf_stub.calls] == ["k_2_2_2", "k_3_3_3", "k_4_4_4"]
    assert [c["kpts"] for c in scf_stub.calls] == ["2x2x2", "3x3x3", "4x4x4"]
    assert all(c["encut"] == 450 for c in scf_stub.calls)
    assert all(c["cmd"] == "vasp_std" for c in run_stub.calls)
