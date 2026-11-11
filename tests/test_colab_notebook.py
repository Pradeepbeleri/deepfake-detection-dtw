"""Guard the Colab notebook's interpreter boundary and cell syntax."""
import ast
import json
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / 'notebooks/lavdf_baseline.ipynb'


def code_cells():
    notebook = json.loads(NOTEBOOK.read_text())
    return [''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code']


def test_colab_cells_compile_and_project_subprocesses_use_explicit_environment():
    calls = []
    for index, source in enumerate(code_cells()):
        tree = ast.parse(source, filename=f'notebook cell {index}')
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            command = node.args[0]
            if (isinstance(node.func, ast.Attribute) and
                    isinstance(node.func.value, ast.Name) and node.func.value.id == 'subprocess' and
                    isinstance(command, ast.List) and command.elts and
                    isinstance(command.elts[0], ast.Name) and command.elts[0].id == 'PYTHON'):
                assert any(keyword.arg == 'env' for keyword in node.keywords)
                calls.append(node)
    # Setup, diagnostics, authentication, dataset and model operations must cross this boundary.
    assert len(calls) >= 10


def test_colab_diagnostic_distinguishes_kernel_and_requires_cuda():
    diagnostic_cell = next(s for s in code_cells() if 'DIAGNOSTIC = ' in s)
    assert 'Colab control kernel' in diagnostic_cell
    assert 'Project ML environment' in diagnostic_cell
    tree = ast.parse(diagnostic_cell)
    script = next(ast.literal_eval(node.value) for node in tree.body
                  if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'DIAGNOSTIC' for t in node.targets))
    compile(script, 'project diagnostic', 'exec')
    assert "torch.cuda.is_available()" in script
    assert "device='cuda'" in script
    assert "torch.cuda.synchronize()" in script


def test_source_group_plan_is_explicit_and_identity_sidecar_is_optional():
    sources = code_cells()
    planning = next(s for s in sources if "PLAN_MODE = 'source-group-only'" in s)
    assert "'--source-group-only'" in planning
    assert "'--usage-rights'" in planning
    assert "'--provenance'" in planning
    assert 'identity disjointness is not established' in planning
    assert "'metadata_sha256'" in planning and "'repository_commit'" in planning


def test_training_and_test_reports_are_bound_before_reuse():
    sources = code_cells()
    train = next(s for s in sources if "training_report = REPORTS / 'training.json'" in s)
    evaluate = next(s for s in sources if "report = REPORTS / 'held-out-test.json'" in s)
    for source in (train, evaluate):
        assert "saved.get('checkpoint_sha256') == digest_file(checkpoint)" in source
        assert "saved.get('manifest_sha256') == digest_file(PLAN)" in source


def test_dinov2_checkout_staged_locally_for_smoke_and_extraction():
    sources = code_cells()
    config = next(s for s in sources if "DINO_DRIVE_SOURCE = PRIVATE / 'dinov2-src'" in s)
    assert "DINO_SOURCE = Path('/content/dinov2-src-7764ea0f')" in config
    assert "'deepfake_dtw.stage_dinov2'" in config
    assert "'--source', str(DINO_DRIVE_SOURCE), '--destination', str(DINO_SOURCE)" in config
    smoke = next(s for s in sources if 'MODEL_SMOKE =' in s)
    extraction = next(s for s in sources if "'-m', 'deepfake_dtw.lavdf_recovery'" in s)
    assert 'str(DINO_SOURCE)' in smoke
    assert "'--dinov2-source', str(DINO_SOURCE)" in extraction
    assert 'DINO_DRIVE_SOURCE' not in smoke and 'DINO_DRIVE_SOURCE' not in extraction
