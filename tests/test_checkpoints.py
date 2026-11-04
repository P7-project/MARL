from pathlib import Path

from marl import checkpoints


def test_bare_checkpoint_filename_uses_checkpoints_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(checkpoints, "PROJECT_ROOT", tmp_path)

    output_path = checkpoints.checkpoint_output_path("policy.pt")

    assert output_path == tmp_path / "checkpoints" / "policy.pt"
    assert output_path.parent.is_dir()


def test_explicit_checkpoint_parent_is_preserved_and_created(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    output_path = checkpoints.checkpoint_output_path("models/policy.pt")

    assert output_path == Path("models/policy.pt")
    assert (tmp_path / output_path.parent).is_dir()
