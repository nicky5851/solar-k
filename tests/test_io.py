from pathlib import Path

from solar_k import io as skio


def test_output_root_defaults_to_repo_root_outputs(monkeypatch):
    monkeypatch.delenv("SOLAR_K_OUTPUT", raising=False)
    assert skio.output_root() == skio.repo_root() / "outputs"


def test_output_root_respects_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("SOLAR_K_OUTPUT", str(tmp_path / "kaggle_working" / "outputs"))
    assert skio.output_root() == (tmp_path / "kaggle_working" / "outputs").resolve()


def test_cache_root_defaults_to_data_root_processed(monkeypatch):
    monkeypatch.delenv("SOLAR_K_CACHE", raising=False)
    monkeypatch.delenv("SOLAR_K_DATA", raising=False)
    assert skio.cache_root() == skio.data_root() / "processed"


def test_cache_root_respects_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("SOLAR_K_CACHE", str(tmp_path / "kaggle_temp" / "cache"))
    assert skio.cache_root() == (tmp_path / "kaggle_temp" / "cache").resolve()
