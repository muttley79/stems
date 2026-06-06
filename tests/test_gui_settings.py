"""Tests for the GUI's tiny persisted settings store."""

from stems.gui import settings


def test_missing_file_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("STEMS_GUI_CONFIG", str(tmp_path / "gui.json"))
    assert settings.load_settings() == {}


def test_round_trip(tmp_path, monkeypatch):
    cfg = tmp_path / "nested" / "gui.json"  # parent dir created on save
    monkeypatch.setenv("STEMS_GUI_CONFIG", str(cfg))
    settings.save_settings({"input_dir": r"D:\music", "output_dir": r"D:\out"})
    assert cfg.is_file()
    loaded = settings.load_settings()
    assert loaded["input_dir"] == r"D:\music"
    assert loaded["output_dir"] == r"D:\out"


def test_corrupt_file_is_ignored(tmp_path, monkeypatch):
    cfg = tmp_path / "gui.json"
    cfg.write_text("{ not valid json", "utf-8")
    monkeypatch.setenv("STEMS_GUI_CONFIG", str(cfg))
    assert settings.load_settings() == {}
