from pathlib import Path


def test_os_specific_requirements_files_exist():
    assert Path("requirements-windows.txt").exists()
    assert Path("requirements-linux.txt").exists()
    assert Path("requirements-tada.txt").exists()
    assert Path("vision_models.json").exists()


def test_windows_requirements_include_pywin32():
    content = Path("requirements-windows.txt").read_text(encoding="utf-8")
    assert "pywin32" in content


def test_linux_requirements_do_not_include_pywin32():
    content = Path("requirements-linux.txt").read_text(encoding="utf-8")
    assert "pywin32" not in content


def test_requirements_include_silero_vad():
    assert "silero-vad" in Path("requirements-windows.txt").read_text(encoding="utf-8")
    assert "silero-vad" in Path("requirements-linux.txt").read_text(encoding="utf-8")


def test_requirements_include_pyside6():
    assert "PySide6" in Path("requirements-windows.txt").read_text(encoding="utf-8")
    assert "PySide6" in Path("requirements-linux.txt").read_text(encoding="utf-8")


def test_requirements_do_not_include_parler_tts():
    assert "parler-tts" not in Path("requirements-windows.txt").read_text(encoding="utf-8")
    assert "parler-tts" not in Path("requirements-linux.txt").read_text(encoding="utf-8")


def test_requirements_include_setuptools():
    assert "setuptools<82" in Path("requirements-windows.txt").read_text(encoding="utf-8")
    assert "setuptools<82" in Path("requirements-linux.txt").read_text(encoding="utf-8")


def test_optional_tada_requirements_file_contains_tada_stack():
    tada = Path("requirements-tada.txt").read_text(encoding="utf-8")
    assert "torchaudio" in tada
    assert "github.com/HumeAI/tada.git" in tada
