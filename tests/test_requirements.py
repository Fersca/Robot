from pathlib import Path


def test_os_specific_requirements_files_exist():
    assert Path("requirements-windows.txt").exists()
    assert Path("requirements-linux.txt").exists()


def test_windows_requirements_include_pywin32():
    content = Path("requirements-windows.txt").read_text(encoding="utf-8")
    assert "pywin32" in content


def test_linux_requirements_do_not_include_pywin32():
    content = Path("requirements-linux.txt").read_text(encoding="utf-8")
    assert "pywin32" not in content
