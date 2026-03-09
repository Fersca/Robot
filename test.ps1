$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

pytest -q `
  -p no:cacheprovider `
  --basetemp .tmp_tests\pytest `
  --ignore pytest-cache-files-6mndw5k6 `
  --ignore pytest-cache-files-io38dnfk
