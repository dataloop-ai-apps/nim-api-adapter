## What was done

Added `agent/requirements.txt` (json-repair) and root `requirements.txt` for CI/install; updated `TestGetAllCatalogModelsDedup` to patch `_fetch_catalog_by_nim_type` with `skip_licenses=True`; aligned `TestFindLicenseValidation.test_raises_on_empty_publisher` with `find_license()` returning None for empty publisher; fixed typo `duplictions` → `duplications` in `nim_agent.py`.

## Root cause

Verification failed because: (1) `json-repair` was imported but not declared; (2) tests mocked removed public APIs; (3) `find_license` behavior changed without test update; (4) comment typo.

## Test results

52 passed, 0 failed (`python3 -m pytest agent/tests/test_agent.py -q`).

## Notes

C2–C5 acceptance items refer to the original PR review deliverable, not this fix pass.
