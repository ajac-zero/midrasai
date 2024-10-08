default:
  @just -l

# Install development dependencies
install:
  @poetry install --all-extras --no-cache

# Run formatter & linter
tidy:
  @poetry run ruff check
  @poetry run ruff format

# Publish to pypi
publish:
  @poetry build
  @poetry publish

# Run all tests
test-all:
  @poetry run pytest -q

# Run tests for local Midras
test-local:
  @poetry run pytest -v tests/test_local_midras.py

# Run tests for Midras server
test-server:
  @poetry run pytest -v tests/test_midras_server.py

# Run tests for Midras client
test-client:
  @poetry run pytest -v tests/test_midras_client.py
