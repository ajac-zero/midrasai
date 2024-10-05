tidy:
  poetry run ruff check
  poetry run ruff format

publish:
  poetry build
  poetry publish


