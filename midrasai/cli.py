import warnings


def cli(host: str = "127.0.0.1", port: int = 8000):
    try:
        import uvicorn

        from midrasai.local.server import app

        uvicorn.run(app, host=host, port=port)

    except ImportError:
        warnings.warn("Local extra dependencies not installed. Server unavailable.")


if __name__ == "__main__":
    import typer

    typer.run(cli)
