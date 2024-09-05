import click


@click.command()
@click.argument("action")
@click.option("--host", default="127.0.0.1", help="Server host")
@click.option("--port", default=8765, type=int, help="Server port")
def midrascli(action, host, port):
    if action == "server":
        import uvicorn

        from midrasai.local.server import app

        uvicorn.run(app, host=host, port=port)
    else:
        click.echo(f"Unknown action: {action}")


if __name__ == "__main__":
    midrascli()
