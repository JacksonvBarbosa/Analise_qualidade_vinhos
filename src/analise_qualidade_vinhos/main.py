
"""
Entry point to trigger training from the command line.
Keeps things explicit for junior-level readability.
"""

from analise_qualidade_vinhos.pipeline.train import cli


def main():
    cli()


if __name__ == "__main__":
    main()