import socket

from soen_toolkit.utils.physical_mappings import main as pm_main


def find_free_port(start_port=5001, max_attempts=100):
    """Find an available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", port))
                return port
        except OSError:
            continue
    msg = f"Could not find a free port in range {start_port}-{start_port + max_attempts}"
    raise RuntimeError(msg)


def main(port: int | None = None, debug: bool = True) -> None:
    """Launch the Physical-Mappings web GUI.

    This simply imports the Flask `app` object from
    `soen_toolkit.utils.physical_mappings.main` and runs it.
    Any command-line flags are passed through to `Flask.run`.

    Args:
        port: Port to run on. If None, automatically find a free port starting from 5001.
        debug: Enable Flask debug mode.

    """
    if port is None:
        port = find_free_port()

    pm_main.app.run(debug=debug, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Launch the SOEN physical-mappings GUI")
    parser.add_argument("--port", type=int, default=None, help="Port to serve the web GUI (auto-detects if not specified)")
    parser.add_argument("--no-debug", action="store_true", help="Disable Flask debug mode")
    args = parser.parse_args()

    main(port=args.port, debug=not args.no_debug)
