"""Local-only service entry point."""
import argparse
import os


def main():
    parser = argparse.ArgumentParser(prog="quantum-lab", description="Local quantum chemistry workbench")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="Start the API and built browser application")
    serve.add_argument("--host", choices=["127.0.0.1", "localhost", "::1"], default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--data-dir", default=os.environ.get("QUANTUM_LAB_DATA", ".quantum-lab"))
    args = parser.parse_args()
    os.environ["QUANTUM_LAB_DATA"] = args.data_dir
    import uvicorn
    uvicorn.run("quantum_lab.api:app", host=args.host, port=args.port, workers=1, log_level="info")


if __name__ == "__main__":
    main()
