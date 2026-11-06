import argparse
import sys

from database import init_db

def main():
    parser = argparse.ArgumentParser(prog="productivity-agent")
    parser.add_argument("command", choices=["init-db", "help"], nargs="?", default="help")
    args = parser.parse_args()

    if args.command == "init-db":
        init_db()
        print("Database initialized (default user ensured).")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
