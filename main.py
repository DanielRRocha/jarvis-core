"""
J.A.R.V.I.S. - Just A Rather Very Intelligent System
Main entry point for the application.
"""

import asyncio
import sys
from interfaces.cli import JarvisCLI
from config.settings import settings


def main():
    """Main entry point."""
    try:
        # Initialize and run the CLI
        cli = JarvisCLI()
        asyncio.run(cli.run())
    except Exception as e:
        print(f"Failed to start JARVIS: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()