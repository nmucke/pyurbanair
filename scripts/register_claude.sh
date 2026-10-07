#!/usr/bin/env bash
# Register this checkout's MCP server with Claude Code (user scope), once.
#
#   pixi run -e mcp register-claude
#
# Asks before changing anything; without a terminal it only prints the command.
set -euo pipefail

launcher="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/start_mcp"
command=(claude mcp add --transport stdio --scope user pyurbanair -- "$launcher")

if ! command -v claude >/dev/null 2>&1; then
    echo "Claude Code's 'claude' CLI is not on PATH. Install it (see"
    echo "https://docs.claude.com/en/docs/claude-code/setup), then run this again."
    exit 0
fi
if claude mcp get pyurbanair >/dev/null 2>&1; then
    echo "Claude Code already has an MCP server named 'pyurbanair':"
    claude mcp get pyurbanair
    exit 0
fi
if [[ ! -t 0 ]]; then
    echo "No terminal to ask in. To register the server, run:"
    echo "  ${command[*]}"
    exit 0
fi
read -r -p "Add the pyurbanair MCP server to Claude Code? [y/N] " answer || answer=""
if [[ "$answer" =~ ^[Yy]([Ee][Ss])?$ ]]; then
    "${command[@]}"
else
    echo "Not registered."
fi
