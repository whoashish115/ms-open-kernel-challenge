#!/usr/bin/env bash
set -eo pipefail

if [[ $# -gt 1 || ($# -eq 1 && $1 != "--install-cli") ]]; then
    printf 'Usage: bash setup.sh [--install-cli]\n' >&2
    exit 2
fi

if command -v okc >/dev/null 2>&1; then
    cli=$(command -v okc)
elif [[ -x "$HOME/.okc/bin/okc" ]]; then
    cli="$HOME/.okc/bin/okc"
elif [[ $1 == "--install-cli" ]]; then
    installer=$(mktemp)
    trap 'rm -f -- "$installer"' EXIT
    curl -fsSL https://openkernelchallenge.dev/install.sh -o "$installer"
    sh "$installer"
    cli="$HOME/.okc/bin/okc"
else
    printf 'Install the challenge CLI with bash setup.sh --install-cli.\n' >&2
    exit 1
fi

"$cli" --version
