#!/bin/sh
# installs panopticon, a harness where coding agents work toward one goal.
#   curl -LsSf https://raw.githubusercontent.com/bhark/panopticon/main/install.sh | sh
#   curl -LsSf https://raw.githubusercontent.com/bhark/panopticon/main/install.sh | sh -s -- v0.2.0
set -eu

REPO="bhark/panopticon"
VERSION="${1:-}"

fail() {
  echo "[!] $1" >&2
  shift
  for hint in "$@"; do echo "    $hint" >&2; done
  exit 1
}

command -v uv >/dev/null 2>&1 || fail \
  "uv is not installed" \
  "panopticon needs Python 3.14 or newer, which uv fetches for you." \
  "Install it: curl -LsSf https://astral.sh/uv/install.sh | sh"

if [ -n "$VERSION" ]; then
  api="https://api.github.com/repos/$REPO/releases/tags/$VERSION"
else
  api="https://api.github.com/repos/$REPO/releases/latest"
fi

echo "[-] Looking up ${VERSION:-the latest release}..."
wheel="$(curl -LsSf "$api" \
  | grep -o '"browser_download_url": *"[^"]*\.whl"' \
  | head -n 1 | sed 's/.*"\(https[^"]*\)"/\1/')"

[ -n "$wheel" ] || fail \
  "no wheel found for ${VERSION:-the latest release}" \
  "Check the releases at https://github.com/$REPO/releases"

echo "[-] Installing $(basename "$wheel")..."
uv tool install --force "panopticon @ $wheel"

bin_dir="$(uv tool dir --bin)"
case ":$PATH:" in
  *":$bin_dir:"*) ;;
  *)
    uv tool update-shell
    # rc files only apply to new shells, so name the escape hatch for this one
    echo ""
    echo "[!] $bin_dir is not on your PATH."
    echo "    Open a new terminal, or run: export PATH=\"$bin_dir:\$PATH\""
    ;;
esac

echo ""
echo "Panopticon is watching. Try: panopticon --help"
echo "It needs git, plus a signed-in claude, codex or kimi CLI (or OPENROUTER_API_KEY)."
echo ""
