#!/bin/sh
# Tochka user-local installer for Linux/macOS. No sudo, git or preinstalled Python required.
set -eu

shell_quote() {
    printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"
}

cleanup_install() {
    [ -z "${temp_dir:-}" ] || rm -rf "$temp_dir"
}

install_tochka() {
    tochka_version='0.5.2'
    temp_dir=''
    trap cleanup_install EXIT
    trap 'exit 130' INT TERM
    modify_path=1
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --no-modify-path) modify_path=0; shift ;;
            --version)
                [ "$#" -ge 2 ] || { echo 'Missing version' >&2; return 2; }
                tochka_version=${2#v}; shift 2 ;;
            --help|-h)
                echo 'Usage: sh install.sh [--version 0.5.2] [--no-modify-path]'
                echo 'Installs Tochka with its native Python API engine (Linux/macOS).'
                echo 'Uses an isolated user environment with uv; no root or model API calls.'
                return 0 ;;
            *) echo "Unknown option: $1" >&2; return 2 ;;
        esac
    done
    if ! printf '%s\n' "$tochka_version" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+$'; then
        echo 'Version must have the form 0.5.2.' >&2; return 2
    fi
    case "$(uname -s)" in
        Linux|Darwin) ;;
        *) echo 'This installer supports Linux and macOS.' >&2; return 1 ;;
    esac
    case "$(uname -m)" in
        x86_64|aarch64|arm64) ;;
        *) echo 'Supported architectures: x86_64 and ARM64.' >&2; return 1 ;;
    esac

    if command -v uv >/dev/null 2>&1; then
        uv_command=$(command -v uv)
    else
        command -v curl >/dev/null 2>&1 || { echo 'Install curl and try again.' >&2; return 1; }
        bootstrap_dir="${XDG_DATA_HOME:-$HOME/.local/share}/tochka/bootstrap"
        uv_command="$bootstrap_dir/uv"
        if [ ! -x "$uv_command" ]; then
            temp_dir=$(mktemp -d "${TMPDIR:-/tmp}/tochka-install.XXXXXX")
            echo 'Downloading uv (isolated Python installer)...'
            curl --proto '=https' --tlsv1.2 -fsSL 'https://astral.sh/uv/0.13.4/install.sh' -o "$temp_dir/uv-install.sh"
            UV_INSTALL_DIR="$bootstrap_dir" UV_NO_MODIFY_PATH=1 sh "$temp_dir/uv-install.sh"
        fi
    fi
    source_url="${TOCHKA_INSTALL_SOURCE:-https://github.com/ignatnovichenko/Tochka/archive/refs/tags/v${tochka_version}.tar.gz}"
    echo "Installing Tochka $tochka_version (Python 3.11 is downloaded automatically if needed)..."
    "$uv_command" tool install --python 3.11 --force "$source_url"
    bin_dir=$("$uv_command" tool dir --bin)
    "$bin_dir/tochka" --help >/dev/null
    echo 'Checking the native Python runtime (no model API calls)...'
    if ! "$bin_dir/tochka" setup --check; then
        echo 'Tochka CLI was installed, but its runtime check failed. Fix the reported problem, then run:' >&2
        shell_quote "$bin_dir/tochka" >&2; printf ' setup\n' >&2
        return 1
    fi
    if [ "$modify_path" -eq 1 ]; then
        if ! "$uv_command" tool update-shell; then
            echo "PATH could not be updated. Use: $bin_dir/tochka" >&2
        fi
    fi
    echo ''
    echo 'Tochka installed. Native Python runtime is ready. Open a new terminal and run: tochka'
    printf 'Run immediately: '; shell_quote "$bin_dir/tochka"; printf '\n'
    echo 'Inside Tochka: / for the command menu; /keys to add your own API key.'
    printf 'Uninstall application: '; shell_quote "$uv_command"; printf ' tool uninstall nir-eval\n'
    echo 'API keys, experiments and results are not removed when uninstalling the application.'
}

install_tochka "$@"
