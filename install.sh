#!/bin/sh
# Tochka user-local installer for Linux/macOS. No sudo, git or preinstalled Python required.
set -eu

shell_quote() {
    printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"
}

cleanup_install() {
    [ -z "${node_stage:-}" ] || rm -rf "$node_stage"
    [ -z "${temp_dir:-}" ] || rm -rf "$temp_dir"
}

node_supported() {
    "$1" -e 'const [major, minor] = process.versions.node.split(".").map(Number); process.exit(major > 22 || (major === 22 && minor >= 22) ? 0 : 1)' >/dev/null 2>&1
}

prepare_node() {
    if command -v node >/dev/null 2>&1 && command -v npm >/dev/null 2>&1 && node_supported "$(command -v node)"; then
        return 0
    fi
    node_version='24.22.2'
    case "$(uname -s)-$(uname -m)" in
        Linux-x86_64) node_platform='linux-x64'; node_sha256='361718c444e49af2ca5a110c83ba6a8901143f23ab285ebbef0341f391f482a9' ;;
        Linux-aarch64|Linux-arm64) node_platform='linux-arm64'; node_sha256='95c264150aceb7f3437a29fd6984924f5f8d1511ea3cbd6c508b83f6c555dae2' ;;
        Darwin-x86_64) node_platform='darwin-x64'; node_sha256='d5e336d588c45be0c37dbd52cc29895593009ca4997a77a1fd02815131ed0e52' ;;
        Darwin-arm64|Darwin-aarch64) node_platform='darwin-arm64'; node_sha256='aa9549349e47dffaa42dcf95d6e06ade8c4bdafc980979883b88767670df385a' ;;
        *) echo 'Supported Node platforms: Linux/macOS, x86_64/ARM64.' >&2; return 1 ;;
    esac
    node_name="node-v${node_version}-${node_platform}"
    node_parent="${XDG_DATA_HOME:-$HOME/.local/share}/tochka/node"
    node_target="$node_parent/$node_name"
    if [ -x "$node_target/bin/node" ] && [ -x "$node_target/bin/npm" ] && [ "$("$node_target/bin/node" --version)" = "v$node_version" ]; then
        PATH="$node_target/bin:$PATH"; export PATH
        return 0
    fi
    if [ -e "$node_target" ] || [ -L "$node_target" ]; then
        echo "Managed Node directory is incomplete: $node_target. Preserve/remove it and rerun the installer." >&2
        return 1
    fi
    for required in curl tar mkdir mv; do
        command -v "$required" >/dev/null 2>&1 || { echo "Install $required and try again." >&2; return 1; }
    done
    if command -v sha256sum >/dev/null 2>&1; then
        checksum_tool='sha256sum'
    elif command -v shasum >/dev/null 2>&1; then
        checksum_tool='shasum'
    else
        echo 'A SHA-256 verifier (sha256sum or shasum) is required.' >&2; return 1
    fi
    mkdir -p "$node_parent"
    node_stage=$(mktemp -d "$node_parent/.node-install.XXXXXX")
    node_archive="$node_stage/node.tar.gz"
    echo "Downloading official Node.js $node_version with npm ($node_platform); no system packages are changed..."
    curl --proto '=https' --tlsv1.2 -fsSL --connect-timeout 15 --max-time 300 "https://nodejs.org/dist/v${node_version}/${node_name}.tar.gz" -o "$node_archive"
    if [ "$checksum_tool" = 'sha256sum' ]; then
        actual_checksum=$(sha256sum "$node_archive")
    else
        actual_checksum=$(shasum -a 256 "$node_archive")
    fi
    actual_checksum=${actual_checksum%% *}
    if [ "$actual_checksum" != "$node_sha256" ]; then
        echo 'Node.js checksum mismatch. Download was not executed or installed.' >&2; return 1
    fi
    tar -xzf "$node_archive" -C "$node_stage"
    if [ ! -x "$node_stage/$node_name/bin/node" ] || [ ! -x "$node_stage/$node_name/bin/npm" ] || [ "$("$node_stage/$node_name/bin/node" --version)" != "v$node_version" ]; then
        echo 'Downloaded Node.js cannot run on this system; installation was not committed.' >&2; return 1
    fi
    # Stage and destination share a filesystem: commit only a verified complete runtime.
    mv "$node_stage/$node_name" "$node_target"
    rm -rf "$node_stage"; node_stage=''
    PATH="$node_target/bin:$PATH"; export PATH
    echo "Node.js/npm installed for Tochka only: $node_target"
}

install_tochka() {
    tochka_version='0.5.1'
    node_stage=''; temp_dir=''
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
                echo 'Usage: sh install.sh [--version 0.5.1] [--no-modify-path]'
                echo 'Installs Tochka, Promptfoo, and Node.js/npm automatically when needed (Linux/macOS).'
                echo 'Uses an isolated user environment with uv; no root or model API calls.'
                return 0 ;;
            *) echo "Unknown option: $1" >&2; return 2 ;;
        esac
    done
    if ! printf '%s\n' "$tochka_version" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+$'; then
        echo 'Version must have the form 0.5.1.' >&2; return 2
    fi
    case "$(uname -s)" in
        Linux|Darwin) ;;
        *) echo 'This installer supports Linux and macOS.' >&2; return 1 ;;
    esac
    case "$(uname -m)" in
        x86_64|aarch64|arm64) ;;
        *) echo 'Supported architectures: x86_64 and ARM64.' >&2; return 1 ;;
    esac

    prepare_node

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
    echo 'Installing and verifying the required Promptfoo runtime (no model API calls)...'
    if ! "$bin_dir/tochka" setup; then
        echo 'Tochka CLI was installed, but Promptfoo setup failed. Fix the reported problem, then run:' >&2
        shell_quote "$bin_dir/tochka" >&2; printf ' setup\n' >&2
        return 1
    fi
    if [ "$modify_path" -eq 1 ]; then
        if ! "$uv_command" tool update-shell; then
            echo "PATH could not be updated. Use: $bin_dir/tochka" >&2
        fi
    fi
    echo ''
    echo 'Tochka installed. Promptfoo runtime is ready. Open a new terminal and run: tochka'
    printf 'Run immediately: '; shell_quote "$bin_dir/tochka"; printf '\n'
    echo 'Inside Tochka: / for the command menu; /keys to add your own API key.'
    printf 'Uninstall application: '; shell_quote "$uv_command"; printf ' tool uninstall nir-eval\n'
    echo 'API keys, experiments and results are not removed when uninstalling the application.'
}

install_tochka "$@"
