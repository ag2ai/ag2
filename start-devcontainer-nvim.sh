#!/bin/bash

# Script to start devcontainer with Neovim configuration mounted
# Usage: ./start-devcontainer-nvim.sh [command]

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Function to print colored messages
print_status() {
    echo -e "${GREEN}[*]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Check if devcontainer CLI is installed
if ! command -v devcontainer &> /dev/null; then
    print_error "devcontainer CLI is not installed"
    echo "Install it with: npm install -g @devcontainers/cli"
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    print_error "Docker is not running"
    echo "Please start Docker and try again"
    exit 1
fi

# Function to build the container
build_container() {
    print_status "Building devcontainer..."
    devcontainer build --workspace-folder "$SCRIPT_DIR"
}

# Function to start the container with mounted Neovim config
start_container() {
    print_status "Starting devcontainer with Neovim config mounted..."

    # Check if local Neovim config exists
    if [ -d "$HOME/.config/nvim" ]; then
        print_status "Mounting local Neovim config from $HOME/.config/nvim"
        MOUNT_ARGS="--mount type=bind,source=$HOME/.config/nvim,target=/home/vscode/.config/nvim"
    else
        print_warning "No Neovim config found at $HOME/.config/nvim"
        MOUNT_ARGS=""
    fi

    # Also mount local Neovim data directory if it exists (for plugins, etc.)
    if [ -d "$HOME/.local/share/nvim" ]; then
        print_status "Mounting local Neovim data from $HOME/.local/share/nvim"
        MOUNT_ARGS="$MOUNT_ARGS --mount type=bind,source=$HOME/.local/share/nvim,target=/home/vscode/.local/share/nvim"
    fi

    # Mount Neovim cache directory if it exists
    if [ -d "$HOME/.cache/nvim" ]; then
        print_status "Mounting local Neovim cache from $HOME/.cache/nvim"
        MOUNT_ARGS="$MOUNT_ARGS --mount type=bind,source=$HOME/.cache/nvim,target=/home/vscode/.cache/nvim"
    fi

    devcontainer up $MOUNT_ARGS --workspace-folder "$SCRIPT_DIR"
}

# Function to install Neovim in the container
install_nvim() {
    print_status "Installing Neovim in the container..."

    # Check if nvim is already installed
    if devcontainer exec --workspace-folder "$SCRIPT_DIR" which nvim &>/dev/null; then
        print_status "Neovim is already installed"
        return 0
    fi

    print_status "Installing Neovim via apt..."
    devcontainer exec --workspace-folder "$SCRIPT_DIR" sudo apt-get update
    devcontainer exec --workspace-folder "$SCRIPT_DIR" sudo apt-get install -y neovim

    # Install common dependencies for Neovim plugins
    print_status "Installing common dependencies (ripgrep, fd-find, etc.)..."
    devcontainer exec --workspace-folder "$SCRIPT_DIR" sudo apt-get install -y ripgrep fd-find

    print_status "Neovim installation complete"
}

# Function to run Neovim in the container
run_nvim() {
    print_status "Starting Neovim in devcontainer..."

    # Ensure Neovim is installed
    install_nvim

    # Pass any additional arguments to nvim
    shift # Remove the 'nvim' command from arguments
    devcontainer exec --workspace-folder "$SCRIPT_DIR" nvim "$@"
}

# Function to execute arbitrary commands in the container
exec_command() {
    shift # Remove the 'exec' command from arguments
    devcontainer exec --workspace-folder "$SCRIPT_DIR" "$@"
}

# Function to open a shell in the container
open_shell() {
    print_status "Opening shell in devcontainer..."
    devcontainer exec --workspace-folder "$SCRIPT_DIR" /bin/zsh
}

# Function to stop the container
stop_container() {
    print_status "Stopping devcontainer..."
    docker stop python-3.10-ag2 2>/dev/null || true
}

# Function to rebuild the container
rebuild_container() {
    print_status "Rebuilding devcontainer..."
    stop_container
    build_container
    start_container
}

# Function to check if container is running
is_container_running() {
    docker ps --format '{{.Names}}' | grep -q '^python-3.10-ag2$'
}

# Main script logic
case "${1:-}" in
    build)
        build_container
        ;;
    start)
        if is_container_running; then
            print_warning "Container is already running"
        else
            start_container
        fi
        ;;
    nvim)
        if ! is_container_running; then
            start_container
        fi
        run_nvim "$@"
        ;;
    exec)
        if ! is_container_running; then
            print_error "Container is not running. Start it first with: $0 start"
            exit 1
        fi
        exec_command "$@"
        ;;
    shell)
        if ! is_container_running; then
            start_container
        fi
        open_shell
        ;;
    stop)
        stop_container
        ;;
    rebuild)
        rebuild_container
        ;;
    status)
        if is_container_running; then
            print_status "Container is running"
        else
            print_warning "Container is not running"
        fi
        ;;
    *)
        echo "Usage: $0 {build|start|nvim|exec|shell|stop|rebuild|status}"
        echo ""
        echo "Commands:"
        echo "  build    - Build the devcontainer image"
        echo "  start    - Start the devcontainer with mounted Neovim config"
        echo "  nvim     - Run Neovim in the devcontainer (installs if needed)"
        echo "  exec     - Execute a command in the running container"
        echo "  shell    - Open a shell in the devcontainer"
        echo "  stop     - Stop the running devcontainer"
        echo "  rebuild  - Rebuild and restart the devcontainer"
        echo "  status   - Check if the container is running"
        echo ""
        echo "Examples:"
        echo "  $0 start           # Start the devcontainer"
        echo "  $0 nvim            # Open Neovim"
        echo "  $0 nvim file.py    # Open a specific file in Neovim"
        echo "  $0 exec python     # Run Python in the container"
        echo "  $0 shell           # Open a shell in the container"
        exit 1
        ;;
esac