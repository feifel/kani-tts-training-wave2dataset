# Function to print colored messages
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

# 1. Create virtual environment
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1: Creating virtual environment"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

VENV_DIR="venv"

if [ -d "$VENV_DIR" ]; then
    print_info "Virtual environment already exists at ./$VENV_DIR"
    print_info "Removing existing virtual environment..."
    rm -rf "$VENV_DIR"
fi
print_info "Creating virtual environment at ./$VENV_DIR..."
python3 -m venv "$VENV_DIR"
print_success "Virtual environment created"

# 2. Install Python packages
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 2: Installing Python dependencies"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

print_info "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

print_info "Upgrading pip..."
pip install --upgrade pip

print_info "Installing requirements from requirements.txt..."
print_info "This may take several minutes..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    print_success "All Python packages installed successfully"
else
    print_error "Failed to install some packages"
    exit 1
fi