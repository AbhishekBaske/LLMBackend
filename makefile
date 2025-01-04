# Makefile for FastAPI Project with Modal Setup and Token Configuration

# Variables
VENV_NAME = env
PYTHON = python3
PIP = pip
UVICORN = uvicorn
FASTAPI_APP = main:app
REQUIREMENTS_FILE = requirements.txt
MODAL_PACKAGE = modal  # You can change this if Modal requires a different package
MODAL_TOKEN_ID = ak-e6QInEmaZhKXVZ7MuwH341
MODAL_TOKEN_SECRET = as-jp58SEwNaQzSkfHkVpxrEH

# Targets

# Create and activate virtual environment
.PHONY: venv
venv:
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(VENV_NAME)
	@echo "Virtual environment created: $(VENV_NAME)"

# Install dependencies
.PHONY: install
install: venv
	@echo "Installing dependencies..."
	$(VENV_NAME)/bin/$(PIP) install -r $(REQUIREMENTS_FILE)

# Install Modal package
.PHONY: modal-setup
modal-setup:
	@echo "Setting up Modal..."
	$(VENV_NAME)/bin/$(PIP) install $(MODAL_PACKAGE)
	@echo "Modal package installed."

# Set Modal token
.PHONY: set-modal-token
set-modal-token:
	@echo "Setting Modal token..."
	$(VENV_NAME)/bin/modal token set --token-id $(MODAL_TOKEN_ID) --token-secret $(MODAL_TOKEN_SECRET)
	@echo "Modal token set successfully."

# Run the FastAPI app using Uvicorn
.PHONY: run
run:
	@echo "Running FastAPI app..."
	$(VENV_NAME)/bin/$(UVICORN) $(FASTAPI_APP) --reload

# Run tests (if you have tests, change to your test command)
.PHONY: test
test:
	@echo "Running tests..."
	$(VENV_NAME)/bin/pytest

# Clean virtual environment
.PHONY: clean
clean:
	@echo "Cleaning up virtual environment..."
	rm -rf $(VENV_NAME)

# Help command to list available Makefile commands
.PHONY: help
help:
	@echo "Available Makefile commands:"
	@echo "  make venv          Create a virtual environment"
	@echo "  make install       Install dependencies"
	@echo "  make modal-setup   Install Modal package"
	@echo "  make set-modal-token  Set the Modal token"
	@echo "  make run           Run the FastAPI app"
	@echo "  make test          Run tests (if any)"
	@echo "  make clean         Clean the virtual environment"
	@echo "  make start         Install dependencies, Modal, set token, and run the app"
