# Controlla se siamo in GitHub Actions
IS_GITHUB := $(GITHUB_ACTIONS)

# Comandi in base al SO
ifeq ($(OS),Windows_NT)
    R_PYTHON = venv\Scripts\python.exe
    SET_PYTHONPATH = set PYTHONPATH=. &
    RM = del /s /q
else
    R_PYTHON = ./venv/bin/python
    SET_PYTHONPATH = PYTHONPATH=.
    RM = rm -rf
endif

# Sovrascrittura se in GitHub Actions
ifeq ($(IS_GITHUB),true)
    R_PYTHON = python
endif

# Usa sempre pip legato all'interprete
PIP = $(R_PYTHON) -m pip

# Crea venv se non esiste e non su github
ifeq ($(IS_GITHUB),)
venv:
	python -m venv venv
endif

# Scarica tutte le dipendenze per preparare l'ambiente
install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo Installazione delle dipendenze terminata.

# Analisi Statistica del codice sorgente (robusta: prende file ricorsivamente)
lint:
ifeq ($(OS),Windows_NT)
	$(SET_PYTHONPATH) $(R_PYTHON) -m pylint --disable=R,C $$(powershell -NoProfile -Command "Get-ChildItem -Recurse -Filter *.py src,tests | ForEach-Object { $$_.FullName }")
else
	$(SET_PYTHONPATH) $(R_PYTHON) -m pylint --disable=R,C $$(find src tests -type f -name "*.py")
endif
	@echo Linting complete.

# Unit test
test:
	$(SET_PYTHONPATH) $(R_PYTHON) -m pytest -vv --cov=src tests/
	@echo Testing complete.

# PACKAGING
init-poetry:
	$(PIP) install poetry
	-poetry init --no-interaction
	$(R_PYTHON) -c "import os; [os.system(f'poetry add {line.strip()}') for line in open('requirements.txt') if line.strip()]"
ifeq ($(OS),Windows_NT)
	@if not exist README.md echo "# Animal AI" > README.md
else
	@test -f README.md || echo "# Animal AI" > README.md
endif

build:
	$(R_PYTHON) -m build
	@echo "Build complete. Check dist/ directory."

clean:
ifeq ($(OS),Windows_NT)
	@if exist dist rmdir /s /q dist
	@if exist build rmdir /s /q build
	@if exist *.egg-info rmdir /s /q *.egg-info
	@if exist .pytest_cache rmdir /s /q .pytest_cache
	@if exist __pycache__ rmdir /s /q __pycache__
	@if exist .coverage del /q .coverage
else
	$(RM) dist/ build/ *.egg-info .pytest_cache .coverage __pycache__
endif

# Docker
ifeq ($(OS),Windows_NT)
    DOCKER_PWD := $(subst \,/,${CURDIR})
else
    DOCKER_PWD := $(CURDIR)
endif

docker:
	docker build -t animal-ai .

docker_run: docker
	docker run --rm --name animal-ai \
		-v "$(DOCKER_PWD)/persistent_data:/app/persistent_data" \
		-v "$(DOCKER_PWD)/results:/app/results" \
		-v "$(DOCKER_PWD)/data:/app/data" \
		animal-ai

run_animal_recogn:
	$(SET_PYTHONPATH) $(R_PYTHON) -u src/animal_ai.py

docker_animal_recogn: docker
	docker run --rm --name animal-ai -p 7860:7860 \
		-v "$(DOCKER_PWD)/persistent_data:/app/persistent_data" \
		-v "$(DOCKER_PWD)/results:/app/results" \
		-v "$(DOCKER_PWD)/data:/app/data" \
		animal-ai