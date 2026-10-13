PROJECT := pitch_garden
SOURCES := $(PROJECT) tests bindings/sptk/src bindings/sptk/tests examples
MODULE  :=

PYTHON_VERSION := 3.12

# The build of PyTorch: gpu or cpu.
TORCH := gpu

UV_RUN := uv run --no-sync

setup:
	git submodule update --init
	uv sync --locked --all-extras --group $(TORCH) --python $(PYTHON_VERSION)

update:
	uv lock --upgrade
	uv sync --locked --all-extras --group $(TORCH) --python $(PYTHON_VERSION)

dist:
	uv build
	$(UV_RUN) twine check dist/*

dist-clean:
	rm -rf dist

doc:
	$(UV_RUN) $(MAKE) -C docs html

doc-clean:
	rm -rf docs/build

check: tool
	uv lock --check
	$(UV_RUN) ruff check $(SOURCES)
	$(UV_RUN) ruff format --check $(SOURCES) docs/source
	$(UV_RUN) pyright $(SOURCES)
	$(UV_RUN) cmake-format --check bindings/sptk/CMakeLists.txt
	$(UV_RUN) mdformat --check *.md bindings/sptk/*.md examples/*/README.md
	./tools/pinact/pinact run --check .github/workflows/*.yaml
	./tools/shellcheck/shellcheck examples/*/run.sh
	./tools/shfmt/shfmt -i 4 -d examples/*/run.sh
	./tools/taplo/taplo fmt --check *.toml bindings/sptk/*.toml
	./tools/yamlfmt/yamlfmt --lint .yamlfmt.yaml .github/workflows/*.yaml configs/*.yaml

format: tool
	$(UV_RUN) ruff check --fix $(SOURCES)
	$(UV_RUN) ruff format $(SOURCES) docs/source
	$(UV_RUN) cmake-format -i bindings/sptk/CMakeLists.txt
	$(UV_RUN) mdformat *.md bindings/sptk/*.md examples/*/README.md
	./tools/pinact/pinact run -u --min-age 14 .github/workflows/*.yaml
	./tools/shfmt/shfmt -i 4 -w examples/*/run.sh
	./tools/taplo/taplo fmt *.toml bindings/sptk/*.toml
	./tools/yamlfmt/yamlfmt .yamlfmt.yaml .github/workflows/*.yaml configs/*.yaml

test:
	[ -n "$(MODULE)" ] && module=$$(find tests -name "test_$(MODULE).py") || module="$(PROJECT) tests"; \
	$(UV_RUN) pytest $$module

tool:
	cd tools && $(MAKE)

tool-clean:
	cd tools && $(MAKE) clean

clean: dist-clean doc-clean tool-clean
	cd bindings/sptk && $(MAKE) clean
	rm -rf .venv
	find . -name "__pycache__" -type d -print0 | xargs -0 rm -rf

.PHONY: setup update dist dist-clean doc doc-clean check format test tool tool-clean clean
