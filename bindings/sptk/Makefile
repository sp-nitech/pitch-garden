SPTK_DIR := third_party/SPTK
BUILD_DIR := build/sptk

UV_RUN := uv run --no-sync

sptk:
	cmake -S $(SPTK_DIR) -B $(BUILD_DIR) -DSPTK_INSTALL=OFF
	cmake --build $(BUILD_DIR) --parallel --target pitch

test: sptk
	$(UV_RUN) pytest tests

clean:
	rm -rf build

.PHONY: sptk test clean
