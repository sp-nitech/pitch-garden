PINACT_VERSION     := 5.0.0
SHELLCHECK_VERSION := 0.11.0
SHFMT_VERSION      := 3.14.1
TAPLO_VERSION      := 0.10.0
YAMLFMT_VERSION    := 0.21.0

all: pinact shellcheck shfmt taplo yamlfmt

pinact:
	mkdir -p pinact
	wget https://github.com/suzuki-shunsuke/pinact/releases/download/v$(PINACT_VERSION)/pinact_linux_amd64.tar.gz -O pinact.tar.gz
	tar xzf pinact.tar.gz -C pinact
	rm -f pinact.tar.gz

shellcheck:
	mkdir -p shellcheck
	wget https://github.com/koalaman/shellcheck/releases/download/v$(SHELLCHECK_VERSION)/shellcheck-v$(SHELLCHECK_VERSION).linux.x86_64.tar.gz -O shellcheck.tar.gz
	tar xzf shellcheck.tar.gz -C shellcheck --strip-components=1
	rm -f shellcheck.tar.gz

shfmt:
	mkdir -p shfmt
	wget https://github.com/mvdan/sh/releases/download/v$(SHFMT_VERSION)/shfmt_v$(SHFMT_VERSION)_linux_amd64 -O shfmt/shfmt
	chmod +x shfmt/shfmt

taplo:
	mkdir -p taplo
	wget https://github.com/tamasfe/taplo/releases/download/$(TAPLO_VERSION)/taplo-linux-x86_64.gz -O taplo.gz
	gunzip -c taplo.gz > taplo/taplo
	chmod +x taplo/taplo
	rm -f taplo.gz

yamlfmt:
	mkdir -p yamlfmt
	wget https://github.com/google/yamlfmt/releases/download/v$(YAMLFMT_VERSION)/yamlfmt_$(YAMLFMT_VERSION)_Linux_x86_64.tar.gz -O yamlfmt.tar.gz
	tar xzf yamlfmt.tar.gz -C yamlfmt
	rm -f yamlfmt.tar.gz

clean:
	rm -rf pinact shellcheck shfmt taplo yamlfmt

.PHONY: all clean
