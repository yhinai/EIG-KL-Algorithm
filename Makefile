# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -I/usr/include/eigen3 -I/usr/local/include/spectra -fopenmp

# Libraries
LIBS = -llapack -lblas -llapacke

# Targets
.PHONY: all clean deps check_dependencies install_system_deps install_spectra build

# Main target
all: check_dependencies build

# Build target for actual compilation
build: clean EIG KL

EIG: cEIG.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LIBS)

KL: cKL.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LIBS)

# Dependencies management

install_system_deps:
	@echo "Installing system dependencies..."
	@if ! command -v apt-get &> /dev/null; then \
		echo "Error: apt-get not found. This Makefile requires Ubuntu/Debian."; \
		exit 1; \
	fi
	sudo apt-get update
	sudo apt-get install -y \
		build-essential \
		cmake \
		libomp-dev \
		liblapack-dev \
		liblapacke-dev \
		libblas-dev \
		libeigen3-dev

install_spectra:
	@echo "Installing Spectra library..."
	@if [ ! -d "/usr/local/include/Spectra" ]; then \
		echo "Spectra not found. Installing..."; \
		cd /tmp && \
		rm -rf spectra && \
		git clone https://github.com/yixuan/spectra.git && \
		cd spectra && \
		mkdir -p build && \
		cd build && \
		cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local && \
		sudo make install; \
	else \
		echo "Spectra already installed."; \
	fi

check_dependencies:
	@if [ ! -d "/usr/include/eigen3" ]; then \
		echo "Eigen3 not found. Installing dependencies..."; \
		$(MAKE) install_system_deps; \
	fi
	@if [ ! -d "/usr/local/include/Spectra" ]; then \
		echo "Spectra not found. Installing..."; \
		$(MAKE) install_spectra; \
	fi

clean:
	rm -f EIG KL