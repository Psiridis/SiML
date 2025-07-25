# SiML
SiML – Simple Machine Learning



## Building and Installing SiML Library
This project uses CMake as its build system and supports out-of-source builds with Ninja or other generators.

### Prerequisites
- CMake (version 3.28 or higher)
- Ninja build system (or any other CMake-supported generator)
- A C++17 compatible compiler

---

### Build and Install Instructions

1. **Configure the project**

   Run this command from the root of the source directory to configure the build and generate build files inside a separate directory (`build/`):
   cmake -S . -B build/ -G Ninja

   Optionally if you want to specify the install path:
   cmake -S . -B build/ -G Ninja -DCMAKE_INSTALL_PREFIX=/path/to/install
   
   Build the project:
   cmake --build build/

   After a successful build, install the headers, library, and CMake configuration files:
   cmake --install build/
   
   EXAMPLE:
   cmake -S . -B build/ -G Ninja -DCMAKE_INSTALL_PREFIX=./install
   cmake --build build/
   cmake --install build/
   
   This will place all installed files inside the install/ directory relative to your current folder.


