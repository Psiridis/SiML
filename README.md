# SiML
**SiML – Simple Machine Learning**

A lightweight and easy-to-use machine learning library written in C++. Designed for simplicity, speed, and extensibility.

---

## 📦 Building and Installing the SiML Library

SiML uses **CMake** as its build system and supports out-of-source builds with **Ninja** or other CMake-supported generators.

### ✅ Prerequisites
- [CMake](https://cmake.org/) (version **3.28** or higher)
- [Ninja](https://ninja-build.org/) build system *(or another CMake-supported generator)*
- A **C++17** compatible compiler (e.g., GCC 9+, Clang 10+, MSVC 2019+)

---

### 🛠️ Build and Install Instructions
**Configure the Project**

From the root of the project, generate build files in a separate `build/` directory:
```bash
cmake --preset dev-debug
```
Build the project:
```bash
cmake --build --preset build-debug
```
After a successful build, install the headers, library, and CMake configuration files:
```bash
cmake --install build/debug
```
## Testing

This project integrates the [Google Test](https://github.com/google/googletest) framework to provide a robust testing mechanism. Testing is controlled by the CMake option flag `SiML_ENABLE_TESTS`. By default, this flag is **ON**, and tests are built. A debug preset, called "dev-debug-no-test" exists which has this flag disabled.
```bash
cmake --preset dev-debug-no-tests
```
Run tests with CTest:
```bash
ctest --preset test-debug
```

---

### 🧪 Build Example
Install locally in a subdirectory called `install/`:
```bash
cmake --preset dev-debug
cmake --build --preset build-debug
ctest --preset test-debug -R SiML
```
This will place all installed files inside the install/ directory relative to your current folder.
```bash
cmake --install build/debug

```
---
