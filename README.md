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
cmake -S . -B build/ -G Ninja
```
Optionally if you want to specify the install path:
```bash
cmake -S . -B build/ -G Ninja -DCMAKE_INSTALL_PREFIX=/path/to/install
```
Build the project:
```bash
cmake --build build/
```
After a successful build, install the headers, library, and CMake configuration files:
```bash
cmake --install build/
```
## Testing

This project integrates the [Google Test](https://github.com/google/googletest) framework to provide a robust testing mechanism. Testing is controlled by the CMake option flag `SiML_ENABLE_TESTS`. By default, this flag is **ON**, and tests are built. To disable tests set the flag to `OFF`.
```bash
cmake -S . -B build/ -G Ninja -DSiML_ENABLE_TESTS=OFF
```
Run tests (all):
```bash
cmake --build build/ --target test
```
Or directly with CTest (filtering only these related with SiML Library):
```bash
ctest --build-dir build/ -R SiML
```

---

### 🧪 Build Example
Install locally in a subdirectory called `install/`:
```bash
cmake -S . -B build/ -G Ninja -DCMAKE_INSTALL_PREFIX=./install -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build/
cmake --install build/
```
This will place all installed files inside the install/ directory relative to your current folder.

---

## Developer Tips

### Using `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`

When configuring the project, you can add the following option to generate a `compile_commands.json` file:

```bash
cmake -S . -B build/ -G Ninja -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```
You can enable various options during configuration using CMake flags. For example:
```bash
cmake -S . -B build/ -G Ninja -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DSiML_ENABLE_TESTS=ON -DCMAKE_INSTALL_PREFIX=./install
```