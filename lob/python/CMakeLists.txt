# orderbook._lob: the C++ engine as a Python extension module.
# scikit-build-core provides Python and pybind11 (build requirements in pyproject.toml).
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_lob MODULE bindings.cpp)
target_link_libraries(_lob PRIVATE lob::core qs_options)
install(TARGETS _lob LIBRARY DESTINATION orderbook)
