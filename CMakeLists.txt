cmake_minimum_required(VERSION 3.10)
project(pypoisson2 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Check for OpenMP
find_package(OpenMP)

# Find Image Libraries
find_package(ZLIB REQUIRED)
find_package(PNG REQUIRED)
find_package(JPEG REQUIRED)

# Hide symbols by default to avoid conflicts between std and big libraries
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)

include_directories(PoissonRecon/Src)

# Standard version (int32 indices)
add_library(pypoisson_std SHARED poisson/src/Wrapper.cpp)
target_compile_definitions(pypoisson_std PRIVATE FAST_COMPILE)
set_target_properties(pypoisson_std PROPERTIES OUTPUT_NAME "pypoisson_std")

# Big version (int64 indices)
add_library(pypoisson_big SHARED poisson/src/Wrapper.cpp)
target_compile_definitions(pypoisson_big PRIVATE FAST_COMPILE BIG_DATA)
set_target_properties(pypoisson_big PROPERTIES OUTPUT_NAME "pypoisson_big")

if(OpenMP_CXX_FOUND)
    target_link_libraries(pypoisson_std PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(pypoisson_big PUBLIC OpenMP::OpenMP_CXX)
endif()

# Install to lib directory in the package
install(TARGETS pypoisson_std pypoisson_big DESTINATION poisson/lib)

# PoissonRecon Executable
add_executable(PoissonRecon PoissonRecon/Src/PoissonRecon.cpp)
target_include_directories(PoissonRecon PRIVATE ${ZLIB_INCLUDE_DIRS} ${PNG_INCLUDE_DIRS} ${JPEG_INCLUDE_DIRS})
target_link_libraries(PoissonRecon PRIVATE ${ZLIB_LIBRARIES} ${PNG_LIBRARIES} ${JPEG_LIBRARIES})

if(OpenMP_CXX_FOUND)
    target_link_libraries(PoissonRecon PRIVATE OpenMP::OpenMP_CXX)
endif()
