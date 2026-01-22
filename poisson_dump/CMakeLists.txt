cmake_minimum_required(VERSION 3.10)
project(PoissonReconDump CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP)
find_package(ZLIB REQUIRED)
find_package(PNG REQUIRED)
find_package(JPEG REQUIRED)

set(REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/..")

add_executable(PoissonReconDump
    ${REPO_ROOT}/poisson/tools/PoissonReconDump.cpp
    ${REPO_ROOT}/poisson/src/Wrapper.cpp
)


target_include_directories(PoissonReconDump PRIVATE
    ${REPO_ROOT}/PoissonRecon/Src
    ${REPO_ROOT}/poisson/src
    ${ZLIB_INCLUDE_DIRS}
    ${PNG_INCLUDE_DIRS}
    ${JPEG_INCLUDE_DIRS}
)

target_link_libraries(PoissonReconDump PRIVATE
    ${ZLIB_LIBRARIES}
    ${PNG_LIBRARIES}
    ${JPEG_LIBRARIES}
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(PoissonReconDump PRIVATE OpenMP::OpenMP_CXX)
endif()
