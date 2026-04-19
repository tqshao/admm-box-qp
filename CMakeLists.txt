cmake_minimum_required(VERSION 3.14)
project(ADMMTrajectoryOptimization
    VERSION 1.0.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---- Dependencies ---------------------------------------------------------
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

include(FetchContent)
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)

# ---- Library --------------------------------------------------------------
add_library(admm_solver src/admm_solver.cpp)
target_link_libraries(admm_solver PUBLIC Eigen3::Eigen)

# ---- Tests ----------------------------------------------------------------
enable_testing()

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# ---- Lateral planner (application layer) ----------------------------------
add_library(lateral_planner src/lateral_planner.cpp)
target_link_libraries(lateral_planner PUBLIC admm_solver nlohmann_json::nlohmann_json)

# ---- Tests ----------------------------------------------------------------
add_executable(test_admm tests/test_admm.cpp)
target_link_libraries(test_admm PRIVATE admm_solver GTest::gtest_main)
add_test(NAME test_admm COMMAND test_admm)

add_executable(test_lateral_planner tests/test_lateral_planner.cpp)
target_link_libraries(test_lateral_planner PRIVATE lateral_planner GTest::gtest_main)
add_test(NAME test_lateral_planner COMMAND test_lateral_planner)

# ---- Apps -----------------------------------------------------------------
add_executable(export_scenarios apps/export_scenarios.cpp)
target_link_libraries(export_scenarios PRIVATE lateral_planner nlohmann_json::nlohmann_json)

# ---- Install rules ---------------------------------------------------------
include(GNUInstallDirs)

# Use generator expressions for include dirs to support install
target_include_directories(admm_solver
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Core library — installable
install(TARGETS admm_solver
    EXPORT ADMMTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(TARGETS lateral_planner export_scenarios
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/admm
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT ADMMTargets
    FILE ADMMTargets.cmake
    NAMESPACE ADMM::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ADMM
)

install(DIRECTORY configs/
    DESTINATION ${CMAKE_INSTALL_DATADIR}/admm/configs
)
