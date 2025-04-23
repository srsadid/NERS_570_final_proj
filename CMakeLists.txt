cmake_minimum_required(VERSION 3.10)
project(NavierStokesProjection)

# Find the deal.II package
# Use your existing find_package line
find_package(deal.II 9.3 REQUIRED HINTS /home/sadid/deal.ii_install ) 

deal_ii_initialize_cached_variables()


#-----------------------------------------------------------------------------
# Information about Build Types
#-----------------------------------------------------------------------------
# - Debug: For development (-O0 -g)
# - Release: For performance (-O3 -march=native, etc., no -g)
# - RelWithDebInfo: Optimized with debug symbols (-O2 -g)
#   select the build type when running cmake, e.g.:
#   cmake .. -DCMAKE_BUILD_TYPE=Release

message(STATUS "Using build type: ${CMAKE_BUILD_TYPE}")
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
#  Compiler Optimization Flags           
#-----------------------------------------------------------------------------
# Check the compiler ID to apply relevant flags (GCC and Clang share many)
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")

  # --- Release Build Optimizations ---

  # Set the main optimization level (e.g., -O3 for aggressive optimization)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3"
      CACHE STRING "Flags used for Release builds" FORCE)

  # Enable optimizations specific to the CPU architecture of the build machine
  # Includes vectorization like AVX, AVX2 etc. Improves performance significantly.
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native"
      CACHE STRING "Flags used for Release builds" FORCE)


  # Enable Link-Time Optimization (LTO) - Increases link time
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -flto"
       CACHE STRING "Flags used for Release builds" FORCE)
  set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -flto"
       CACHE STRING "Flags used for Release linker" FORCE)


  # --- Report Flags ---
  message(STATUS "Applied custom GCC/Clang optimization/debug flags.")
  # You can see the full flags with 'make VERBOSE=1'.

else()
  message(STATUS "Compiler is not GCC or Clang (${CMAKE_CXX_COMPILER_ID}). No custom optimization flags applied by this block.")
endif()

#-----------------------------------------------------------------------------
# End Section: Optional Compiler Optimization Flags
#-----------------------------------------------------------------------------



#-----------------------------------------------------------------------------
# Main Executable Target
#-----------------------------------------------------------------------------
# List all source files for the executable
add_executable(navier_stokes_projection
  src/main.cc                     # Main function
  src/run_time_parameters.cc      # Runtime parameters implementation
  src/navier_stokes_projection/navier_stokes_projection_instantiation.cc
)


target_include_directories(navier_stokes_projection PRIVATE src) 


# Setup deal.II linkage etc.
deal_ii_setup_target(navier_stokes_projection)

# Install rule (from original)
install(TARGETS navier_stokes_projection
        RUNTIME DESTINATION bin
) 
#-----------------------------------------------------------------------------


#-----------------------------------------------------------------------------
# Testing Setup 
#-----------------------------------------------------------------------------
# Enable the testing framework
enable_testing()

# Add the test executable target
add_executable(run_unit_tests
  # Source file containing test main() and test functions
  tests/basic_tests.cc

  src/run_time_parameters.cc
  src/navier_stokes_projection/navier_stokes_projection_instantiation.cc # Still


)

# Set include directories for the test executable
target_include_directories(run_unit_tests PRIVATE src)

# Setup deal.II linkage etc. for the test executable
deal_ii_setup_target(run_unit_tests)

# Add the test executable to CTest registry
# This defines a test named "UnitTests" that CTest will run
add_test(NAME UnitTests COMMAND run_unit_tests)

# Example for adding a parallel test run (uncomment and adjust if needed)
# Requires find_package(MPI) called earlier (deal.II usually finds it)
# This runs the same test executable using mpirun with 4 processes
# add_test(NAME ParallelUnitTests COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:run_unit_tests>)

#-----------------------------------------------------------------------------
# Final Messages
#-----------------------------------------------------------------------------
message(STATUS "Build configuration complete!")
message(STATUS "Main executable target: navier_stokes_projection")
message(STATUS "Test executable target: run_unit_tests")
message(STATUS "To build: run 'make'")
message(STATUS "To run tests: run 'make test'")
message(STATUS "To specify build type (e.g., Release or Debug): cmake .. -DCMAKE_BUILD_TYPE=Release") 
message(STATUS "Installation prefix: ${CMAKE_INSTALL_PREFIX}")


