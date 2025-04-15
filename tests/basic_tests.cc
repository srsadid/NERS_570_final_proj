/**
 * @file basic_tests.cc
 * @brief Basic unit tests for the NavierStokesProjection project.
 */

 #include <iostream>
 #include <fstream>
 #include <vector>
 #include <string>
 #include <cmath> // For std::abs
 
 // Include necessary headers from your project and deal.II
 // Use relative paths assuming tests/ is at the same level as src/
 #include "../src/navier_stokes_projection.h"
 #include "../src/run_time_parameters.h"
 #include <deal.II/base/mpi.h>
 #include <deal.II/base/utilities.h> // For Utilities::MPI::*
 
 // Use your namespace
 using namespace dealii;
 using namespace Step35;
 
 // Simple assertion helper for tests
 void test_assert(bool condition, const std::string &message) {
     if (!condition) {
         std::cerr << "TEST FAILED: " << message << std::endl;
         // Use MPI_Abort to ensure all processes stop in parallel test failure
         MPI_Abort(MPI_COMM_WORLD, 1);
     } else {
         // Only rank 0 prints success messages to avoid clutter
         if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
             std::cout << "  TEST PASSED: " << message << std::endl;
     }
 }
 
 // --- Test Functions ---
 
 /**
  * @brief Tests reading parameters from a file.
  */
 bool test_parameter_reading(const std::string& param_filename) {
     try {
         RunTimeParameters::Data_Storage data; // Constructor sets defaults
         // Ensure the test parameter file exists relative to the build directory execution path
         data.read_data(param_filename);      // Read the test file
 
         // Check values known to be in 'test_params.prm'
         test_assert(std::abs(data.dt - 0.01) < 1e-9, "Parameter dt read correctly (0.01)");
         test_assert(data.Reynolds == 10.0, "Parameter Reynolds read correctly (10.0)");
         test_assert(data.pressure_degree == 1, "Parameter pressure_degree read correctly (1)");
         test_assert(data.n_global_refines == 0, "Parameter n_global_refines read correctly (0)");
 
     } catch (std::exception &exc) {
         std::cerr << "Exception during parameter reading test: " << exc.what() << std::endl;
         return false;
     }
     return true;
 }
 
 /**
  * @brief Tests basic setup: triangulation, DoF distribution.
  * @warning Assumes NavierStokesProjection has getter methods:
  * get_n_vel_dofs(), get_n_pres_dofs()
  * @warning Assumes NavierStokesProjection::create_triangulation_and_dofs
  * has been modified to create a hyper_cube if test_data indicates.
  */
 bool test_basic_setup() {
     try {
         // Create minimal parameters programmatically for setup test
         RunTimeParameters::Data_Storage test_data;
         test_data.pressure_degree = 1;  // Q2/Q1 elements
         test_data.dt = 0.01;
         test_data.initial_time = 0.0;
         test_data.final_time = 0.01;
         test_data.Reynolds = 100;
         test_data.form = RunTimeParameters::Method::standard;
         test_data.n_global_refines = 0; // Coarse mesh
         // Add a flag or use a specific parameter value to signal hypercube creation
         // For example, let's assume n_global_refines=-1 means use hypercube
         test_data.n_global_refines = -1; // Signal to create hypercube in modified create_triangulation...
         // Set other necessary defaults from Data_Storage constructor if needed
 
         // Create the problem object. Assumes the constructor calls
         // create_triangulation_and_dofs, which now handles the hypercube case.
         NavierStokesProjection<2> test_problem(test_data);
 
         // Add basic checks using NEW getter methods
         // REQUIRES adding these public methods to NavierStokesProjection class:
         // unsigned int get_n_vel_dofs() const { return dof_handler_velocity.n_dofs(); }
         // unsigned int get_n_pres_dofs() const { return dof_handler_pressure.n_dofs(); }
         test_assert(test_problem.get_n_vel_dofs() > 0, "Velocity DoFs > 0 after setup");
         test_assert(test_problem.get_n_pres_dofs() > 0, "Pressure DoFs > 0 after setup");
 
     } catch (std::exception &exc) {
         std::cerr << "Exception during basic setup test: " << exc.what() << std::endl;
         return false;
     }
     return true;
 }
 
 // --- Main Test Runner ---
 
 int main(int argc, char *argv[]) {
     // Initialize MPI (required even for serial tests if using parallel deal.II build)
     Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
 
     bool all_tests_passed = true;
     int root_rank = 0;
     int my_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
 
     if (my_rank == root_rank) {
         std::cout << "===== Running Basic Unit Tests =====" << std::endl;
     }
 
     // Test 1: Parameter Reading
     if (my_rank == root_rank) std::cout << "\n--- Testing Parameter Reading ---" << std::endl;
     // Path relative to where the test executable runs (usually the build directory)
     std::string test_prm_file = "../tests/test_params.prm"; // Adjust path if needed
     bool params_ok = test_parameter_reading(test_prm_file);
     all_tests_passed = all_tests_passed && params_ok;
     // Barrier to ensure output order is nice (optional)
     MPI_Barrier(MPI_COMM_WORLD);
 
 
     // Test 2: Basic Setup (Requires modifying NavierStokesProjection slightly)
     if (my_rank == root_rank) std::cout << "\n--- Testing Basic Setup ---" << std::endl;
     bool setup_ok = test_basic_setup();
     all_tests_passed = all_tests_passed && setup_ok;
     MPI_Barrier(MPI_COMM_WORLD);
 
 
     // Add calls to other test functions here...
     // e.g., assembly smoke tests, initialization tests
 
 
     // Final Report
     MPI_Barrier(MPI_COMM_WORLD); // Wait for all processes before final message
     if (my_rank == root_rank) {
         std::cout << "\n===== Test Summary =====" << std::endl;
         if (all_tests_passed) {
             std::cout << "All tests PASSED." << std::endl;
         } else {
             std::cout << "Some tests FAILED." << std::endl;
         }
         std::cout << "========================" << std::endl;
     }
 
     return (all_tests_passed ? 0 : 1); // Return 0 on success, 1 on failure
 }