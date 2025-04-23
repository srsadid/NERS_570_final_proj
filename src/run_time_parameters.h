/**
 * @file run_time_parameters.h
 * @brief Declares structures and classes for handling runtime parameters.
 */

#ifndef RUN_TIME_PARAMETERS_H
#define RUN_TIME_PARAMETERS_H

#include <deal.II/base/parameter_handler.h> // For ParameterHandler
// PETSc includes are likely not needed directly in this header,
// they are properties of the NavierStokesProjection class itself.
// #include <deal.II/lac/petsc_solver.h>
// #include <deal.II/lac/petsc_precondition.h>
// #include <deal.II/lac/petsc_sparse_matrix.h>
// #include <deal.II/lac/petsc_vector.h>
#include <string> // For std::string

namespace NERS570_proj {
    /**
     * @brief Namespace containing classes and enums related to runtime parameters.
     */
    namespace RunTimeParameters {

      /**
       * @brief Enum defining the type of pressure update method used.
       */
      enum class Method {
          standard,   /**< Standard pressure update: p^{n+1} = p^n + phi^{n+1} */
          rotational  /**< Rotational pressure update (includes viscous term) */
      };

      /**
       * @brief A class to declare, read, and store runtime parameters.
       *
       * This class uses deal.II's ParameterHandler to define acceptable parameters,
       * read them from an input file, and store their values for use by the
       * simulation classes.
       */
      class Data_Storage
      {
      public:
        /**
         * @brief Constructor. Declares all parameters with default values and descriptions.
         */
        Data_Storage();

        /**
         * @brief Reads parameter values from a specified file.
         *
         * Parses the input file using the ParameterHandler and stores the values
         * in the public member variables. Throws an exception if the file cannot be opened.
         * @param filename The path to the parameter file.
         */
        void read_data(const std::string &filename); //

        // --- Public Member Variables storing parameter values ---

        /** @brief Type of pressure update method (Standard or Rotational). */
        Method form; //
        /** @brief Time step size. */
        double dt; //
        /** @brief Initial simulation time. */
        double initial_time; //
        /** @brief Final simulation time. */
        double final_time; //
        /** @brief Reynolds number. */
        double Reynolds; //
        /** @brief mesh file name **/
        std::string mesh_filename;
        /** @brief Number of global mesh refinements. */
        unsigned int n_global_refines; //
        /** @brief Polynomial degree for the pressure finite element space (Q_p). */
        unsigned int pressure_degree; //
        /** @brief Maximum iterations for the velocity solver (GMRES). */
        unsigned int vel_max_iterations; //
        /** @brief Krylov subspace size for GMRES (restart parameter). */
        unsigned int vel_Krylov_size; //
        /** @brief Number of off-diagonal elements for ILU preconditioner (if used). */
        unsigned int vel_off_diagonals; //
        /** @brief Frequency (in time steps) for updating the velocity preconditioner. */
        unsigned int vel_update_prec; //
        /** @brief Relative tolerance for the velocity solver. */
        double       vel_eps; //
        /** @brief Diagonal strengthening coefficient for ILU preconditioner (if used). */
        double       vel_diag_strength; //
        /** @brief Flag to enable/disable verbose output during simulation. */
        bool         verbose; //
        /** @brief Frequency (in time steps) for writing output files. */
        unsigned int output_interval; //

      protected:
        /** @brief deal.II ParameterHandler object used for parameter management. */
        dealii::ParameterHandler prm; //
      };

    } // namespace RunTimeParameters
  } // namespace NERS570_proj

  #endif // RUN_TIME_PARAMETERS_H