/**
 * @file equation_data.h
 * @brief Defines Function classes for initial and boundary conditions.
 *
 * This file provides specific implementations of deal.II Function classes
 * representing the analytical velocity and pressure fields used for setting
 * initial conditions and Dirichlet boundary conditions for the driven cavity
 * benchmark problem (or a similar channel flow problem).
 */

#ifndef EQUATION_DATA_H
#define EQUATION_DATA_H

// deal.II includes
#include <deal.II/base/parameter_handler.h> // Not strictly needed here
#include <deal.II/base/point.h>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>  // Not strictly needed here
#include <deal.II/base/multithread_info.h> // Not strictly needed here
#include <deal.II/base/thread_management.h> // Not strictly needed here
#include <deal.II/base/work_stream.h> // Not strictly needed here
#include <deal.II/base/parallel.h> // Not strictly needed here
#include <deal.II/base/utilities.h> // Not strictly needed here
#include <deal.II/base/conditional_ostream.h> // Not strictly needed here
#include <deal.II/base/exceptions.h> // For Assert, AssertDimension, AssertIndexRange

#include <deal.II/lac/vector.h> // Not strictly needed here
#include <deal.II/lac/sparse_matrix.h> // Not strictly needed here
#include <deal.II/lac/dynamic_sparsity_pattern.h> // Not strictly needed here
#include <deal.II/lac/solver_cg.h> // Not strictly needed here
#include <deal.II/lac/precondition.h> // Not strictly needed here
#include <deal.II/lac/solver_gmres.h> // Not strictly needed here
#include <deal.II/lac/sparse_ilu.h> // Not strictly needed here
// #include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/affine_constraints.h> // Not strictly needed here

// PETSc includes likely not needed here
// #include <deal.II/lac/petsc_solver.h>
// #include <deal.II/lac/petsc_precondition.h>
// #include <deal.II/lac/petsc_sparse_matrix.h>
// #include <deal.II/lac/petsc_vector.h>

#include <deal.II/grid/tria.h> // Not strictly needed here
#include <deal.II/grid/grid_generator.h> // Not strictly needed here
#include <deal.II/grid/grid_refinement.h> // Not strictly needed here
#include <deal.II/grid/grid_in.h> // Not strictly needed here

#include <deal.II/dofs/dof_handler.h> // Not strictly needed here
#include <deal.II/dofs/dof_tools.h> // Not strictly needed here
#include <deal.II/dofs/dof_renumbering.h> // Not strictly needed here

#include <deal.II/fe/fe_q.h> // Not strictly needed here
#include <deal.II/fe/fe_values.h> // Not strictly needed here
#include <deal.II/fe/fe_tools.h> // Not strictly needed here
#include <deal.II/fe/fe_system.h> // Not strictly needed here

#include <deal.II/numerics/matrix_creator.h> // Not strictly needed here
#include <deal.II/numerics/matrix_tools.h> // Not strictly needed here
#include <deal.II/numerics/vector_tools.h> // Not strictly needed here
#include <deal.II/numerics/data_out.h> // Not strictly needed here

// Standard library includes
#include <fstream> // Not strictly needed here
#include <cmath>
#include <iostream> // Not strictly needed here
#include <vector>

namespace Step35 {
  /**
   * @brief Namespace containing Function classes for problem-specific data.
   */
  namespace EquationData {

    /**
     * @brief Base class for multi-component functions allowing component selection.
     *
     * This helper class derives from dealii::Function<dim> and adds a mechanism
     * (`set_component`) to specify which vector component the `value` and
     * `value_list` methods should return. This is useful when interpolating
     * vector fields component by component.
     *
     * @tparam dim The spatial dimension of the problem.
     */
    template <int dim>
    class MultiComponentFunction : public dealii::Function<dim>
    {
    public:
      /**
       * @brief Constructor. Initializes the base Function class for 1 component
       * (as interpolation happens one component at a time) and sets the
       * initial time. Defaults the active component to 0.
       * @param initial_time The initial time value for the function.
       */
      MultiComponentFunction(const double initial_time = 0.); //

      /**
       * @brief Sets the active component to be evaluated by `value` or `value_list`.
       * @param d The index of the component (0 to dim-1).
       */
      void set_component(const unsigned int d); //

    protected:
      /** @brief The currently active component index. */
      unsigned int comp; //
    };

    // --- Implementation of MultiComponentFunction ---

    template <int dim>
    MultiComponentFunction<dim>::MultiComponentFunction(const double initial_time)
      : dealii::Function<dim>(1, initial_time), // Initialize base for 1 component output
        comp(0) // Default to component 0
    {}

    template <int dim>
    void MultiComponentFunction<dim>::set_component(const unsigned int d)
    {
      // Ensure requested component index is valid
      Assert(d < dim, dealii::ExcIndexRange(d, 0, dim)); //
      this->comp = d; // Set the active component
    }


    /**
     * @brief Function class for the exact/initial/boundary velocity field.
     *
     * Represents the parabolic velocity profile U(y) = 4*Um*y*(H-y)/H^2 for the
     * x-component (u), and zero for the y-component (v). Inherits from
     * MultiComponentFunction to allow selecting u or v via `set_component`.
     *
     * @tparam dim The spatial dimension (expected to be 2).
     */
    template <int dim>
    class Velocity : public MultiComponentFunction<dim>
    {
    public:
      /**
       * @brief Constructor. Initializes the base class with the initial time.
       * @param initial_time The initial time value.
       */
      Velocity(const double initial_time = 0.0); //

      /**
       * @brief Returns the value of the specified velocity component at point p.
       * @param p The point at which to evaluate the velocity.
       * @param component Legacy component argument (ignored, use `set_component`).
       * @return The value of the active velocity component (`this->comp`).
       */
      virtual double value(const dealii::Point<dim>  &p,
                           const unsigned int component = 0) const override; //

      /**
       * @brief Evaluates the active velocity component for a list of points.
       * @param points Vector of points at which to evaluate.
       * @param values Vector to store the computed velocity component values.
       * @param component Legacy component argument (ignored).
       */
      virtual void value_list(const std::vector<dealii::Point<dim>> &points,
                              std::vector<double> &values,
                              const unsigned int component = 0) const override; //
    };

    // --- Implementation of Velocity ---

    template <int dim>
    Velocity<dim>::Velocity(const double initial_time)
      : MultiComponentFunction<dim>(initial_time) // Call base constructor
    {}

    template <int dim>
    double Velocity<dim>::value(const dealii::Point<dim> &p,
                                const unsigned int /*component*/) const // Ignore component argument
    {
      // Assumes dim=2
      if (this->comp == 0) // Evaluate x-velocity (u)
      {
        // Parameters for the parabolic profile (hardcoded)
        const double Um = 1.5; // Max velocity
        const double H  = 4.1; // Channel height (or relevant dimension)
        // Parabolic profile formula
        return 4. * Um * p[1] * (H - p[1]) / (H * H); //
      }
      else // Evaluate y-velocity (v) or other components
        return 0.; //
    }

    template <int dim>
    void Velocity<dim>::value_list(const std::vector<dealii::Point<dim>> &points,
                                   std::vector<double> &values,
                                   const unsigned int /*component*/) const // Ignore component argument
    {
      const unsigned int n_points = points.size(); //
      AssertDimension(values.size(), n_points); // Ensure output vector has correct size
      // Evaluate value() for each point in the list
      for (unsigned int i = 0; i < n_points; ++i)
        values[i] = Velocity<dim>::value(points[i]); //
    }


    /**
     * @brief Function class for the exact/initial/boundary pressure field.
     *
     * Represents a linear pressure drop p(x) = P0 - x.
     *
     * @tparam dim The spatial dimension.
     */
    template <int dim>
    class Pressure : public dealii::Function<dim>
    {
    public:
      /**
       * @brief Constructor. Initializes the base Function class for 1 component.
       * @param initial_time The initial time value.
       */
      Pressure(const double initial_time = 0.0); //

      /**
       * @brief Returns the value of the pressure at point p.
       * @param p The point at which to evaluate the pressure.
       * @param component Component index (must be 0 for scalar pressure).
       * @return The pressure value.
       */
      virtual double value(const dealii::Point<dim>  &p,
                           const unsigned int component = 0) const override; //

      /**
       * @brief Evaluates the pressure for a list of points.
       * @param points Vector of points at which to evaluate.
       * @param values Vector to store the computed pressure values.
       * @param component Component index (must be 0).
       */
      virtual void value_list(const std::vector<dealii::Point<dim>> &points,
                              std::vector<double> &values,
                              const unsigned int component = 0) const override; //
    };

    // --- Implementation of Pressure ---

    template <int dim>
    Pressure<dim>::Pressure(const double initial_time)
      : dealii::Function<dim>(1, initial_time) // Initialize base for 1 component
    {}

    template <int dim>
    double Pressure<dim>::value(const dealii::Point<dim> &p,
                                const unsigned int component) const
    {
      (void)component; // Suppress unused parameter warning
      AssertIndexRange(component, 1); // Ensure component is 0
      // Linear pressure drop formula (hardcoded P0 = 25)
      return 25. - p[0]; //
    }

    template <int dim>
    void Pressure<dim>::value_list(const std::vector<dealii::Point<dim>> &points,
                                   std::vector<double> &values,
                                   const unsigned int component) const
    {
      (void)component; // Suppress unused parameter warning
      AssertIndexRange(component, 1); // Ensure component is 0
      const unsigned int n_points = points.size(); //
      AssertDimension(values.size(), n_points); // Ensure output vector has correct size
      // Evaluate value() for each point in the list
      for (unsigned int i = 0; i < n_points; ++i)
        values[i] = Pressure<dim>::value(points[i]); //
    }

  } // namespace EquationData
} // namespace Step35

#endif // EQUATION_DATA_H