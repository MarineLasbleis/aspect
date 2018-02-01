#include <aspect/simulator_access.h>
#include <aspect/global.h>
#include <aspect/simulator.h>
#include <aspect/simulator/assemblers/interface.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>




namespace aspect
{
  namespace MaterialModel
  {
    using namespace dealii;

    template <int dim>
    class InnerCore : public MaterialModel::Simple<dim>
    {
      public:
        /**
         * Constructor.
         */
        InnerCore ();

        virtual void evaluate(const MaterialModel::MaterialModelInputs<dim> &in,
                              MaterialModel::MaterialModelOutputs<dim> &out) const;

        /**
         * A function that is called at the beginning of each time step to
         * indicate what the model time is for which the boundary values will
         * next be evaluated. For the current class, the function passes to
         * the parsed function what the current time is.
         */
        virtual
        void
        update ();

        /**
         * Declare the parameters this class takes through input files. The
         * default implementation of this function does not describe any
         * parameters. Consequently, derived classes do not have to overload
         * this function if they do not take any runtime parameters.
         */
        static
        void
        declare_parameters (ParameterHandler &prm);

        /**
         * Read the parameters this class declares from the parameter file.
         * The default implementation of this function does not read any
         * parameters. Consequently, derived classes do not have to overload
         * this function if they do not take any runtime parameters.
         */
        virtual
        void
        parse_parameters (ParameterHandler &prm);

        /**
         * A function object representing resistance to phase change at the
         * inner core boundary as a function the position (and, optionally,
         * the model time).
         */
        Functions::ParsedFunction<dim> resistance_to_phase_change;

      private:
        /**
         * Parameters related to the phase transition.
         */
        double transition_radius;
        double transition_width;
        double transition_temperature;
        double transition_clapeyron_slope;
        double transition_density_change;
        bool compute_quadratic_pressure_profile;

        /**
         * A function object representing the hydrostatic pressure in the
         * inner core as a function of radius. This is needed to compute
         * the depth of phase transitions.
         * Note that we can not simply use the adiabatic pressure here,
         * as the equations are solved for the dynamic pressure, and the
         * adiabatic conditions are used as initial guess for the solution.
         */
        Functions::ParsedFunction<1> hydrostatic_pressure_profile;

        /**
         * Percentage of material that has already undergone the phase
         * transition to the higher-pressure material.
         */
        virtual
        double
        phase_function (const Point<dim> &position,
                        const double temperature) const;

        /**
         * Hydrostatic pressure profile.
         */
        virtual
        double
        hydrostatic_pressure (const double radius) const;
    };

  }
}

namespace aspect
{
  namespace MaterialModel
  {
    template <int dim>
    InnerCore<dim>::InnerCore ()
      :
      resistance_to_phase_change (1),
      hydrostatic_pressure_profile (1)
    {}

    template <int dim>
    double
    InnerCore<dim>::
    phase_function (const Point<dim> &position,
                    const double temperature) const
    {
      // We need to convert the depth to pressure,
      // keeping in mind that for this model, the pressure is only the dynamic pressure,
      // so we have to get the hydrostatic pressure explicitly as an input.
      const double radius = this->get_geometry_model().maximal_depth() - this->get_geometry_model().depth(position);
      Assert(radius >= 0,
             ExcMessage("There is a point in the model where the depth is larger "
                        "than the maximal depth of this geometry."));

      const double pressure_deviation = hydrostatic_pressure(radius) - hydrostatic_pressure(transition_radius);

      double depth_deviation = transition_radius - radius;
      if (std::fabs(pressure_deviation) > 100.0 * std::numeric_limits<double>::epsilon()
          * (std::fabs(hydrostatic_pressure(radius)) + std::fabs(hydrostatic_pressure(transition_radius)))/2)
        depth_deviation *= (1.0 - transition_clapeyron_slope * (temperature - transition_temperature) / pressure_deviation);

      double phase_func;
      // use delta function for width = 0
      if (transition_width == 0)
        phase_func = (depth_deviation > 0) ? 1 : 0;
      else
        phase_func = 0.5 * (1.0 + std::tanh(depth_deviation / transition_width));
      return phase_func;
    }


    template <int dim>
    double
    InnerCore<dim>::
    hydrostatic_pressure (const double radius) const
    {
      if (compute_quadratic_pressure_profile)
        {
          // Compute a quadratic hydrostatic pressure profile, based on a linear gravity model.
          AssertThrow (dynamic_cast<const GravityModel::RadialLinear<dim>*>(&this->get_gravity_model())
                       != 0,
                       ExcMessage ("Automatic computation of the hydrostatic pressure profile is "
                                   "only implemented for the 'radial linear' gravity model."));

          const double max_radius = this->get_geometry_model().maximal_depth();
          const Point<dim> surface_point = this->get_geometry_model().representative_point(0);
          const double gravity_magnitude = this->get_gravity_model().gravity_vector(surface_point).norm();

          // The gravity is zero in the center of the Earth, and we assume the density to be constant and equal to 1.
          // We fix the surface pressure to 0 (and we are only interested in pressure differences anyway).
          return gravity_magnitude * 0.5 * (1.0 - std::pow(radius/max_radius,2));
        }
      else
        return hydrostatic_pressure_profile.value(Point<1>(radius));
    }


    template <int dim>
    void
    InnerCore<dim>::
    evaluate(const MaterialModel::MaterialModelInputs<dim> &in,
             MaterialModel::MaterialModelOutputs<dim> &out) const
    {
      // First, we use the material descriptions of the 'simple' material model to fill all of the material
      // model outputs. Below, we will then overwrite selected properties (the specific heat) to make the
      // product of density and specific heat a constant.
      Simple<dim>::evaluate(in, out);

      // We want the right-hand side of the momentum equation to be (- Ra T gravity) and
      // density * cp to be 1
      for (unsigned int q=0; q < in.position.size(); ++q)
        {
          out.densities[q] = - out.thermal_expansion_coefficients[q] * in.temperature[q]
                             + phase_function (in.position[q], in.temperature[q]) * transition_density_change;
          if (std::abs(out.densities[q]) > 0.0)
            out.specific_heat[q] /= out.densities[q];
        }
    }


    template <int dim>
    void
    InnerCore<dim>::update()
    {
      // we get time passed as seconds (always) but may want
      // to reinterpret it in years
      if (this->convert_output_to_years())
        resistance_to_phase_change.set_time (this->get_time() / year_in_seconds);
      else
        resistance_to_phase_change.set_time (this->get_time());

      if (this->convert_output_to_years())
        hydrostatic_pressure_profile.set_time (this->get_time() / year_in_seconds);
      else
        hydrostatic_pressure_profile.set_time (this->get_time());
    }


    template <int dim>
    void
    InnerCore<dim>::declare_parameters (ParameterHandler &prm)
    {
      Simple<dim>::declare_parameters (prm);

      prm.enter_subsection("Material model");
      {
        prm.enter_subsection("Inner core");
        {
          prm.enter_subsection("Phase change resistance function");
          {
            Functions::ParsedFunction<dim>::declare_parameters (prm, 1);
          }
          prm.leave_subsection();

          prm.enter_subsection("Hydrostatic pressure function");
          {
            Functions::ParsedFunction<1>::declare_parameters (prm, 1);
          }
          prm.leave_subsection();

          prm.declare_entry ("Phase transition radius", "0.0",
                             Patterns::Double (0),
                             "The distance from the center of the Earth where the phase "
                             "transition occurs. "
                             "Units: m.");
          prm.declare_entry ("Phase transition width", "0.0",
                             Patterns::Double (0),
                             "The width of the phase transition. The argument of the phase function "
                             "is scaled with this value, leading to a jump between phases "
                             "for a value of zero and a gradual transition for larger values. "
                             "Units: m.");
          prm.declare_entry ("Phase transition temperature", "0.0",
                             Patterns::Double (0),
                             "The temperature at which the phase transition occurs in the depth "
                             "given by the 'Phase transition depth' parameter.  Higher or lower "
                             "temperatures lead to phase transition occurring in shallower or greater "
                             "depths, depending on the Clapeyron slope given in 'Phase transition "
                             "Clapeyron slope'. "
                             "Units: K.");
          prm.declare_entry ("Phase transition Clapeyron slope", "0.0",
                             Patterns::Double (),
                             "The Clapeyron slope of the phase transition. A positive "
                             "Clapeyron slope indicates that the phase transition will occur in "
                             "a greater depth if the temperature is higher than the one given in "
                             "Phase transition temperatures (and in a smaller depth if the "
                             "temperature is smaller than the one given in Phase transition temperatures). "
                             "For negative Clapeyron slopes, the effect is in the opposite direction. "
                             "Units: Pa/K.");
          prm.declare_entry ("Phase transition density change", "0.0",
                             Patterns::Double (),
                             "The density change that occurs across the phase transition. "
                             "A positive value means that the density increases with depth. "
                             "Units: kg/m$^3$.");
          prm.declare_entry ("Compute quadratic pressure profile from gravity", "true",
                             Patterns::Bool (),
                             "Whether to automatically compute the hydrostatic pressure profile "
                             "(that is used to compute the location of phase transitions) from "
                             "the magnitude of the gravity, assuming a linear gravity profile "
                             "and a constant density (if true), or to use the function that is "
                             "given in 'Hydrostatic pressure function' (if false).");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }


    template <int dim>
    void
    InnerCore<dim>::parse_parameters (ParameterHandler &prm)
    {
      Simple<dim>::parse_parameters (prm);

      prm.enter_subsection("Material model");
      {
        prm.enter_subsection("Inner core");
        {
          prm.enter_subsection("Phase change resistance function");
          try
            {
              resistance_to_phase_change.parse_parameters (prm);
            }
          catch (...)
            {
              std::cerr << "ERROR: FunctionParser failed to parse\n"
                        << "\t'Phase boundary model.Function'\n"
                        << "with expression\n"
                        << "\t'" << prm.get("Function expression") << "'"
                        << "More information about the cause of the parse error \n"
                        << "is shown below.\n";
              throw;
            }
          prm.leave_subsection();

          prm.enter_subsection("Hydrostatic pressure function");
          try
            {
              hydrostatic_pressure_profile.parse_parameters (prm);
            }
          catch (...)
            {
              std::cerr << "ERROR: FunctionParser failed to parse\n"
                        << "\t'Hydrostatic pressure model.Function'\n"
                        << "with expression\n"
                        << "\t'" << prm.get("Function expression") << "'"
                        << "More information about the cause of the parse error \n"
                        << "is shown below.\n";
              throw;
            }
          prm.leave_subsection();

          transition_radius          = prm.get_double ("Phase transition radius");
          transition_width           = prm.get_double ("Phase transition width");
          transition_temperature     = prm.get_double ("Phase transition temperature");
          transition_clapeyron_slope = prm.get_double ("Phase transition Clapeyron slope");
          transition_density_change  = prm.get_double ("Phase transition density change");
          compute_quadratic_pressure_profile = prm.get_bool ("Compute quadratic pressure profile from gravity");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }

  }
}



namespace aspect
{
  namespace HeatingModel
  {
    using namespace dealii;

    /**
     * A class that implements a constant radiogenic heating rate.
     *
     * @ingroup HeatingModels
     */
    template <int dim>
    class ConstantCoreHeating : public Interface<dim>
    {
      public:
        /**
         * Return the heating terms. For the current class, this
         * function obviously simply returns a constant value.
         */
        virtual
        void
        evaluate (const MaterialModel::MaterialModelInputs<dim> &material_model_inputs,
                  const MaterialModel::MaterialModelOutputs<dim> &material_model_outputs,
                  HeatingModel::HeatingModelOutputs &heating_model_outputs) const;

        /**
         * @name Functions used in dealing with run-time parameters
         * @{
         */
        /**
         * Declare the parameters this class takes through input files.
         */
        static
        void
        declare_parameters (ParameterHandler &prm);

        /**
         * Read the parameters this class declares from the parameter file.
         */
        virtual
        void
        parse_parameters (ParameterHandler &prm);
        /**
         * @}
         */

      private:
        double radiogenic_heating_rate;
    };
  }
}

namespace aspect
{
  namespace HeatingModel
  {
    template <int dim>
    void
    ConstantCoreHeating<dim>::
    evaluate (const MaterialModel::MaterialModelInputs<dim> &/*material_model_inputs*/,
              const MaterialModel::MaterialModelOutputs<dim> &/*material_model_outputs*/,
              HeatingModel::HeatingModelOutputs &heating_model_outputs) const
    {
      for (unsigned int q=0; q<heating_model_outputs.heating_source_terms.size(); ++q)
        {
          // return a constant value
          heating_model_outputs.heating_source_terms[q] = radiogenic_heating_rate;
          heating_model_outputs.lhs_latent_heat_terms[q] = 0.0;
        }
    }



    template <int dim>
    void
    ConstantCoreHeating<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Heating model");
      {
        prm.enter_subsection("Constant core heating");
        {
          prm.declare_entry ("Radiogenic heating rate", "0e0",
                             Patterns::Double (0),
                             "The specific rate of heating due to radioactive decay (or other bulk sources "
                             "you may want to describe). This parameter corresponds to the variable "
                             "$H$ in the temperature equation stated in the manual, and the heating "
                             "term is $\rho H$. Units: W/kg.");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }



    template <int dim>
    void
    ConstantCoreHeating<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Heating model");
      {
        prm.enter_subsection("Constant core heating");
        {
          radiogenic_heating_rate    = prm.get_double ("Radiogenic heating rate");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }
  }
}


// explicit instantiations
namespace aspect
{
  namespace MaterialModel
  {
    ASPECT_REGISTER_MATERIAL_MODEL(InnerCore,
                                   "inner core material",
                                   "A simple material model that is like the "
                                   "'Simple' model, but has a constant $\rho c_p$, "
                                   "and implements a function that characterizes the "
                                   "resistance to melting/freezing at the inner core "
                                   "boundary.")
  }

  namespace HeatingModel
  {
    ASPECT_REGISTER_HEATING_MODEL(ConstantCoreHeating,
                                  "constant core heating",
                                  "Implementation of a model in which the heating "
                                  "rate is constant.")
  }
}





namespace aspect
{
  /**
   * A new assembler class that implements boundary conditions for the
   * normal stress and the normal velocity that take into account the
   * rate of phase change (melting/freezing) at the inner-outer core
   * boundary. The model is based on Deguen, Alboussiere, and Cardin
   * (2013), Thermal convection in Earth’s inner core with phase change
   * at its boundary. GJI, 194, 1310-133.
   *
   * The mechanical boundary conditions for the inner core are
   * tangential stress-free and continuity of the normal stress at the
   * inner-outer core boundary. For the non-dimensional equations, that
   * means that we can define a 'phase change number' $\mathcal{P}$ so
   * that the normal stress at the boundary is $-\mathcal{P} u_r$ with
   * the radial velocity $u_r$. This number characterizes the resistance
   * to phase change at the boundary, with $\mathcal{P}\rightarrow\infty$
   * corresponding to infinitely slow melting/freezing (free slip
   * boundary), and $\mathcal{P}\rightarrow0$ corresponding to
   * instantaneous melting/freezing (zero normal stress, open boundary).
   *
   * In the weak form, this results in boundary conditions of the form
   * of a surface integral:
   * $$\int_S \mathcal{P} (\mathbf u \cdot \mathbf n) (\mathbf v \cdot \mathbf n) dS,$$,
   * with the normal vector $\mathbf n$.
   *
   * The function value of $\mathcal{P}$ is taken from the inner core
   * material model.
   */
  template <int dim>
  class PhaseBoundaryAssembler :
    public aspect::Assemblers::Interface<dim>,
    public SimulatorAccess<dim>
  {

    public:

      virtual
      void
      execute (internal::Assembly::Scratch::ScratchBase<dim>   &scratch_base,
               internal::Assembly::CopyData::CopyDataBase<dim> &data_base) const
      {
        internal::Assembly::Scratch::StokesSystem<dim> &scratch = dynamic_cast<internal::Assembly::Scratch::StokesSystem<dim>& > (scratch_base);
        internal::Assembly::CopyData::StokesSystem<dim> &data = dynamic_cast<internal::Assembly::CopyData::StokesSystem<dim>& > (data_base);

        const Introspection<dim> &introspection = this->introspection();
        const FiniteElement<dim> &fe            = this->get_fe();
        const unsigned int stokes_dofs_per_cell = data.local_dof_indices.size();
        const unsigned int n_q_points           = scratch.face_finite_element_values.n_quadrature_points;

        //assemble force terms for the matrix for all boundary faces
        if (scratch.cell->face(scratch.face_number)->at_boundary())
          {
            scratch.face_finite_element_values.reinit (scratch.cell, scratch.face_number);

            for (unsigned int q=0; q<n_q_points; ++q)
              {
                const double P = dynamic_cast<const MaterialModel::InnerCore<dim>&>
                                 (this->get_material_model()).resistance_to_phase_change
                                 .value(scratch.material_model_inputs.position[q]);

                for (unsigned int i = 0, i_stokes = 0; i_stokes < stokes_dofs_per_cell; /*increment at end of loop*/)
                  {
                    if (introspection.is_stokes_component(fe.system_to_component_index(i).first))
                      {
                        scratch.phi_u[i_stokes] = scratch.face_finite_element_values[introspection
                                                                                     .extractors.velocities].value(i, q);
                        ++i_stokes;
                      }
                    ++i;
                  }

                const Tensor<1,dim> normal_vector = scratch.face_finite_element_values.normal_vector(q);
                const double JxW = scratch.face_finite_element_values.JxW(q);

                // boundary term: P*u*n*v*n*JxW(q)
                for (unsigned int i=0; i<stokes_dofs_per_cell; ++i)
                  for (unsigned int j=0; j<stokes_dofs_per_cell; ++j)
                    data.local_matrix(i,j) += P *
                                              scratch.phi_u[i] *
                                              normal_vector *
                                              scratch.phi_u[j] *
                                              normal_vector *
                                              JxW;
              }
          }
      }
  };

  template <int dim>
  void set_assemblers_phase_boundary(const SimulatorAccess<dim> &simulator_access,
                                     Assemblers::Manager<dim> &assemblers)
  {
    AssertThrow (dynamic_cast<const MaterialModel::InnerCore<dim>*>
                 (&simulator_access.get_material_model()) != 0,
                 ExcMessage ("The phase boundary assembler can only be used with the "
                             "material model 'inner core material'!"));

    PhaseBoundaryAssembler<dim> *phase_boundary_assembler = new PhaseBoundaryAssembler<dim>();
    assemblers.stokes_system_on_boundary_face.push_back (std_cxx11::unique_ptr<PhaseBoundaryAssembler<dim> > (phase_boundary_assembler));
  }
}

template <int dim>
void signal_connector (aspect::SimulatorSignals<dim> &signals)
{
  signals.set_assemblers.connect (&aspect::set_assemblers_phase_boundary<dim>);
}

ASPECT_REGISTER_SIGNALS_CONNECTOR(signal_connector<2>,
                                  signal_connector<3>)
