/*
  Copyright (C) 2011 - 2018 by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
 any later version.

  ASPECT is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with ASPECT; see the file LICENSE.  If not see
  <http://www.gnu.org/licenses/>.
*/

#include <aspect/material_model/melt_global.h>


namespace aspect
{
  namespace MaterialModel
  {
    using namespace dealii;

    template <int dim>
    class MeltSRB : public MaterialModel::MeltGlobal<dim>
    {
        public:
        /**
         * Constructor.
         */
        MeltSRB ();

        virtual void evaluate(const typename Interface<dim>::MaterialModelInputs &in,
                              typename Interface<dim>::MaterialModelOutputs &out) const;

        virtual double reference_darcy_coefficient () const;

    }
  }
}


namespace aspect
{
    namespace MaterialModel
    {
        template <int dim>
        double
        MeltSRB<dim>::
        reference_darcy_coefficient () const
        {
            return reference_permeability;   // the reference permeability is directly the one provided in the parameter file.
        }

        template <int dim>
        void
        MeltSRB<dim>::
        evaluate(const typename Interface<dim>::MaterialModelInputs &in, typename Interface<dim>::MaterialModelOutputs &out) const
        {
            for (unsigned int i=0; i<in.position.size(); ++i)
            {
                out.viscosities[i] = eta_0;
                for (unsigned int c=0; c<in.composition[i].size(); ++c)
                {
                out.reaction_terms[i][c] = 0.0;

                if (this->get_parameters().use_operator_splitting && reaction_rate_out != nullptr)
                    reaction_rate_out->reaction_rates[i][c] = 0.0;
                }

                if (this->include_melt_transport())
                {
                    const unsigned int porosity_idx = this->introspection().compositional_index_for_name("porosity");
                    const double porosity = std::min(1.0, std::max(in.composition[i][porosity_idx],0.0));

                    // calculate viscosity based on local melt
                    out.viscosities[i] *= std::max(1-porosity, 1e-6);
                }

                out.entropy_derivative_pressure[i]    = 0.0;
                out.entropy_derivative_temperature[i] = 0.0;
                out.thermal_expansion_coefficients[i] = thermal_expansivity;
                out.specific_heat[i] = reference_specific_heat;
                out.thermal_conductivities[i] = thermal_conductivity;
                out.compressibilities[i] = 0.0;
            }
            
            MeltOutputs<dim> *melt_out = out.template get_additional_output<MeltOutputs<dim> >();

            if (melt_out != nullptr)
            {
                const unsigned int porosity_idx = this->introspection().compositional_index_for_name("porosity");

                for (unsigned int i=0; i<in.position.size(); ++i)
                {
                    double porosity = std::max(in.composition[i][porosity_idx],0.0);

                    melt_out->fluid_viscosities[i] = eta_f;
                    melt_out->permeabilities[i] = reference_permeability * std::pow(porosity,3) * std::pow(1.0-porosity,2);
                    melt_out->fluid_density_gradients[i] = Tensor<1,dim>();
                    melt_out->fluid_densities[i] = reference_rho_f
                    melt_out->compaction_viscosities[i] = xi_0 * std::max(1- porosity, 1e-6)/std::max(porosity, 1e-6);

                }
            }
        }


    }
}





// explicit instantiations
namespace aspect
{
  namespace MaterialModel
  {
    ASPECT_REGISTER_MATERIAL_MODEL(MeltSRB,
                                   "melt srb",
                                   "A melt model based on Sramek, Bercovici Ricard 2007,"
                                   "without the melting or freezing. It is based on the "
                                   "global melt model. ")
  }