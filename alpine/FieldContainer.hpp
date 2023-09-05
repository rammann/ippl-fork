#ifndef IPPL_FIELD_CONTAINER_H
#define IPPL_FIELD_CONTAINER_H

#include <memory>
#include "Manager/BaseManager.h"

    // Define the FieldsContainer class
    template <typename T, unsigned Dim = 3>
    class FieldContainer{
    
    public:
        FieldContainer(Vector_t<double, Dim> hr, Vector_t<double, Dim> rmin,
                        Vector_t<double, Dim> rmax, ippl::e_dim_tag decomp[Dim])
            : hr_m(hr), rmin_m(rmin), rmax_m(rmax) {
            for (unsigned int i = 0; i < Dim; i++) {
                decomp_m[i] = decomp[i];
            }
     }
    
    VField_t<T, Dim> E_m;
    Field_t<Dim> rho_m;
    Field_t<Dim> phi_m;

    Vector_t<T, Dim> nr_m;

    ippl::e_dim_tag decomp_m[Dim];

    Vector_t<double, Dim> hr_m;
    Vector_t<double, Dim> rmin_m;
    Vector_t<double, Dim> rmax_m;
    
    void initializeFields(Mesh_t<Dim>& mesh, FieldLayout_t<Dim>& fl) {
        E_m.initialize(mesh, fl);
        rho_m.initialize(mesh, fl);
    }
    
    
    };

#endif