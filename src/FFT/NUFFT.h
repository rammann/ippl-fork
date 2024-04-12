#ifndef NUFFT_GUARD
#define NUFFT_GUARD

#include <functional>
#include <finufft.h>
#include <Particle/ParticleAttrib.h>
#include <complex> 


namespace ippl{


namespace detail{
template <class T>
struct finufftType {};

template <>
struct finufftType<float> {
    std::function<int(int, int, int64_t*, int, int, float, finufftf_plan*, finufft_opts*)> makeplan     = finufftf_makeplan; 
    std::function<int(finufftf_plan, int, float*, float*, float*, int, float*, float*, float*)> setpts  = finufftf_setpts; 
    std::function<int(finufftf_plan, std::complex<float>*, std::complex<float>*)> execute               = finufftf_execute; 
    std::function<int(finufftf_plan)> destroy                                                           = finufftf_destroy;
            
    using complexType = std::complex<float>;
    using plan_t      = finufftf_plan;
};

template <>
struct finufftType<double> {
    std::function<int(int, int, int64_t*, int, int, double, finufft_plan*, finufft_opts*)> makeplan         = finufft_makeplan; 
    std::function<int(finufft_plan, int, double*, double*, double*, int, double*, double*, double*)> setpts = finufft_setpts; 
    std::function<int(finufft_plan, std::complex<double>*, std::complex<double>*)> execute                  = finufft_execute; 
    std::function<int(finufft_plan)> destroy                                                                = finufft_destroy; 
            
    using complexType = std::complex<double>;
    using plan_t      = finufft_plan;
};
}// namespace ippl::detail


/**
 * Nufft class based on Sri's implementation on the 131 branch
*/
template <size_t Dim, class T, class Mesh, class Centering>
class NUFFT{

public:

    typedef FieldLayout<Dim> Layout_t;
    typedef Kokkos::complex<T> KokkosComplex_t;
    typedef Field<KokkosComplex_t,Dim,Mesh,Centering> ComplexField_t;

    using complexType = typename detail::finufftType<T>::complexType;
    using plan_t = typename detail::finufftType<T>::plan_t;
    using view_field_type = typename detail::ViewType<complexType, 3, Kokkos::LayoutLeft>::view_type;
    using view_particle_real_type = typename detail::ViewType<T, 1, Kokkos::LayoutLeft>::view_type;
    using view_particle_complex_type = typename detail::ViewType<complexType, 1, Kokkos::LayoutLeft>::view_type;

    NUFFT() = default;

    NUFFT(const Layout_t& layout, const detail::size_type& localNp, int type, const ParameterList& params){

        std::array<int64_t, 3> nmodes;

        const NDIndex<Dim>& lDom = layout.getLocalNDIndex();

        nmodes.fill(1);

        for(size_t d = 0; d < Dim; ++d) {
            nmodes[d] = lDom[d].length();;
        }

        type_m = type;
        if(tempField_m.size() < lDom.size()) {
            Kokkos::realloc(tempField_m, lDom[0].length(), lDom[1].length(), lDom[2].length());
        }
        for(size_t d = 0; d < Dim; ++d) {
            if(tempR_m[d].size() < localNp) {
                Kokkos::realloc(tempR_m[d], localNp);
            }
        }
        if(tempQ_m.size() < localNp) {
            Kokkos::realloc(tempQ_m, localNp);
        }
        setup(nmodes, params);
    }
    
    ~NUFFT(){
        ier_m = nufft_m.destroy(plan_m);
    }
    
    template<class... Properties>
    void transform(const ParticleAttrib< Vector<T, Dim>, Properties... >& R, ParticleAttrib<T, Properties... >& Q, ComplexField_t& f){
        
        // Views of the field f, positions R, charge Q for the transform 
        auto fview = f.getView();
        auto Rview = R.getView();
        auto Qview = Q.getView();
        const int nghost = f.getNghost();

        // Get properties about the domain dimensions
        auto localNp = R.getParticleCount();
        const Layout_t& layout = f.getLayout(); 
        const UniformCartesian<T, Dim>& mesh = f.get_mesh();
        const Vector<T, Dim>& dx = mesh.getMeshSpacing();
        const Vector<T, Dim>& origin = mesh.getOrigin();
        const auto& domain = layout.getDomain();
        Vector<T, Dim> Len;
        Vector<int, Dim> N;

        for (unsigned d=0; d < Dim; ++d) {
            N[d] = domain[d].length();
            Len[d] = dx[d] * N[d];
        }

        std::cout << origin[0] << " " << origin[1] << " " << origin[2] << "\n";
        std::cout << Len[0] << " " << Len[1] << " " << Len[2] << "\n";

        const double pi = std::acos(-1.0);

        auto tempField = tempField_m;
        auto tempQ = tempQ_m;
        Kokkos::View<T*,Kokkos::LayoutLeft> tempR[3] = {};
        for(size_t d = 0; d < Dim; ++d) {
            tempR[d] = tempR_m[d];
        }
        using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;

        Kokkos::parallel_for("copy from field data NUFFT",
                             mdrange_type({nghost, nghost, nghost},
                                          {fview.extent(0) - nghost,
                                           fview.extent(1) - nghost,
                                           fview.extent(2) - nghost
                                          }),
                             KOKKOS_LAMBDA(const size_t i,
                                           const size_t j,
                                           const size_t k)
                             {
                                 tempField(i-nghost, j-nghost, k-nghost).real(fview(i, j, k).real()); //= fview(i, j, k).real();
                                 tempField(i-nghost, j-nghost, k-nghost).imag(fview(i, j, k).imag()); //= fview(i, j, k).imag();
                             });


        Kokkos::parallel_for("copy from particle data NUFFT",
                             localNp,
                             KOKKOS_LAMBDA(const size_t i)
                             {
                                 for(size_t d = 0; d < Dim; ++d) {
                                    tempR[d](i) = Rview(i)[d];
                                    //tempR[d](i) = (Rview(i)[d]) * (2.0 * pi / Len[d]);
                                    //tempR[d](i) = (Rview(i)[d])  / Len[d];
                                    //tempR[d](i) = -pi + ((2 * pi / Len[d]) * (Rview(i)[d] - origin[d]));
                                 }
                                 tempQ(i).real(Qview(i)); // = Qview(i);
                                 tempQ(i).imag(0.0); // = 0.0;
                             });

        ier_m = nufft_m.setpts(plan_m, localNp, tempR[0].data(), tempR[1].data(), tempR[2].data(), 0, 
                     NULL, NULL, NULL);

        ier_m = nufft_m.execute(plan_m, tempQ.data(), tempField.data());
        Kokkos::fence();


        if(type_m == 1) { 
            Kokkos::parallel_for("copy to field data NUFFT",
                                 mdrange_type({nghost, nghost, nghost},
                                              {fview.extent(0) - nghost,
                                               fview.extent(1) - nghost,
                                               fview.extent(2) - nghost
                                              }),
                                 KOKKOS_LAMBDA(const size_t i,
                                               const size_t j,
                                               const size_t k)
                                 {
                                     fview(i, j, k).real(tempField(i-nghost, j-nghost, k-nghost).real()); // = tempField(i-nghost, j-nghost, k-nghost).real();
                                     fview(i, j, k).imag(tempField(i-nghost, j-nghost, k-nghost).imag()); // = tempField(i-nghost, j-nghost, k-nghost).imag();
                                 });
        }
        else if(type_m == 2) {
            Kokkos::parallel_for("copy to particle data NUFFT",
                                 localNp,
                                 KOKKOS_LAMBDA(const size_t i)
                                 {
                                     Qview(i) = tempQ(i).real();
                                 });
        }
    }


private:

    void setup(std::array<int64_t, 3>& nmodes, const ParameterList& params){
        
        finufft_opts opts;
	    finufft_default_opts(&opts);
        tol_m = 1e-6;

        if(!params.get<bool>("use_finufft_defaults")) {
           tol_m = params.get<T>("tolerance");
           //opts.method = params.get<int>("cpu_method");
           opts.spread_sort = params.get<int>("cpu_sort");
           opts.spread_kerevalmeth = params.get<int>("cpu_kerevalmeth");
        }

        //opts.gpu_maxbatchsize = 0;

        int iflag;

        if(type_m == 1) {
            iflag = -1;
        }
        else if(type_m == 2) {
            iflag = 1;
        }
        else {
            throw std::logic_error("Only type 1 and type 2 NUFFT are allowed now");
        }

        //dim in cufinufft is int
        int dim = static_cast<int>(Dim);
        ier_m = nufft_m.makeplan(type_m, dim, nmodes.data(), iflag, 1, tol_m,
                       		 &plan_m, &opts);
    }

    detail::finufftType<T> nufft_m;
    plan_t plan_m;
    int ier_m;
    T tol_m;
    int type_m;
    view_field_type tempField_m;
    view_particle_real_type tempR_m[3] = {};
    view_particle_complex_type tempQ_m;
};



} // namespace ippl



#endif