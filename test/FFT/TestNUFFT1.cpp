#include "Ippl.h"
#include "Utility/ParameterList.h"

#include <iostream>
#include <typeinfo>
#include <array>
#include <Kokkos_Random.hpp>
#include <random>

template<class PLayout>
struct Bunch : public ippl::ParticleBase<PLayout>
{

    Bunch(PLayout& playout)
    : ippl::ParticleBase<PLayout>(playout)
    {
        this->addAttribute(Q);
    }

    ~Bunch(){ }
    
    typedef ippl::ParticleAttrib<double> charge_container_type;
    charge_container_type Q;

};

template <typename T, class GeneratorPool, unsigned Dim>
struct generate_random {

  using view_type = typename ippl::detail::ViewType<T, 1>::view_type;
  using value_type  = typename T::value_type;
  using view_type_scalar = typename ippl::detail::ViewType<value_type, 1>::view_type;
  // Output View for the random numbers
  view_type x;

  view_type_scalar Q;

  // The GeneratorPool
  GeneratorPool rand_pool;

  T minU, maxU;

  // Initialize all members
  generate_random(view_type x_,view_type_scalar Q_,  GeneratorPool rand_pool_, 
                  T& minU_, T& maxU_)
      : x(x_), Q(Q_), rand_pool(rand_pool_), 
        minU(minU_), maxU(maxU_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const size_t i) const {
    // Get a random number state from the pool for the active thread
    typename GeneratorPool::generator_type rand_gen = rand_pool.get_state();

    for (unsigned d = 0; d < Dim; ++d) {
        x(i)[d] = rand_gen.drand(minU[d], maxU[d]);
    }
    Q(i) = rand_gen.drand(0.0, 1.0);

    // Give the state back, which will allow another thread to acquire it
    rand_pool.free_state(rand_gen);
  }
};

int main(int argc, char* argv[]){

     
    ippl::initialize(argc, argv);
    {
        typedef ippl::Vector<double, 3> Vector_t;
        constexpr unsigned int dim = 3;
        using Mesh_t               = ippl::UniformCartesian<double, dim>;
        using Centering_t          = Mesh_t::DefaultCentering;

        const double pi = std::acos(-1.0);

        typedef ippl::ParticleSpatialLayout<double, 3> playout_type;
        typedef Bunch<playout_type> bunch_type;

        // Points per dim and index setup for complex field
        ippl::Vector<int, dim> pt = {32, 32, 32};
        ippl::Index I(pt[0]);
        ippl::Index J(pt[1]);
        ippl::Index K(pt[2]);
        ippl::NDIndex<dim> owned(I, J, K);

        // Determine SERIAL, PARALLEL dims
        std::array<bool, dim> isParallel;  
        isParallel.fill(false);

        // Field Layout
        ippl::FieldLayout<dim> layout(MPI_COMM_WORLD, owned, isParallel);

        // Grid spacing
        std::array<double, dim> dx = {
        2 * pi / double(pt[0]),
        2 * pi / double(pt[1]),
        2 * pi / double(pt[2]),
        };
        Vector_t hx = {dx[0], dx[1], dx[2]};
        Vector_t origin = {0, 0, 0};
        ippl::UniformCartesian<double, 3> mesh(owned, hx, origin);
        typedef ippl::Field<Kokkos::complex<double>, dim, Mesh_t, Centering_t> field_type;

        field_type field(mesh, layout);
        field_type field_dft(mesh, layout);


        //playout_type pl(layout, mesh);
        playout_type pl;

        bunch_type bunch(pl);
        bunch.setParticleBC(ippl::BC::NO);
    
        using size_type = ippl::detail::size_type;
        
        // Number of points
        size_type Np = std::pow(16,3);
        
        // Default finufft parameters
        ippl::ParameterList fftParams;
        fftParams.add("use_finufft_defaults", true); 

        typedef ippl::NUFFT<3,double,Mesh_t, Centering_t> NUFFT_type;

        std::unique_ptr<NUFFT_type> nufft;
        // Type 1 transform
        int type = 1;
        

        // Random point generation
        Vector_t minU = origin;
        Vector_t maxU = {dx[0]*pt[0], dx[1]*pt[1], dx[2]*pt[2]};

        size_type nloc = Np/ippl::Comm->size();

        bunch.create(nloc);
        
        nufft = std::make_unique<NUFFT_type>(layout, Np, type, fftParams);
        
        Kokkos::Random_XorShift64_Pool<> rand_pool64((size_type)(42));
        
        Kokkos::parallel_for(nloc,
                            generate_random<Vector_t, Kokkos::Random_XorShift64_Pool<>, dim>(
                            bunch.R.getView(), bunch.Q.getView(), rand_pool64, minU, maxU));
        
        // Type 1 transform
        nufft->transform(bunch.R, bunch.Q, field);
        
        auto field_result = Kokkos::create_mirror_view_and_copy(
                            Kokkos::HostSpace(), field.getView());

        ippl::Vector<int, 3> kVec;
        kVec[0] = (int)(0.37 * pt[0]);
        kVec[1] = (int)(0.26 * pt[1]);
        kVec[2] = (int)(0.13 * pt[2]);

        const int nghost = field.getNghost();

        int iInd = (pt[0]/2 + kVec[0] + nghost);
        int jInd = (pt[1]/2 + kVec[1] + nghost);
        int kInd = (pt[2]/2 + kVec[2] + nghost);


        Kokkos::complex<double> reducedValue(0.0, 0.0);

        auto Rview = bunch.R.getView();
        auto Qview = bunch.Q.getView();

        Kokkos::complex<double> imag = {0.0, 1.0};

        Kokkos::parallel_reduce("NUDFT type1", nloc,
                                KOKKOS_LAMBDA(const size_t idx, Kokkos::complex<double>& valL) {

                                    double arg = 0.0;
                                    for(size_t d = 0; d < dim; ++d) {
                                        arg += kVec[d]*Rview(idx)[d];
                                    }

                                    valL += (Kokkos::cos(arg) 
                                    - imag * Kokkos::sin(arg)) * Qview(idx);
                                }, Kokkos::Sum<Kokkos::complex<double>>(reducedValue));
        
        double abs_error_real = std::fabs(reducedValue.real() - field_result(iInd, jInd, kInd).real());
        double rel_error_real = std::fabs(reducedValue.real() - field_result(iInd, jInd, kInd).real()) /std::fabs(reducedValue.real());
        double abs_error_imag = std::fabs(reducedValue.imag() - field_result(iInd, jInd, kInd).imag());
        double rel_error_imag = std::fabs(reducedValue.imag() - field_result(iInd, jInd, kInd).imag()) /std::fabs(reducedValue.imag());
    
        std::cout << "Abs Error in real part: " << std::setprecision(16) 
                << abs_error_real << " Rel. error in real part: " << std::setprecision(16) << rel_error_real << std::endl;
        std::cout << "Abs Error in imag part: " << std::setprecision(16) 
                << abs_error_imag << " Rel. error in imag part: " << std::setprecision(16) << rel_error_imag << std::endl;
        std::cout << "Field result: " << std::setprecision(16) 
                << field_result(iInd,jInd,kInd).real() << " " << std::setprecision(16) << field_result(iInd,jInd,kInd).imag() 
                << "index: " << iInd << "," << jInd << "," << kInd << std::endl;

        
    }
    ippl::finalize();

    return 0;
}