#ifndef ORTHOTREEADAPTORS
#define ORTHOTREEADAPTORS

#include <algorithm>
#include <climits>
#include <concepts>
//#include <execution>
#include <functional>
#include <iterator>
#include <numeric>
#include <optional>
#include <type_traits>
#include <stdexcept>

#include <array>
#include <bitset>
#include <map>
#include <unordered_map>
#include <queue>
#include <set>
#include <span>
#include <vector>
#include <tuple>

#include <assert.h>
#include <math.h>

#ifndef autoc
#define autoc auto const
#define undef_autoc
#endif

#ifndef autoce
#define autoce auto constexpr
#define undef_autoce
#endif

#define LOOPIVDEP

namespace ippl
{
namespace Octree
{

    // Container accessing function
    template<typename container_type>
    inline auto& cont_at(container_type& container, typename std::remove_reference_t<container_type>::key_type const& id) noexcept
    {
        return container.at(id);
    }

    // Power function
    constexpr uint64_t pow_ce(uint64_t a, uint8_t e) { return e == 0 ? 1 : a * pow_ce(a, e - 1); }

    namespace
    {
        using std::array;
        using std::bitset;
        using std::span;
        using std::vector;
        using std::unordered_map;
        using std::map;
        using std::queue;
        using std::set;
        using std::multiset;
    }

    // Grid id
    using grid_id_type = uint32_t;

    // Type of the dimension
    using dim_type = uint8_t;

    // Type of depth
    using depth_type = uint8_t;

    // Content id type
    using entity_id_type = size_t;

// Adaptor concepts
    template <class adaptor_type, typename vector_type, typename box_type, typename geometry_type = double>
    concept AdaptorBasicsConcept =
        requires (vector_type & pt, dim_type iDimension) { {adaptor_type::point_comp(pt, iDimension)}->std::convertible_to<geometry_type&>; }
    && requires (vector_type const& pt, dim_type iDimension) { {adaptor_type::point_comp_c(pt, iDimension)}->std::convertible_to<geometry_type>; }
    && requires (box_type& box) { { adaptor_type::box_min(box) }->std::convertible_to<vector_type&>; }
    && requires (box_type& box) { { adaptor_type::box_max(box) }->std::convertible_to<vector_type&>; }
    && requires (box_type const& box) { { adaptor_type::box_min_c(box) }->std::convertible_to<vector_type const&>; }
    && requires (box_type const& box) { { adaptor_type::box_max_c(box) }->std::convertible_to<vector_type const&>; }
    ;

    template <class adaptor_type, typename vector_type, typename box_type, typename geometry_type = double>
    concept AdaptorConcept =
        requires { AdaptorBasicsConcept<adaptor_type, vector_type, box_type, geometry_type>; }
    && requires (box_type const& box, vector_type const& pt) { { adaptor_type::does_box_contain_point(box, pt)}->std::convertible_to<bool>; }
    && requires (box_type const& e1, box_type const& e2, bool e1_must_contain_e2) { { adaptor_type::are_boxes_overlapped(e1, e2, e1_must_contain_e2)}->std::convertible_to<bool>; }
    && requires (span<vector_type const> const& vPoint) { { adaptor_type::box_of_points(vPoint)}->std::convertible_to<box_type>; }
    && requires (span<box_type const> const& vBox) { { adaptor_type::box_of_boxes(vBox)}->std::convertible_to<box_type>; }
    ;

    template <dim_type nDimension, typename vector_type, typename box_type, typename geometry_type = double>
    struct AdaptorGeneralBasics
    {
        static constexpr geometry_type& point_comp(vector_type& pt, dim_type iDimension) noexcept { return pt[iDimension]; }
        static constexpr geometry_type const& point_comp_c(vector_type const& pt, dim_type iDimension) noexcept { return pt[iDimension]; }

        static constexpr vector_type& box_min(box_type& box) noexcept { return box.Min; }
        static constexpr vector_type& box_max(box_type& box) noexcept { return box.Max; }
        static constexpr vector_type const& box_min_c(box_type const& box) noexcept { return box.Min; }
        static constexpr vector_type const& box_max_c(box_type const& box) noexcept { return box.Max; }
    };    

    template <dim_type nDimension, typename vector_type, typename box_type, typename adaptor_basics_type, typename geometry_type = double>
    struct AdaptorGeneralBase : adaptor_basics_type
    {
        using base = adaptor_basics_type;
        static_assert(AdaptorBasicsConcept<base, vector_type, box_type, geometry_type>);

        static constexpr geometry_type size2(vector_type const& pt) noexcept
        {
        auto d2 = geometry_type{ 0 };
        for (dim_type iDim = 0; iDim < nDimension; ++iDim)
        {
            autoc d = base::point_comp_c(pt, iDim);
            d2 += d * d;
        }
        return d2;
        }

        static constexpr geometry_type size(vector_type const& pt) noexcept
        {
        return sqrt(size2(pt));
        }

        static constexpr vector_type add(vector_type const& ptL, vector_type const& ptR) noexcept
        {
        auto pt = vector_type{};
        for (dim_type iDim = 0; iDim < nDimension; ++iDim)
            base::point_comp(pt, iDim) = base::point_comp_c(ptL, iDim) + base::point_comp_c(ptR, iDim);

        return pt;
        }

        static constexpr vector_type subtract(vector_type const& ptL, vector_type const& ptR) noexcept
        {
        auto pt = vector_type{};
        for (dim_type iDim = 0; iDim < nDimension; ++iDim)
            base::point_comp(pt, iDim) = base::point_comp_c(ptL, iDim) - base::point_comp_c(ptR, iDim);

        return pt;
        }

        static constexpr vector_type div(vector_type const& ptL, geometry_type const& rScalarR) noexcept
        {
        auto pt = vector_type{};
        for (dim_type iDim = 0; iDim < nDimension; ++iDim)
            base::point_comp(pt, iDim) = base::point_comp_c(ptL, iDim) / rScalarR;

        return pt;
        }

        static constexpr geometry_type distance(vector_type const& ptL, vector_type const& ptR) noexcept
        {
        return size(subtract(ptL, ptR));
        }

        static constexpr geometry_type distance2(vector_type const& ptL, vector_type const& ptR) noexcept
        {
        return size2(subtract(ptL, ptR));
        }

        static constexpr bool are_points_equal(vector_type const& ptL, vector_type const& ptR, geometry_type rAccuracy) noexcept
        {
        return distance2(ptL, ptR) <= rAccuracy * rAccuracy;
        }

        static constexpr bool does_box_contain_point(box_type const& box, vector_type const& pt, geometry_type tolerance = 0) noexcept
        {
        for (dim_type iDimension = 0; iDimension < nDimension; ++iDimension)
            if (!(base::point_comp_c(base::box_min_c(box), iDimension) - tolerance <= base::point_comp_c(pt, iDimension) && base::point_comp_c(pt, iDimension) <= base::point_comp_c(base::box_max_c(box), iDimension) + tolerance))
            return false;

        return true;
        }

        static constexpr bool does_box_contain_point_strict(box_type const& box, vector_type const& pt) noexcept
        {
        for (dim_type iDimension = 0; iDimension < nDimension; ++iDimension)
            if (!(base::point_comp_c(base::box_min_c(box), iDimension) < base::point_comp_c(pt, iDimension) && base::point_comp_c(pt, iDimension) < base::point_comp_c(base::box_max_c(box), iDimension)))
            return false;

        return true;
        }


        static constexpr bool does_point_touch_box(box_type const& box, vector_type const& pt) noexcept
        {
        for (dim_type iDimension = 0; iDimension < nDimension; ++iDimension)
            if ((base::point_comp_c(base::box_min_c(box), iDimension) == base::point_comp_c(pt, iDimension)))
            return false;

        return true;
        }

        enum EBoxRelation : int8_t { Overlapped = -1, Adjecent = 0, Separated = 1 };
        static constexpr EBoxRelation box_relation(box_type const& e1, box_type const& e2) noexcept
        {
        enum EBoxRelationCandidate : uint8_t { OverlappedC = 0x1, AdjecentC = 0x2, SeparatedC = 0x4 };
        int8_t rel = 0;
        for (dim_type iDimension = 0; iDimension < nDimension; ++iDimension)
        {
            if (base::point_comp_c(base::box_min_c(e1), iDimension) < base::point_comp_c(base::box_max_c(e2), iDimension) && base::point_comp_c(base::box_max_c(e1), iDimension) > base::point_comp_c(base::box_min_c(e2), iDimension))
            rel |= EBoxRelationCandidate::OverlappedC;
            else if (base::point_comp_c(base::box_min_c(e1), iDimension) == base::point_comp_c(base::box_max_c(e2), iDimension) || base::point_comp_c(base::box_max_c(e1), iDimension) == base::point_comp_c(base::box_min_c(e2), iDimension))
            rel |= EBoxRelationCandidate::AdjecentC;
            else if (base::point_comp_c(base::box_min_c(e1), iDimension) > base::point_comp_c(base::box_max_c(e2), iDimension) || base::point_comp_c(base::box_max_c(e1), iDimension) < base::point_comp_c(base::box_min_c(e2), iDimension))
            return EBoxRelation::Separated;
        }
        return (rel & EBoxRelationCandidate::AdjecentC) == EBoxRelationCandidate::AdjecentC ? EBoxRelation::Adjecent : EBoxRelation::Overlapped;
        }

        static constexpr bool are_boxes_overlapped_strict(box_type const& e1, box_type const& e2) noexcept
        {
        return box_relation(e1, e2) == EBoxRelation::Overlapped;
        }

        static constexpr bool are_boxes_overlapped(box_type const& e1, box_type const& e2, bool e1_must_contain_e2 = true, bool fOverlapPtTouchAllowed = false) noexcept
        {
        autoc e1_contains_e2min = does_box_contain_point(e1, base::box_min_c(e2));

        return e1_must_contain_e2
            ? e1_contains_e2min && does_box_contain_point(e1, base::box_max_c(e2))
            : fOverlapPtTouchAllowed
            ? e1_contains_e2min || does_box_contain_point(e1, base::box_max_c(e2)) || does_box_contain_point(e2, base::box_max_c(e1))
            : box_relation(e1, e2) == EBoxRelation::Overlapped
            ;
        }

        static inline box_type box_inverted_init() noexcept
        {
        auto ext = box_type{};
        auto& ptMin = base::box_min(ext);
        auto& ptMax = base::box_max(ext);

        //LOOPIVDEP
        for (dim_type iDimension = 0; iDimension < nDimension; ++iDimension)
        {
            base::point_comp(ptMin, iDimension) = std::numeric_limits<geometry_type>::max();
            base::point_comp(ptMax, iDimension) = std::numeric_limits<geometry_type>::lowest();
        }

        return ext;
        }

        static box_type box_of_points(span<vector_type const> const& vPoint) noexcept
        {
        auto ext = box_inverted_init();
        for (autoc& pt : vPoint)
            for (dim_type iDimension = 0; iDimension < nDimension; ++iDimension)
            {
            if (base::point_comp_c(base::box_min_c(ext), iDimension) > base::point_comp_c(pt, iDimension))
                base::point_comp(base::box_min(ext), iDimension) = base::point_comp_c(pt, iDimension);

            if (base::point_comp_c(base::box_max_c(ext), iDimension) < base::point_comp_c(pt, iDimension))
                base::point_comp(base::box_max(ext), iDimension) = base::point_comp_c(pt, iDimension);
            }

        return ext;
        }

        static box_type box_of_boxes(span<box_type const> const& vExtent) noexcept
        {
        auto ext = box_inverted_init();
        for (autoc& e : vExtent)
            for (dim_type iDimension = 0; iDimension < nDimension; ++iDimension)
            {
            if (base::point_comp_c(base::box_min_c(ext), iDimension) > base::point_comp_c(base::box_min_c(e), iDimension))
                base::point_comp(base::box_min(ext), iDimension) = base::point_comp_c(base::box_min_c(e), iDimension);

            if (base::point_comp_c(base::box_max_c(ext), iDimension) < base::point_comp_c(base::box_max_c(e), iDimension))
                base::point_comp(base::box_max(ext), iDimension) = base::point_comp_c(base::box_max_c(e), iDimension);
            }

        return ext;
        }

        static void move_box(box_type& box, vector_type const& vMove) noexcept
        {
        LOOPIVDEP
        for (dim_type iDimension = 0; iDimension < nDimension; ++iDimension)
        {
            base::point_comp(base::box_min(box), iDimension) += base::point_comp_c(vMove, iDimension);
            base::point_comp(base::box_max(box), iDimension) += base::point_comp_c(vMove, iDimension);
        }
        }

        static constexpr std::optional<double> is_ray_hit(box_type const& box, vector_type const& rayBasePoint, vector_type const& rayHeading, geometry_type tolerance) noexcept
        {
        if (does_box_contain_point(box, rayBasePoint, tolerance))
            return 0.0;

        autoc& ptBoxMin = base::box_min_c(box);
        autoc& ptBoxMax = base::box_max_c(box);

        autoce inf = std::numeric_limits<double>::infinity();

        auto aRMinMax = array<array<double, nDimension>, 2>();
        for (dim_type iDimension = 0; iDimension < nDimension; ++iDimension)
        {
            autoc hComp = base::point_comp_c(rayHeading, iDimension);
            if (hComp == 0)
            {
            if (base::point_comp_c(ptBoxMax, iDimension) + tolerance < base::point_comp_c(rayBasePoint, iDimension))
                return std::nullopt;

            if (base::point_comp_c(ptBoxMin, iDimension) - tolerance > base::point_comp_c(rayBasePoint, iDimension))
                return std::nullopt;

            aRMinMax[0][iDimension] = -inf;
            aRMinMax[1][iDimension] = +inf;
            continue;
            }

            aRMinMax[0][iDimension] = (base::point_comp_c(hComp > 0.0 ? ptBoxMin : ptBoxMax, iDimension) - tolerance - base::point_comp_c(rayBasePoint, iDimension)) / hComp;
            aRMinMax[1][iDimension] = (base::point_comp_c(hComp < 0.0 ? ptBoxMin : ptBoxMax, iDimension) + tolerance - base::point_comp_c(rayBasePoint, iDimension)) / hComp;
        }

        autoc rMin = *std::ranges::max_element(aRMinMax[0]);
        autoc rMax = *std::ranges::min_element(aRMinMax[1]);
        if (rMin > rMax || rMax < 0.0)
            return std::nullopt;

        return rMin < 0 ? rMax : rMin;
        }
    };

    template<dim_type nDimension, typename vector_type, typename box_type, typename geometry_type = double>
    using AdaptorGeneral = AdaptorGeneralBase<nDimension, vector_type, box_type, AdaptorGeneralBasics<nDimension, vector_type, box_type, geometry_type>, geometry_type>;

    template<size_t N> using bitset_arithmetic = bitset<N>;

    template<size_t N>
    bitset_arithmetic<N> operator+ (bitset_arithmetic<N> const& lhs, bitset_arithmetic<N> const& rhs) noexcept
    {
        bool carry = false;
        auto ans = bitset_arithmetic<N>();
        for (size_t i = 0; i < N; ++i)
        {
        autoc sum = (lhs[i] ^ rhs[i]) ^ carry;
        carry = (lhs[i] && rhs[i]) || (lhs[i] && carry) || (rhs[i] && carry);
        ans[i] = sum;
        }

        assert(!carry); // unhandled overflow
        return ans;
    }

    template<size_t N>
    bitset_arithmetic<N> operator+ (bitset_arithmetic<N> const& lhs, size_t rhs) noexcept
    {
        return lhs + bitset_arithmetic<N>(rhs);
    }

    template<size_t N>
    bitset_arithmetic<N> operator- (bitset_arithmetic<N> const& lhs, bitset_arithmetic<N> const& rhs) noexcept
    {
        assert(lhs >= rhs);

        auto ret = lhs;
        bool borrow = false;
        for (size_t i = 0; i < N; ++i)
        {
        if (borrow)
        {
            if (ret[i]) { ret[i] = rhs[i];  borrow = rhs[i]; }
            else { ret[i] = !rhs[i]; borrow = true; }
        }
        else
        {
            if (ret[i]) { ret[i] = !rhs[i]; borrow = false; }
            else { ret[i] = rhs[i];  borrow = rhs[i]; }
        }
        }

        return ret;
    }

    template<size_t N>
    bitset_arithmetic<N> operator- (bitset_arithmetic<N> const& lhs, size_t rhs) noexcept
    {
        return lhs - bitset_arithmetic<N>(rhs);
    }

    template<size_t N>
    bitset_arithmetic<N> operator* (bitset_arithmetic<N> const& lhs, bitset_arithmetic<N> const& rhs) noexcept
    {
        auto ret = bitset_arithmetic<N>{};

        if (lhs.count() < rhs.count())
        {
        for (size_t i = 0; i < N; ++i)
            if (lhs[i])
            ret = ret + (rhs << i);
        }
        else
        {
        for (size_t i = 0; i < N; ++i)
            if (rhs[i])
            ret = ret + (lhs << i);
        }

        return ret;
    }

    template<size_t N>
    bitset_arithmetic<N> operator* (bitset_arithmetic<N> const& lhs, size_t rhs) noexcept
    {
        return lhs * bitset_arithmetic<N>(rhs);
    }
    
    template<size_t N>
    bitset_arithmetic<N> operator* (size_t rhs, bitset_arithmetic<N> const& lhs) noexcept
    {
        return lhs * bitset_arithmetic<N>(rhs);
    }

    template<size_t N>
    static std::tuple<bitset_arithmetic<N>, bitset_arithmetic<N>> gf2_div(bitset_arithmetic<N> const& dividend, bitset_arithmetic<N> divisor) noexcept
    {
        if (divisor.none())
        {
        assert(false);
        return {};
        }

        if (dividend.none())
        return {};

        auto quotent = bitset_arithmetic<N>{ 0 };
        if (dividend == divisor)
        return { bitset_arithmetic<N>(1), quotent };

        if (dividend < divisor)
        return { quotent, dividend };


        size_t sig_dividend = 0;
        for (size_t i = 0, id = N - 1; i < N; ++i, --id)
        if (dividend[id]) { sig_dividend = id; break; }

        size_t sig_divisor = 0;
        for (size_t i = 0, id = N - 1; i < N; ++i, --id)
        if (divisor[id]) { sig_divisor = id; break; }

        size_t nAlignment = (sig_dividend - sig_divisor);
        divisor <<= nAlignment;
        nAlignment += 1;
        auto remainder = dividend;
        while (nAlignment--)
        {
        if (divisor <= remainder)
        {
            quotent[nAlignment] = true;
            remainder = remainder - divisor;
        }
        divisor >>= 1;
        }

        return { quotent, remainder };
    }

    template<size_t N>
    bitset_arithmetic<N> operator / (bitset_arithmetic<N> const& dividend, bitset_arithmetic<N> const& divisor) noexcept
    {
        return std::get<0>(gf2_div(dividend, divisor));
    }

    template<size_t N>
    auto operator<=> (bitset_arithmetic<N> const& lhs, bitset_arithmetic<N> const& rhs) noexcept
    {
        using R = std::strong_ordering;
        for (size_t i = 0, id = N - 1; i < N; ++i, --id)
        if (lhs[id] ^ rhs[id])
            return lhs[id] ? R::greater : R::less;

        return R::equal;
    }

    struct bitset_arithmetic_compare final
    {
        template<size_t N>
        bool operator()(bitset_arithmetic<N> const& lhs, bitset_arithmetic<N> const& rhs) const noexcept { return lhs < rhs; }
    };

}
}
#endif