// Test of the reflecting-BC fill in EBGodunovBC::Set{X,Y,Z}BCs.
//
// Build and run with
//
//   make -j4 && ./main3d.gnu.TEST.ex
//
// (DIM = 2 works too; the z checks are then skipped.) The kernel under test is
// header-only, so this links only AMReX Base -- none of the rest of the
// library is needed.
//
// Idea: build a 1-D cell field whose ghost cells are filled the way AMReX fills
// reflect_even / reflect_odd, form the PLM face states Ip/Im everywhere
// including the ghost faces by direct reconstruction, then hand the interior
// pair to Set*BCs and check that the values it mirrors into the ghost faces
// agree, to round-off, with the directly reconstructed ones.
//
// The 1-D profile is laid out along the direction under test and placed at a
// deliberately non-zero index in the other two directions, so that a dropped
// or mis-scaled Array4 offset shows up as a failure rather than passing by
// accident.

#include <AMReX.H>
#include <AMReX_Print.H>
#include <hydro_ebgodunov_bcs_K.H>
#include <cmath>
#include <vector>

using namespace amrex;

// cells: clo..chi ; faces: clo..chi+1
static constexpr int NG    = 5;   // ghost cells each side (2 spare for the slope stencil)
static constexpr int N     = 8;   // interior cells
static constexpr int domlo = 0;
static constexpr int domhi = N-1;
static constexpr int clo   = domlo - NG;
static constexpr int chi   = domhi + NG;

// index used in the two directions that are not under test
static constexpr int OFF   = 5;

static int idx (int c) { return c - clo; }

// smooth, deliberately asymmetric interior profile
static double interior (int c) { return 1.3 + 0.7*std::sin(0.31*c) + 0.05*c*c; }

struct Field {
    std::vector<double> s, Ip, Im;
    Field () : s(chi-clo+1), Ip(chi-clo+1), Im(chi-clo+1) {}
};

static Field build (bool odd)
{
    Field f;
    for (int c = domlo; c <= domhi; ++c) { f.s[idx(c)] = interior(c); }
    const double sgn = odd ? -1.0 : 1.0;
    for (int g = 1; g <= NG; ++g) {
        f.s[idx(domlo-g)] = sgn * f.s[idx(domlo+g-1)];   // mirror about face domlo
        f.s[idx(domhi+g)] = sgn * f.s[idx(domhi-g+1)];   // mirror about face domhi+1
    }
    for (int c = clo+1; c <= chi-1; ++c) {
        const double slope = 0.5*(f.s[idx(c+1)] - f.s[idx(c-1)]);
        f.Ip[idx(c)] = f.s[idx(c)] + 0.5*slope;   // state at the high face of cell c
        f.Im[idx(c)] = f.s[idx(c)] - 0.5*slope;   // state at the low  face of cell c
    }
    return f;
}

// A one-cell-wide Array4 whose only extent is along dir, sitting at OFF in the
// other two directions. Written out per direction on purpose: forming the Dim3
// by indexing off &b.x is undefined behaviour and compilers do drop the store.
static Array4<double const> mk (std::vector<double> const& v, int dir)
{
    Dim3 b{OFF,OFF,OFF};
    Dim3 e{OFF+1,OFF+1,OFF+1};   // Array4 end is exclusive
    if      (dir == 0) { b.x = clo; e.x = chi+1; }
    else if (dir == 1) { b.y = clo; e.y = chi+1; }
    else               { b.z = clo; e.z = chi+1; }
    return Array4<double const>(const_cast<double*>(v.data()), b, e, 1);
}

int run_checks ()
{
    int bad = 0;
    const double tol = 1e-13;

    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
    for (int odd = 0; odd < 2; ++odd) {

        Field f   = build(odd != 0);
        auto  sa  = mk(f.s , dir);
        auto  Ipa = mk(f.Ip, dir);
        auto  Ima = mk(f.Im, dir);
        const int bc = odd ? BCType::reflect_odd : BCType::reflect_even;

        // Every face the EBGodunov kernels touch, ghost faces included.
        for (int F = domlo-2; F <= domhi+3; ++F)
        {
            double lo = f.Ip[idx(F-1)];
            double hi = f.Im[idx(F  )];
            const double lo_exact = lo;
            const double hi_exact = hi;

            int I = OFF, J = OFF, K = OFF;
            if      (dir == 0) { I = F; }
            else if (dir == 1) { J = F; }
            else               { K = F; }

            if (dir == 0) {
                EBGodunovBC::SetXBCs(I,J,K,0, sa, Ipa, Ima, lo, hi,
                                     0.0, 0.0, bc, bc, domlo, domhi, false);
            } else if (dir == 1) {
                EBGodunovBC::SetYBCs(I,J,K,0, sa, Ipa, Ima, lo, hi,
                                     0.0, 0.0, bc, bc, domlo, domhi, false);
#if (AMREX_SPACEDIM == 3)
            } else {
                EBGodunovBC::SetZBCs(I,J,K,0, sa, Ipa, Ima, lo, hi,
                                     0.0, 0.0, bc, bc, domlo, domhi, false);
#endif
            }

            double lo_want = lo_exact;
            double hi_want = hi_exact;
            if (F == domlo || F == domhi+1) {
                // On the reflecting face itself: odd -> zero, even -> the two
                // sides must agree, and we keep the interior extrapolation.
                if (odd)                 { lo_want = hi_want = 0.0; }
                else if (F == domlo)     { lo_want = hi_exact; }
                else                     { hi_want = lo_exact; }
            }

            if (std::abs(lo-lo_want) > tol || std::abs(hi-hi_want) > tol) {
                Print().SetPrecision(15)
                    << "FAIL dir=" << dir << (odd ? " odd " : " even ")
                    << "F=" << F
                    << "  lo " << lo << " (want " << lo_want << ")"
                    << "  hi " << hi << " (want " << hi_want << ")\n";
                ++bad;
            }
        }
    }}

    if (bad) { Print() << "FAILURES: " << bad << "\n"; }
    else     { Print() << "EBGodunov reflect BCs: all mirrored fills match direct reconstruction\n"; }
    return bad;
}

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    const int bad = run_checks();
    amrex::Finalize();
    return bad ? 1 : 0;
}
