/**
 * \file hydro_ebgodunov_K.H
 *
 * \addtogroup EBGodunov
 *  @{
 */

#ifndef HYDRO_EBGODUNOV_K_H
#define HYDRO_EBGODUNOV_K_H

#include <AMReX_Gpu.H>
#include <AMReX_BCRec.H>
#include <AMReX_BC_TYPES.H>
#include <AMReX_Array.H>
#include <iomanip>
#include <hydro_constants.H>

/* This header file contains the inlined __host__ __device__ functions required
   for setting BCs for computing the advective velocity and transverse terms in
   3D EBGodunov ExtrapVel.
*/

/*
   Conventions used below.

   The functions take the pair of face-based states (lo,hi) at face (i,j,k) in
   the direction of interest, where "lo" is extrapolated from the cell on the
   low side of the face and "hi" from the cell on the high side.  They are
   passed as references and modified in place.

   In addition they take the two face-based arrays holding those states for the
   whole box, so that reflecting boundary conditions can be filled beyond the
   domain face by mirroring interior values.  Using the x-direction as the
   example, with the PLM arrays Ipx/Imx produced by (EB)PLM::PredictVelOnXFace,
   the state pair at face I is

       lo = Ip(I-1,j,k,n)      hi = Im(I  ,j,k,n)

   so Ip and Im are exactly the arrays the caller already reads to form lo,hi.

   Reflection about the low domain face (face index domlo) maps face I to face
   2*domlo-I and exchanges the two sides of the face, hence

       lo(I) =  hi(2*domlo-I) =  Im(2*domlo-I  ,j,k,n)
       hi(I) =  lo(2*domlo-I) =  Ip(2*domlo-I-1,j,k,n)

   (with an overall sign flip for reflect_odd), and similarly about the high
   domain face (face index domhi+1) with the mirror index 2*(domhi+1)-I.  On the
   reflecting face itself the mirror index is the face itself, so symmetry
   simply requires lo == hi; there we keep the state extrapolated from the
   interior cell (lo = hi at domlo, hi = lo at domhi+1), which is the behaviour
   used everywhere else in AMReX-Hydro.

   IMPORTANT: Ip and Im must still hold the extrapolated (PLM) states when these
   functions are called.  In EBGodunov::ExtrapVelToFacesOnBox the upwinded
   intermediate edge state used to be written back into Imx/Imy/Imz from within
   the same kernel launch, which made the mirrored reads above racy; it is now
   written to its own xedge/yedge/zedge arrays.  Do not reintroduce that reuse.
*/

namespace EBGodunovBC {

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void SetXBCs (const int i, const int j, const int k, const int n,
              const amrex::Array4<const amrex::Real> &s,
              const amrex::Array4<const amrex::Real> &Ip,
              const amrex::Array4<const amrex::Real> &Im,
              amrex::Real &lo, amrex::Real &hi,
              amrex::Real velm, amrex::Real velp,
              const int bclo, const int bchi,
              const int domlo, const int domhi,
              const bool is_velocity )
{
    using namespace amrex;


    // Low X
    if (i <= domlo)
    {
        if ( (bclo == BCType::direction_dependent) && is_velocity &&
                (n == XVEL) && (s(domlo-1,j,k,n) >= 0.0) )
        {
            lo = s(domlo-1, j, k, n);
            hi = s(domlo-1, j, k, n);
        }
        else if ( bclo == BCType::ext_dir ||
                 (bclo == BCType::direction_dependent && velp >= 0.0) )
        {
            lo = s(domlo-1,j,k,n);
            // For turbulent inflow, there are times when the inflow face
            // may have a predicted outflowing velocity. Here, we preserve
            // the normal component of the Dirichlet BC, but allow the
            // tangential components to transport values from the interior.
            if( n == XVEL && is_velocity ) hi=lo;
        }
        else if ( bclo == BCType::foextrap || bclo == BCType::hoextrap ||
                 (bclo == BCType::direction_dependent && velp < 0.0) )
        {
            lo = hi;
        }
        else if (bclo == BCType::reflect_even)
        {
            if ( i == domlo )
            {
                // Mirror of the face is the face itself; symmetry gives lo==hi.
                lo = hi;
            }
            else
            {
                AMREX_ASSERT(Ip && Im);
                lo =  Im(2*domlo-i  ,j,k,n);
                hi =  Ip(2*domlo-i-1,j,k,n);
            }
        }
        else if (bclo == BCType::reflect_odd)
        {
            if ( i == domlo )
            {
                hi = Real(0.);
                lo = Real(0.);
            }
            else
            {
                AMREX_ASSERT(Ip && Im);
                lo = -Im(2*domlo-i  ,j,k,n);
                hi = -Ip(2*domlo-i-1,j,k,n);
            }
        }
    }
    // High X
    else if (i > domhi)
    {
        if ( (bchi == BCType::direction_dependent) && is_velocity &&
                (n == XVEL) && (s(domhi+1,j,k,n) <= 0.0) )
        {
            hi = s(domhi+1, j, k, n);
            lo = s(domhi+1, j, k, n);
        }
        else if ( (bchi == BCType::ext_dir) ||
             (bchi == BCType::direction_dependent && velm <= 0.0) )
        {
            hi = s(domhi+1,j,k,n) ;
            if( n ==XVEL && is_velocity ) lo=hi;
        }
        else if ( bchi == BCType::foextrap || bchi == BCType::hoextrap ||
                 (bchi== BCType::direction_dependent && velm > 0.0) )
        {
            hi = lo;
        }
        else if (bchi == BCType::reflect_even)
        {
            if ( i == domhi+1 )
            {
                hi = lo;
            }
            else
            {
                AMREX_ASSERT(Ip && Im);
                hi =  Ip(2*(domhi+1)-i-1,j,k,n);
                lo =  Im(2*(domhi+1)-i  ,j,k,n);
            }
        }
        else if (bchi == BCType::reflect_odd)
        {
            if ( i == domhi+1 )
            {
                hi = Real(0.);
                lo = Real(0.);
            }
            else
            {
                AMREX_ASSERT(Ip && Im);
                hi = -Ip(2*(domhi+1)-i-1,j,k,n);
                lo = -Im(2*(domhi+1)-i  ,j,k,n);
            }
        }
    }
    else
        return;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void SetYBCs (const int i, const int j, const int k, const int n,
              const amrex::Array4<const amrex::Real> &s,
              const amrex::Array4<const amrex::Real> &Ip,
              const amrex::Array4<const amrex::Real> &Im,
              amrex::Real &lo, amrex::Real &hi,
              amrex::Real velm, amrex::Real velp,
              const int bclo, const int bchi,
              const int domlo, const int domhi,
              const bool is_velocity )
{
    using namespace amrex;


    // Low Y
    if (j <= domlo)
    {
        if ( (bclo == BCType::direction_dependent) && is_velocity &&
                (n == YVEL) && (s(i,domlo-1,k,n) >= 0.0) )
        {
            lo = s(i, domlo-1, k, n);
            hi = s(i, domlo-1, k, n);
        }
        else if ( bclo == BCType::ext_dir ||
                 (bclo == BCType::direction_dependent && velp >= 0.0) )
        {
            lo = s(i,domlo-1,k,n);
            if ( n == YVEL && is_velocity ) hi = lo;
        }
        else if ( bclo == BCType::foextrap || bclo == BCType::hoextrap ||
                 (bclo == BCType::direction_dependent && velp < 0.0) )
        {
            lo = hi;
        }
        else if (bclo == BCType::reflect_even)
        {
            if ( j == domlo )
            {
                lo = hi;
            }
            else
            {
                AMREX_ASSERT(Ip && Im);
                lo =  Im(i,2*domlo-j  ,k,n);
                hi =  Ip(i,2*domlo-j-1,k,n);
            }
        }
        else if(bclo == BCType::reflect_odd)
        {
            if ( j == domlo )
            {
                hi = Real(0.);
                lo = Real(0.);
            }
            else
            {
                AMREX_ASSERT(Ip && Im);
                lo = -Im(i,2*domlo-j  ,k,n);
                hi = -Ip(i,2*domlo-j-1,k,n);
            }
        }
    }
    // High Y
    else if (j > domhi)
    {
        if ( (bchi == BCType::direction_dependent) && is_velocity &&
                (n == YVEL) && (s(i,domhi+1,k,n) <= 0.0) )
        {
            hi = s(i, domhi+1, k, n);
            lo = s(i, domhi+1, k, n);
        }
        else if ( (bchi == BCType::ext_dir) ||
             (bchi == BCType::direction_dependent && velm <= 0.0) )
        {
            hi = s(i,domhi+1,k,n);
            if( n == YVEL && is_velocity ) lo = hi ;
        }
        else if ( bchi == BCType::foextrap || bchi == BCType::hoextrap ||
                 (bchi== BCType::direction_dependent && velm > 0.0) )
        {
            hi = lo;
        }
        else if (bchi == BCType::reflect_even)
        {
            if ( j == domhi+1 )
            {
                hi = lo;
            }
            else
            {
                AMREX_ASSERT(Ip && Im);
                hi =  Ip(i,2*(domhi+1)-j-1,k,n);
                lo =  Im(i,2*(domhi+1)-j  ,k,n);
            }
        }
        else if (bchi == BCType::reflect_odd)
        {
            if ( j == domhi+1 )
            {
                hi = Real(0.);
                lo = Real(0.);
            }
            else
            {
                AMREX_ASSERT(Ip && Im);
                hi = -Ip(i,2*(domhi+1)-j-1,k,n);
                lo = -Im(i,2*(domhi+1)-j  ,k,n);
            }
        }
    }
    else
        return;
}

#if (AMREX_SPACEDIM==3)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void SetZBCs (const int i, const int j, const int k, const int n,
              const amrex::Array4<const amrex::Real> &s,
              const amrex::Array4<const amrex::Real> &Ip,
              const amrex::Array4<const amrex::Real> &Im,
              amrex::Real &lo, amrex::Real &hi,
              amrex::Real velm, amrex::Real velp,
              const int bclo, const int bchi,
              const int domlo, const int domhi,
              const bool is_velocity)
{
    using namespace amrex;


    // Low Z
    if (k <= domlo)
    {
        if ( (bclo == BCType::direction_dependent) && is_velocity &&
                (n == ZVEL) && (s(i,j,domlo-1,n) >= 0.0) )
        {
            lo = s(i, j, domlo-1, n);
            hi = s(i, j, domlo-1, n);
        }
        else if ( bclo == BCType::ext_dir ||
                 (bclo == BCType::direction_dependent && velp >= 0.0) )
        {
            lo =s(i,j,domlo-1,n);
            if ( n == ZVEL && is_velocity ) hi = lo;
        }
        else if ( bclo == BCType::foextrap || bclo == BCType::hoextrap ||
                 (bclo == BCType::direction_dependent && velp < 0.0) )
        {
            lo = hi;
        }
        else if (bclo == BCType::reflect_even)
        {
            if ( k == domlo )
            {
                lo = hi;
            }
            else
            {
                AMREX_ASSERT(Ip && Im);
                lo =  Im(i,j,2*domlo-k  ,n);
                hi =  Ip(i,j,2*domlo-k-1,n);
            }
        }
        else if(bclo == BCType::reflect_odd)
        {
            if ( k == domlo )
            {
                hi = Real(0.);
                lo = Real(0.);
            }
            else
            {
                AMREX_ASSERT(Ip && Im);
                lo = -Im(i,j,2*domlo-k  ,n);
                hi = -Ip(i,j,2*domlo-k-1,n);
            }
        }
    }
    // High Z
    else if (k > domhi)
    {
        if ( (bchi == BCType::direction_dependent) && is_velocity &&
                (n == ZVEL) && (s(i,j,domhi+1,n) <= 0.0) )
        {
            hi = s(i, j, domhi+1, n);
            lo = s(i, j, domhi+1, n);
        }
        else if ( (bchi == BCType::ext_dir) ||
             (bchi == BCType::direction_dependent && velm <= 0.0) )
        {
            hi = s(i,j,domhi+1,n);
            if ( n == ZVEL && is_velocity ) lo = hi ;
        }
        else if ( bchi == BCType::foextrap || bchi == BCType::hoextrap ||
                 (bchi== BCType::direction_dependent && velm > 0.0) )
        {
            hi = lo;
        }
        else if (bchi == BCType::reflect_even)
        {
            if ( k == domhi+1 )
            {
                hi = lo;
            }
            else
            {
                AMREX_ASSERT(Ip && Im);
                hi =  Ip(i,j,2*(domhi+1)-k-1,n);
                lo =  Im(i,j,2*(domhi+1)-k  ,n);
            }
        }
        else if (bchi == BCType::reflect_odd)
        {
            if ( k == domhi+1 )
            {
                hi = Real(0.);
                lo = Real(0.);
            }
            else
            {
                AMREX_ASSERT(Ip && Im);
                hi = -Ip(i,j,2*(domhi+1)-k-1,n);
                lo = -Im(i,j,2*(domhi+1)-k  ,n);
            }
        }
    }
    else
        return;
}
#endif
}
#endif
/** @} */
