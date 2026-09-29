! pyudales algebraic Vreman discrepancy; GPL-3.0-or-later, matching uDALES.
! No MPI or solver state dependencies: independently testable in solver precision.
module modsgsdiscrepancy
  use, intrinsic :: ieee_arithmetic, only : ieee_is_finite
  implicit none
  private
  public :: lsgs_discrepancy, sgs_bias_b0, sgs_bias_b1, sgs_bias_b2
  public :: sgs_discrepancy_height, sgs_discrepancy_za_over_h, sgs_discrepancy_zb_over_h
  public :: sgs_discrepancy_epsilon, sgs_discrepancy_cap
  public :: discrepancy_valid, discrepancy_features, discrepancy_multiplier
  logical :: lsgs_discrepancy = .false.
  real :: sgs_bias_b0 = 0., sgs_bias_b1 = 0., sgs_bias_b2 = 0.
  real :: sgs_discrepancy_height = 0., sgs_discrepancy_za_over_h = 0., sgs_discrepancy_zb_over_h = 0.
  real :: sgs_discrepancy_epsilon = 0., sgs_discrepancy_cap = 0.
contains
  logical function discrepancy_valid()
    real :: values(8), limit, za, zb
    values = [sgs_bias_b0, sgs_bias_b1, sgs_bias_b2, sgs_discrepancy_height, &
      sgs_discrepancy_za_over_h, sgs_discrepancy_zb_over_h, sgs_discrepancy_epsilon, sgs_discrepancy_cap]
    discrepancy_valid = .false.
    if (.not. all(ieee_is_finite(values))) return
    if (sgs_discrepancy_height <= 0. .or. sgs_discrepancy_epsilon <= 0.) return
    if (sgs_discrepancy_zb_over_h <= sgs_discrepancy_za_over_h) return
    ! A margin avoids overflow after rounding exp(L); also keep exp(-L) normal.
    limit = min(log(huge(1.)), -log(tiny(1.))) - 1.
    if (sgs_discrepancy_cap <= 0. .or. sgs_discrepancy_cap > limit) return
    ! Check the dimensional endpoints without overflowing in a trapped build.
    if (sgs_discrepancy_height > 1.) then
      if (max(abs(sgs_discrepancy_za_over_h),abs(sgs_discrepancy_zb_over_h)) > &
          huge(1.)/sgs_discrepancy_height/4.) return
    end if
    if (max(abs(sgs_discrepancy_za_over_h),abs(sgs_discrepancy_zb_over_h)) > huge(1.)/4.) return
    za = sgs_discrepancy_height*sgs_discrepancy_za_over_h
    zb = sgs_discrepancy_height*sgs_discrepancy_zb_over_h
    if (zb <= za) return
    discrepancy_valid = .true.
  end function discrepancy_valid

  pure subroutine discrepancy_features(a, z, q, phi)
    real, intent(in) :: a(3,3), z
    real, intent(out) :: q, phi
    real :: scale, an(3,3), snorm, onorm, epsilon_scaled, za, zb
    ! Scale before squaring, preserving the physical epsilon in inverse seconds.
    scale = max(maxval(abs(a)), sgs_discrepancy_epsilon)
    an = a/scale
    epsilon_scaled = sgs_discrepancy_epsilon/scale
    snorm = sum((0.5*(an + transpose(an)))**2)
    onorm = sum((0.5*(an - transpose(an)))**2)
    q = (onorm - snorm)/(onorm + snorm + epsilon_scaled**2)
    za = sgs_discrepancy_height*sgs_discrepancy_za_over_h
    zb = sgs_discrepancy_height*sgs_discrepancy_zb_over_h
    phi = 0.
    if (z > za .and. z < zb) phi = sin(acos(-1.)*((z-za)/(zb-za)))**2
  end subroutine discrepancy_features

  pure subroutine discrepancy_multiplier(a, z, multiplier, saturation)
    real, intent(in) :: a(3,3), z
    real, intent(out) :: multiplier, saturation
    real :: q, phi, bscale, normalized_g, ratio, t, divisor, numerator, denominator
    call discrepancy_features(a, z, q, phi)
    bscale = max(abs(sgs_bias_b0),abs(sgs_bias_b1),abs(sgs_bias_b2))
    ratio = 0.
    if (bscale > 0.) then
      normalized_g = sgs_bias_b0/bscale + (sgs_bias_b1/bscale)*phi + (sgs_bias_b2/bscale)*q
      if (normalized_g /= 0.) then
        ! At |g/L| >= 20 tanh is unity in the solver's double precision.
        ! Avoid overflow for arbitrarily large finite coefficients.
        divisor = max(bscale,sgs_discrepancy_cap)
        numerator = normalized_g*(bscale/divisor)
        denominator = sgs_discrepancy_cap/divisor
        if (abs(numerator) >= 20.*denominator) then
          ratio = sign(20.,normalized_g)
        else
          ratio = numerator/denominator
        end if
      end if
    end if
    t = tanh(ratio)
    multiplier = exp(sgs_discrepancy_cap*t)
    saturation = 0.
    if (abs(t) >= 0.95) saturation = 1.
  end subroutine discrepancy_multiplier
end module modsgsdiscrepancy
