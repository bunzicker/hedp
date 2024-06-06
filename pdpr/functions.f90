MODULE PDPR

    use iso_fortran_env
    use iso_c_binding

    implicit none
    INTEGER(c_int), PARAMETER :: cd = c_double, ci = c_int
    INTEGER(c_int), PARAMETER :: c_cplx = c_double_complex 
    REAL(c_double), PARAMETER :: pi = acos(-1.0_cd)

    contains

    SUBROUTINE meshgrid(x, y, nx, ny, xx, yy) BIND(C)

        ! Create a 2D grid from two one dimensional arrays. 
        INTEGER(c_int), INTENT(in), VALUE :: nx, ny
        REAL(cd), DIMENSION(nx), INTENT(in) :: x
        REAL(cd), DIMENSION(ny), INTENt(in) :: y
        REAL(cd), DIMENSION(nx, ny), INTENT(out) :: xx, yy
        
        yy = spread(x, 1, size(y))
        xx = spread(y, 2, size(x))

    END SUBROUTINE meshgrid


    SUBROUTINE trapz_2d(f, x, y, nx, ny, res) BIND(C)

        ! Numerically integrate a 2D, complex valued array over x and y. 

        ! Parameters
        ! f: The array to be integrated.
        ! x, y: 1D arrays defining the coordinates of values in f.
        ! nx, ny: The length of arrays x and y.
        ! res: The complex valued result.

        implicit none
        INTEGER(ci), INTENT(in), VALUE :: nx, ny
        REAL(cd), DIMENSION(nx), INTENT(in) :: x
        REAL(cd), DIMENSION(ny), INTENT(in) :: y
        COMPLEX(c_cplx), DIMENSION(nx, ny), INTENT(in) :: f
        COMPLEX(c_cplx), INTENT(out) :: res

        REAL(cd) :: dx, dy

        dx = x(2) - x(1)        ! dx must be constant over the whole array
        dy = y(2) - y(1)        ! dy must be constant over the whole array

        res = 0.25_cd*dx*dy*SUM(f(1:nx-1, 1:ny - 1) + f(1:nx-1, 2:ny) + &
                f(2:nx-1, 1:ny-1) + f(2:nx, 2:ny))

    END SUBROUTINE trapz_2d

    SUBROUTINE propagator(U1, x1, n_x1, y1, n_y1, z1, &
                                x2, n_x2, y2, n_y2, z2, U2, k &
                         ) BIND(C)

    ! Propagator defined in Eq. 8.4 in Czapla's thesis.
    implicit none
    INTEGER(ci), INTENT(in), VALUE :: n_x1, n_y1, n_x2, n_y2
    REAL(cd), INTENT(in), VALUE :: z1, z2, k
    REAL(cd), DIMENSION(n_x1), INTENT(in) :: x1
    REAL(cd), DIMENSION(n_y1), INTENT(in) :: y1
    COMPLEX(c_cplx), DIMENSION(n_x1, n_y1), INTENT(in) :: U1
    REAL(cd), DIMENSION(n_x2), INTENT(in) :: x2
    REAL(cd), DIMENSION(n_y2), INTENT(in) :: y2
    COMPLEX(c_cplx), DIMENSION(n_x2, n_y2), INTENT(OUT) :: U2

    INTEGER(ci) :: ix2, iy2
    REAL(cd) :: dz, x2i, y2i
    REAL(cd), DIMENSION(n_x1, n_y1) :: xx1, yy1, dx_sq, dy_sq
    REAL(cd), DIMENSION(n_x1, n_y1) :: separation
    COMPLEX(c_cplx), DIMENSION(n_x1, n_y1) :: term1, term2, term3, integrand
    COMPLEX(c_cplx) :: ik, int_res
    
    ! Useful constants/arrays
    ik = CMPLX(0.0, k, kind=c_cplx)
    dz = z2 - z1
    CALL meshgrid(x1, y1, n_x1, n_y1, xx1, yy1)

    ! Loop through all pixels in x2, y2
    DO iy2 = 1, n_y2

        y2i = y2(iy2)
        dy_sq = (y2i - yy1)**2

        DO ix2 = 1, n_x2

            x2i = x2(ix2)
            dx_sq = (x2i - xx1)**2

            separation(:, :) = SQRT(dx_sq(:, :) + dy_sq(:, :) + dz**2)
            term1 = U1/separation**2
            term2 = (1/separation) - ik
            term3 = EXP(ik*separation)

            integrand = term1*term2*term3
            CALL trapz_2d(integrand, x1, y1, n_x1, n_y1, int_res) ! Integrate over x1, y1
            U2(ix2, iy2) = int_res*dz/(2.0_cd*pi)
        END DO
    END DO
END SUBROUTINE propagator


END MODULE PDPR