MODULE radiationinterpolator

    use iso_c_binding
    use iso_fortran_env

    implicit none
    REAL(c_double), PARAMETER :: sol = 299792458.0                  ! m/s
    REAL(c_double), PARAMETER :: e = 1.602176634e-19                ! C
    REAL(c_double), PARAMETER :: me = 9.1093837015e-31              ! kg
    REAL(c_double), PARAMETER :: epsilon_0 = 8.8541878128e-12       ! F/m
    REAL(c_double), PARAMETER :: pi = 3.14159265358979
    REAL(c_double), PARAMETER :: field_coeff = e/(4*pi*epsilon_0*sol)

    CONTAINS 
            
    SUBROUTINE cross(x,y, output) bind(C)
        ! Compute the cross product of two vectors, x cross y
        
          implicit none
          REAL(c_double), DIMENSION(3), INTENT(IN) :: x, y
          REAL(c_double), DIMENSION(3), INTENT(INOUT) :: output
        
          output(1) = x(2)*y(3) - x(3)*y(2)
          output(2) = x(3)*y(1) - x(1)*y(3)
          output(3) = x(1)*y(2) - x(2)*y(1)
        
    END SUBROUTINE cross
            
    SUBROUTINE field(r_part, r_det, beta, beta_dot, E_field) bind(C)
    ! Compute the electric field at r_det due to a particle at r_part.
    
        implicit none
        REAL(c_double), DIMENSION(3), INTENT(IN) :: r_part, r_det
        REAL(c_double), DIMENSION(3), INTENT(IN) :: beta, beta_dot
        REAL(c_double), DIMENSION(3) :: R_vec, n_hat, numerator, n_cross_bd
        REAL(c_double) :: R_mag, denominator
        REAL(c_double), DIMENSION(3), INTENT(OUT) :: E_field
        
        ! Separation vector between particle and detector
        R_vec = r_det - r_part
        R_mag = SQRT(R_vec(1)**2 + R_vec(2)**2 + R_vec(3)**2)
        n_hat = R_vec/R_mag
        
        call cross(n_hat - beta, beta_dot, n_cross_bd)
        call cross(n_hat, n_cross_bd, numerator)
        denominator = R_mag*(1 - dot_product(n_hat, beta))**3
        E_field = field_coeff*numerator/denominator  
      
    END SUBROUTINE field
    
    SUBROUTINE interpolator(t, t_prev, t_det_array, field, nt_det, &
        output) bind(C)
    ! Interpolate field onto t_det_array 
        
        implicit none
        INTEGER(c_int), INTENT(IN), VALUE :: nt_det
        REAL(c_double), INTENT(IN) :: t, t_prev
        REAL(c_double), DIMENSION(nt_det), INTENT(IN) :: t_det_array
        REAL(c_double), DIMENSION(3), INTENT(IN) :: field
        REAL(c_double) :: t_det_0, dt_det
        INTEGER(c_int) :: n_slot, n_slot_prev, n_iter
        REAL(c_double) :: scale_fac, t_temp
        REAL(c_double), DIMENSION(nt_det, 3), INTENT(INOUT):: output
            
        t_det_0 = t_det_array(1)
        dt_det = t_det_array(2) - t_det_array(1)
        
        ! Define the temporary variables
        n_slot = min(FLOOR((t - t_det_0)/dt_det), nt_det - 1)
        n_slot_prev = FLOOR((t_prev - t_det_0)/dt_det)
        n_iter = n_slot_prev
        t_temp = t_prev
        
        do while (n_iter < n_slot)
            scale_fac = (t_det_array(n_iter + 2) - t_temp)/dt_det
            output(n_iter + 1, :) = output(n_iter + 1, :) + scale_fac*field
            n_iter = n_iter + 1
            t_temp = t_det_array(n_iter + 1)
        end do
        
        ! Use a different scale to prevent double counting
        scale_fac = (t - t_det_array(n_slot + 1))/dt_det
        output(n_slot + 1, :) = output(n_slot + 1, :) + scale_fac*field

    END SUBROUTINE interpolator

    SUBROUTINE field_over_time(pos_over_time, beta_over_time, &
                                    sim_times, nt_sim, &
                                    det_times, nt_det, &
                                    det_pos, all_radiation) bind(C)
        ! Calculate the radiated field due to the entire trajectory of a
        ! particle. 
        implicit none
        INTEGER(c_int), INTENT(IN), VALUE:: nt_sim, nt_det
        REAL(c_double), DIMENSION(nt_sim), INTENT(IN) :: sim_times
        REAL(c_double), DIMENSION(nt_sim, 3), INTENT(IN) :: pos_over_time
        REAL(c_double), DIMENSION(nt_sim, 3), INTENT(IN) :: beta_over_time
        REAL(c_double), DIMENSION(nt_det), INTENT(IN) :: det_times
        REAL(c_double), DIMENSION(3), INTENT(IN) :: det_pos
        REAL(c_double), DIMENSION(nt_det, 3), INTENT(INOUT) :: all_radiation

        INTEGER(c_int) :: i
        REAL(c_double) :: t, dt_sim, t_prev, R, R_prev
        REAL(c_double) :: t_det, t_det_prev, t_det_max, t_det_min
        REAL(c_double), DIMENSION(3) :: r_part, r_part_prev, beta
        REAL(c_double), DIMENSION(3) :: beta_dot, R_vec, R_vec_prev, field_at_t

        t_det_min = det_times(1)
        t_det_max = det_times(nt_det)

        DO i = 2, nt_sim
            t = sim_times(i)
            t_prev = sim_times(i - 1)
            dt_sim = t - t_prev

            ! Get particle properties at current time
            r_part = pos_over_time(i, :)
            r_part_prev = pos_over_time(i - 1, :)
            beta = beta_over_time(i, :)
            beta_dot = (beta - beta_over_time(i - 1, :))/dt_sim

            ! Distance to detector
            R_vec = det_pos - r_part
            R_vec_prev = det_pos - r_part_prev
            R = SQRT(R_vec(1)**2 + R_vec(2)**2 + R_vec(3)**2)
            R_prev = SQRT(R_vec_prev(1)**2 + R_vec_prev(2)**2 + & 
                                R_vec_prev(3)**2)

            ! Time to detector
            t_det = R/sol + t
            t_det_prev = R_prev/sol + t_prev 

            call field(r_part, det_pos, beta, beta_dot, field_at_t)

            IF (t_det < t_det_max .AND. t_det_prev > t_det_min) THEN
                call interpolator(t_det, t_det_prev, det_times, field_at_t, & 
                                    nt_det, all_radiation)
            END IF
        END DO
    END SUBROUTINE field_over_time
    
END MODULE radiationinterpolator
    
    
    