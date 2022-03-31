import numpy as np
import time as timer
import fluxes as fx
import variables as var
import matplotlib.pyplot as plt
import cupy as cp
import copy


nonlinear_ssp_rk_switch = {
    2: [[1 / 2, 1 / 2, 1 / 2]],
    3: [[3 / 4, 1 / 4, 1 / 4],
        [1 / 3, 2 / 3, 2 / 3]]
}


class StepperSingleSpecies:
    def __init__(self, dt, step, resolutions, order, steps, grid, nu):
        self.x_res, self.v_res = resolutions
        self.resolutions = resolutions
        self.order = order
        self.dt = dt
        self.step = step
        self.steps = steps
        # nu = hyperviscosity
        self.mean_flux = fx.MeanFlux(resolution=self.v_res, order=order, charge_mass=-1.0)
        # self.mean_flux.initialize_zero_pad(grid=grid)
        self.fluctuation_flux = fx.FluctuationFlux(resolutions=resolutions, order=order, charge_mass=-1.0, nu=nu)
        self.fluctuation_flux.initialize_zero_pad(grid=grid)
        self.total_distribution = var.Distribution(resolutions=resolutions, order=order, charge_mass=-1.0)

        # RK coefficients
        self.rk_coefficients = np.array(nonlinear_ssp_rk_switch.get(3, "nothing"))

        # tracking arrays
        self.time = 0
        self.next_time = 0
        self.field_energy = np.array([])
        self.time_array = np.array([])
        self.thermal_energy = np.array([])
        self.density_array = np.array([])
        self.moment_array = np.array([])
        self.l2_array = np.array([])
        # self.saved_field = np.array([])
        num = int(self.steps // 20 + 1)

        # semi-implicit matrix
        self.inv_backward_advection = None
        self.build_advection_matrix(grid=grid)

        # save-times
        self.save_times = np.array([0])
        # self.save_times = np.append(np.linspace(140, 170, num=301), 0)

    def main_loop_adams_bashforth(self, mean_distribution, fluctuating_distribution, elliptic, grid):
        """
        Evolve the Vlasov equation in wavenumber space using the Adams-Bashforth time integration scheme
        """
        print('\nBeginning main loop...')

        # Compute first two steps with ssp-rk3 and save fluxes
        # zeroth step
        elliptic.poisson_solve_single_species(distribution=fluctuating_distribution, grid=grid)
        self.mean_flux.semi_discrete_rhs(mean_distribution=mean_distribution,
                                         fluctuating_distribution=fluctuating_distribution,
                                         elliptic=elliptic, grid=grid)
        self.fluctuation_flux.semi_discrete_rhs(mean_distribution=mean_distribution, elliptic=elliptic, grid=grid)
        mean_flux0 = self.mean_flux.output.arr
        fluc_flux0 = self.fluctuation_flux.output.arr

        # first step
        self.ssp_rk3(mean_distribution=mean_distribution, fluctuating_distribution=fluctuating_distribution,
                     elliptic=elliptic, grid=grid)
        self.time += self.dt
        # save fluxes
        elliptic.poisson_solve_single_species(distribution=fluctuating_distribution, grid=grid)
        self.mean_flux.semi_discrete_rhs(mean_distribution=mean_distribution,
                                         fluctuating_distribution=fluctuating_distribution,
                                         elliptic=elliptic, grid=grid)
        self.fluctuation_flux.semi_discrete_rhs(mean_distribution=mean_distribution, elliptic=elliptic, grid=grid)
        mean_flux1 = self.mean_flux.output.arr
        fluc_flux1 = self.fluctuation_flux.output.arr

        # second stage
        self.ssp_rk3(mean_distribution=mean_distribution, fluctuating_distribution=fluctuating_distribution,
                     elliptic=elliptic, grid=grid)
        self.time += self.dt

        # store first two fluxes
        previous_mean_fluxes = [mean_flux1, mean_flux0]
        previous_fluc_fluxes = [fluc_flux1, fluc_flux0]

        # Begin loop
        # save_counter = 0
        for i in range(2, self.steps):
            previous_mean_fluxes, previous_fluc_fluxes = self.adams_bashforth(
                mean_distribution=mean_distribution,
                fluctuating_distribution=fluctuating_distribution,
                elliptic=elliptic, grid=grid,
                prev_mean_fluxes=previous_mean_fluxes,
                prev_fluc_fluxes=previous_fluc_fluxes)

            self.time += self.step

            if i % 50 == 0:
                self.time_array = np.append(self.time_array, self.time)
                elliptic.poisson_solve_single_species(distribution=fluctuating_distribution, grid=grid)
                self.field_energy = np.append(self.field_energy, elliptic.compute_field_energy(grid=grid))
                # fluctuating_distribution.inverse_fourier_transform()
                self.total_distribution.arr = copy.deepcopy(fluctuating_distribution.arr)
                self.total_distribution.arr[0, :, :] = mean_distribution.arr
                self.thermal_energy = np.append(self.thermal_energy,
                                                self.total_distribution.total_thermal_energy(grid=grid))
                self.density_array = np.append(self.density_array,
                                               fluctuating_distribution.total_density(grid=grid)
                                               + mean_distribution.zero_moment(grid=grid))
                # self.moment_array = np.append(self.moment_array, distribution.total_momentum(grid=grid))
                # self.l2_array = np.append(self.l2_array, distribution.l2_norm(grid=grid))
                # Max time-step velocity space
                # elliptic.field.inverse_fourier_transform()
                # max_field = cp.amax(elliptic.field.arr_nodal)
                # max_dt = grid.v.min_dv / max_field / (2 * self.order + 1) / (2 * np.pi) * 0.01
                print('Took 50 steps, time is {:0.3e}'.format(self.time))
                # print('Max velocity-flux dt is {:0.3e}'.format(max_dt))

            # if np.abs(self.time - self.save_times[save_counter]) < 6.0e-3:
            #     print('Reached save time at {:0.3e}'.format(self.time) + ', saving data...')
            #     DataFile.save_data(distribution=distribution.arr_nodal.get(),
            #                        density=distribution.zero_moment.arr_nodal.get(),
            #                        field=elliptic.field.arr_nodal.get(), time=self.time)
            #     save_counter += 1

    def ssp_rk3(self, mean_distribution, fluctuating_distribution, elliptic, grid):
        # Stage set-up
        mean_stage0 = var.Scalar(resolution=self.v_res, order=self.order)
        mean_stage1 = var.Scalar(resolution=self.v_res, order=self.order)
        fluc_stage0 = var.Distribution(resolutions=self.resolutions, order=self.order, charge_mass=None)
        fluc_stage1 = var.Distribution(resolutions=self.resolutions, order=self.order, charge_mass=None)

        # zero stage
        elliptic.poisson_solve_single_species(distribution=fluctuating_distribution, grid=grid)
        self.mean_flux.semi_discrete_rhs(mean_distribution=mean_distribution,
                                         fluctuating_distribution=fluctuating_distribution,
                                         elliptic=elliptic, grid=grid)

        self.fluctuation_flux.semi_discrete_rhs(mean_distribution=mean_distribution, elliptic=elliptic, grid=grid)
        self.fluctuation_flux.output.arr += self.fluctuation_flux.spectral_advection(
            distribution_arr=fluctuating_distribution.arr, grid=grid
        )

        mean_stage0.arr = mean_distribution.arr + self.dt * self.mean_flux.output.arr
        fluc_stage0.arr = fluctuating_distribution.arr + self.dt * self.fluctuation_flux.output.arr

        # first stage
        elliptic.poisson_solve_single_species(distribution=fluc_stage0, grid=grid)
        self.mean_flux.semi_discrete_rhs(mean_distribution=mean_stage0,
                                         fluctuating_distribution=fluc_stage0, elliptic=elliptic, grid=grid)
        self.fluctuation_flux.semi_discrete_rhs(mean_distribution=mean_stage0, elliptic=elliptic, grid=grid)
        self.fluctuation_flux.output.arr += self.fluctuation_flux.spectral_advection(
            distribution_arr=fluc_stage0.arr, grid=grid
        )

        mean_stage1.arr = (
                self.rk_coefficients[0, 0] * mean_distribution.arr +
                self.rk_coefficients[0, 1] * mean_stage0.arr +
                self.rk_coefficients[0, 2] * self.dt * self.mean_flux.output.arr
        )
        fluc_stage1.arr = (
                self.rk_coefficients[0, 0] * fluctuating_distribution.arr +
                self.rk_coefficients[0, 1] * fluc_stage0.arr +
                self.rk_coefficients[0, 2] * self.dt * self.fluctuation_flux.output.arr
        )

        # second stage
        elliptic.poisson_solve_single_species(distribution=fluc_stage1, grid=grid)
        self.mean_flux.semi_discrete_rhs(mean_distribution=mean_stage1,
                                         fluctuating_distribution=fluc_stage1, elliptic=elliptic, grid=grid)
        self.fluctuation_flux.semi_discrete_rhs(mean_distribution=mean_stage1, elliptic=elliptic, grid=grid)
        self.fluctuation_flux.output.arr += self.fluctuation_flux.spectral_advection(
            distribution_arr=fluc_stage1.arr, grid=grid
        )

        mean_distribution.arr = (
                self.rk_coefficients[1, 0] * mean_distribution.arr +
                self.rk_coefficients[1, 1] * mean_stage1.arr +
                self.rk_coefficients[1, 2] * self.dt * self.mean_flux.output.arr
        )
        fluctuating_distribution.arr = (
                self.rk_coefficients[1, 0] * fluctuating_distribution.arr +
                self.rk_coefficients[1, 1] * fluc_stage1.arr +
                self.rk_coefficients[1, 2] * self.dt * self.fluctuation_flux.output.arr
        )

    def adams_bashforth(self, mean_distribution, fluctuating_distribution, elliptic, grid,
                        prev_mean_fluxes, prev_fluc_fluxes):
        # Compute Poisson constraint
        elliptic.poisson_solve_single_species(distribution=fluctuating_distribution, grid=grid)

        # Compute velocity flux
        self.mean_flux.semi_discrete_rhs(mean_distribution=mean_distribution,
                                         fluctuating_distribution=fluctuating_distribution,
                                         elliptic=elliptic, grid=grid)
        self.fluctuation_flux.semi_discrete_rhs(mean_distribution=mean_distribution, elliptic=elliptic, grid=grid)

        # Update distribution according to explicit treatment of velocity flux and crank-nicholson for advection
        mean_distribution.arr += self.dt * ((23 / 12 * self.mean_flux.output.arr -
                                             4 / 3 * prev_mean_fluxes[0] +
                                             5 / 12 * prev_mean_fluxes[1]))
        fluctuating_distribution.arr += self.dt * ((23 / 12 * self.fluctuation_flux.output.arr -
                                                    4 / 3 * prev_fluc_fluxes[0] +
                                                    5 / 12 * prev_fluc_fluxes[1]) +
                                                   0.5 * self.fluctuation_flux.spectral_advection(
                    distribution_arr=fluctuating_distribution.arr,
                    grid=grid))
        # Do inverse half backward advection step
        fluctuating_distribution.arr = cp.einsum('nmjk,nmk->nmj', self.inv_backward_advection,
                                                 fluctuating_distribution.arr)
        mean_fluxes = [self.mean_flux.output.arr, prev_mean_fluxes[0]]
        fluc_fluxes = [self.fluctuation_flux.output.arr, prev_fluc_fluxes[0]]
        return [mean_fluxes, fluc_fluxes]

    def build_advection_matrix(self, grid):
        """ Construct the global backward advection matrix """
        backward_advection_operator = (cp.eye(grid.v.order)[None, None, :, :] -
                                       0.5 * self.dt * -1j * grid.x.device_wavenumbers[:, None, None, None] *
                                       grid.v.translation_matrix[None, :, :, :])
        self.inv_backward_advection = cp.linalg.inv(backward_advection_operator)
