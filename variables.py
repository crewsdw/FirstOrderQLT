import numpy as np
import cupy as cp
import tools.dispersion as dispersion
import scipy.optimize as opt
import cupyx.scipy.signal as sig
import scipy.signal as ssig
# import matplotlib.pyplot as plt
# import dielectric
import numpy.polynomial as poly
import scipy.special as sp
# from copy import deepcopy

cp.random.seed(1111)


class SpaceScalar:
    def __init__(self, resolution):
        self.res = resolution
        self.arr_nodal, self.arr_spectral = None, None

    def fourier_transform(self):
        # self.arr_spectral = cp.fft.fftshift(cp.fft.fft(self.arr_nodal, norm='forward'))
        self.arr_spectral = cp.fft.rfft(self.arr_nodal, norm='forward')

    def inverse_fourier_transform(self):
        # self.arr_nodal = cp.real(cp.fft.ifft(cp.fft.fftshift(self.arr_spectral), norm='forward'))
        self.arr_nodal = cp.fft.irfft(self.arr_spectral, norm='forward')

    def integrate(self, grid):
        arr_add = cp.append(self.arr_nodal, self.arr_nodal[0])
        # x_add = cp.append(grid.x.device_arr, grid.x.device_arr[-1] + grid.x.dx)
        return trapz(arr_add, grid.x.dx)

    def integrate_energy(self, grid):
        arr = 0.5 * self.arr_nodal ** 2.0
        arr_add = cp.append(arr, arr[0])
        # x_add = cp.append(grid.x.device_arr, grid.x.device_arr[-1] + grid.x.dx)
        return trapz(arr_add, grid.x.dx)

    def compute_wigner_distribution(self, grid):
        spectrum = cp.fft.fftshift(cp.fft.fft(self.arr_nodal))
        full_wavenumbers = 2 * np.pi * cp.fft.fftshift(cp.fft.fftfreq(self.arr_nodal.shape[0], d=grid.x.dx))
        fourier_functions = (spectrum[None, :] *
                             cp.exp(1j * full_wavenumbers[None, :] * grid.x.device_arr[:, None]))
        return sig.fftconvolve(cp.conj(fourier_functions), fourier_functions, mode='same', axes=1), full_wavenumbers

    def compute_hilbert(self, vt_shift, grid):
        # Compute hilbert transform of data with a phase shift
        self.fourier_transform()
        self.arr_spectral = cp.multiply(self.arr_spectral, cp.exp(-1j * grid.x.device_wavenumbers * vt_shift))
        self.inverse_fourier_transform()
        return ssig.hilbert(self.arr_nodal.get())


class Distribution:
    def __init__(self, resolutions, order, charge_mass):
        self.x_res, self.v_res = resolutions
        self.order = order
        self.charge_mass = charge_mass

        # arrays
        self.arr, self.arr_nodal = None, None
        self.zero_moment = SpaceScalar(resolution=resolutions[0])
        self.first_moment = SpaceScalar(resolution=resolutions[0])
        self.second_moment = SpaceScalar(resolution=resolutions[0])
        self.local_l2 = SpaceScalar(resolution=resolutions[0])

        # post-processing attributes
        self.avg_dist, self.delta_f = None, None

        # attributes for higher quad
        self.ell, self.gl_weights = None, None

    def compute_zero_moment(self, grid):
        self.inverse_fourier_transform()
        self.zero_moment.arr_nodal = grid.v.zero_moment(function=self.arr_nodal, idx=[1, 2])
        self.zero_moment.fourier_transform()
        # self.zero_moment.arr_spectral = grid.v.zero_moment(function=self.arr, idx=[1, 2])

    def total_momentum(self, grid):
        self.inverse_fourier_transform()
        self.first_moment.arr_nodal = grid.v.first_moment(function=self.arr_nodal, idx=[1, 2])
        return self.first_moment.integrate(grid=grid)

    def total_thermal_energy(self, grid):
        self.inverse_fourier_transform()
        self.second_moment.arr_nodal = grid.v.second_moment(function=self.arr_nodal, idx=[1, 2])
        return 0.5 * self.second_moment.integrate(grid=grid)

    def set_up_higher_quad(self, grid):
        """ f^2 is order 2(n-1) and needs GL quadrature of order n"""
        local_order = self.order
        gl_nodes, gl_weights = poly.legendre.leggauss(local_order)
        # Evaluate Legendre polynomials at finer grid
        ps = np.array([sp.legendre(s)(gl_nodes) for s in range(self.order)])
        # Interpolation polynomials at fine points
        ell = np.tensordot(grid.v.local_basis.inv_vandermonde, ps, axes=([0], [0]))
        self.ell = cp.asarray(ell)
        self.gl_weights = cp.asarray(gl_weights)

    def l2_norm(self, grid):
        """ Compute the L2-norm sqrt(integral(f^2, dx*dv)) """
        # Interpolated function at fine points
        interp_poly = cp.tensordot(self.arr, self.ell, axes=([2], [0]))
        # Integral in velocity-space
        quad = self.gl_weights[None, None, :] * interp_poly ** 2.0 / grid.v.J[None, :, None]
        self.local_l2.arr_nodal = quad.reshape((quad.shape[0], quad.shape[1] * quad.shape[2])).sum(axis=1)

        # return integral
        return cp.sqrt(self.local_l2.integrate(grid=grid))

    def average_distribution(self, grid):
        self.avg_dist = np.real(self.arr[0, :].get())

    def average_on_boundaries(self):
        self.arr[:, :, 0], self.arr[:, :, -1] = (
            (self.arr[:, :, 0] + cp.roll(self.arr, shift=+1, axis=1)[:, :, -1]) / 2,
            (cp.roll(self.arr, shift=-1, axis=1)[:, :, 0] + self.arr[:, :, -1]) / 2)

    def compute_delta_f(self):
        self.delta_f = self.arr_nodal.get() - self.avg_dist[None, :, :]

    def compute_average_gradient(self, grid):
        return np.tensordot(self.avg_dist,
                            grid.v.local_basis.derivative_matrix, axes=([1], [0])) * grid.v.J[:, None].get()

    def field_particle_covariance(self, Elliptic, Grid):
        fluctuation_field = cp.array(self.delta_f * Elliptic.field.arr_nodal.get()[:, None, None])
        return trapz2(fluctuation_field, Grid.x.dx).get() / Grid.x.length

    def variance_of_field_particle_covariance(self, Elliptic, Grid, covariance):
        fluctuation_field = cp.array(self.delta_f * Elliptic.field.arr_nodal.get()[:, None, None])
        return trapz2((fluctuation_field - cp.array(covariance)) ** 2, Grid.x.dx).get() / Grid.x.length

    def total_density(self, grid):
        self.inverse_fourier_transform()
        self.compute_zero_moment(grid=grid)
        return self.zero_moment.integrate(grid=grid)

    def grid_flatten(self):
        return self.arr_nodal.reshape(self.x_res, self.v_res * self.order)

    def spectral_flatten(self):
        return self.arr.reshape(self.arr.shape[0], self.v_res * self.order)

    def initialize_maxwellian(self, grid, vt, perturbation=True):
        maxwellian = grid.v.compute_maxwellian(thermal_velocity=vt, drift_velocity=0)

        if perturbation:
            f1 = cp.zeros((grid.x.device_modes.shape[0], self.v_res, self.order)) + 0j
            for idx in range(grid.x.wavenumbers.shape[0]):
                # if 50 < idx < 500:
                if idx == 300:
                    f1[idx, :, :] = cp.sqrt(1.0e-11) * grid.x.wavenumbers[idx] * maxwellian * cp.exp(
                        2j * cp.pi * cp.random.random(1))
        else:
            f1 = 0

        inverse = cp.fft.irfft(f1, axis=0, norm='forward')
        self.arr_nodal = inverse
        print('Finished initialization...')

    def initialize_bump_on_tail(self, grid, vt, u, chi, vb, vtb, perturbation=True):
        # ix, iv = cp.ones_like(grid.x.device_arr), cp.ones_like(grid.v.device_arr)
        # maxwellian = cp.tensordot(ix, grid.v.compute_maxwellian(thermal_velocity=vt,
        #                                                         drift_velocity=u), axes=0)
        # bump = chi * cp.tensordot(ix, grid.v.compute_maxwellian(thermal_velocity=vtb,
        #                                                         drift_velocity=vb), axes=0)
        maxwellian = grid.v.compute_maxwellian(thermal_velocity=vt, drift_velocity=u)
        bump = chi * grid.v.compute_maxwellian(thermal_velocity=vtb, drift_velocity=vb)
        f = (maxwellian + bump) / (1 + chi)
        # self.arr_nodal = (maxwellian + bump) / (1 + chi)
        # self.fourier_transform()

        # compute perturbation
        if perturbation:
            # obtain eigenvalues by solving the dispersion relation
            sols = np.zeros_like(grid.x.wavenumbers) + 0j
            # guess_r, guess_i = 0.03 / grid.x.fundamental, -0.003 / grid.x.fundamental  # L=1000
            # guess_r, guess_i = 0.02 / grid.x.fundamental, -0.002 / grid.x.fundamental  # L=
            guess_r, guess_i = 5.5, -23 / 20
            for idx, wave in enumerate(grid.x.wavenumbers):
                if idx == 0:
                    continue
                solution = opt.root(dispersion.dispersion_fsolve, x0=np.array([guess_r, guess_i]),
                                    args=(wave, u, vt, chi, vb, vtb), jac=dispersion.jacobian_fsolve, tol=1.0e-15)
                guess_r, guess_i = solution.x
                sols[idx] = (guess_r + 1j * guess_i)

            df = (grid.v.compute_maxwellian_gradient(thermal_velocity=vt, drift_velocity=u) +
                  chi * grid.v.compute_maxwellian_gradient(thermal_velocity=vtb, drift_velocity=vb)) / (1 + chi)

            def eigenfunction(z, k):
                pi2 = 2.0 * np.pi
                return (df / (z - grid.v.device_arr)) / k * cp.exp(1j * pi2 * cp.random.random(1))

            unstable_modes = grid.x.wavenumbers[np.imag(sols) > 0.003]
            mode_idxs = grid.x.device_modes[np.imag(sols) > 0.003]
            unstable_eigs = sols[np.imag(sols) > 0.003]
            largest_growth_rate = cp.amax(np.imag(unstable_eigs) * unstable_modes)
            smallest_growth_rate = cp.amin(np.absolute(np.imag(unstable_eigs) * unstable_modes))
            # eig_sum, pi2 = 0, 2 * np.pi
            # f1 = cp.zeros_like(self.arr) + 0j
            f1 = cp.zeros((grid.x.device_modes.shape[0], self.v_res, self.order)) + 0j
            for idx in range(sols.shape[0]):
                if idx < 50:
                    continue
                if np.imag(sols[idx]) > 0:
                    growth_rate = 1.0e-3 * np.abs(np.imag(sols[idx])) * grid.x.wavenumbers[idx] / largest_growth_rate
                    f1[idx, :, :] = (-self.charge_mass * growth_rate *
                                     eigenfunction(sols[idx], grid.x.wavenumbers[idx]))
                else:
                    # growth_rate = 1.0e-3 * grid.x.wavenumbers[idx]  # * smallest_growth_rate
                    # f1[idx, :, :] = (-self.charge_mass * growth_rate *
                    #                  eigenfunction(sols[idx], grid.x.wavenumbers[idx]))
                    f1[idx, :, :] = cp.sqrt(1.0e-11) * grid.x.wavenumbers[idx] * f * cp.exp(2j * cp.pi * cp.random.random(1))
            # for idx in range(unstable_modes.shape[0])
            #     growth_rate = 1.0e-3 * np.imag(unstable_eigs[idx]) * unstable_modes[idx] / largest_growth_rate
            #     f1[mode_idxs[idx], :, :] = (-self.charge_mass * growth_rate *
            #                                 eigenfunction(unstable_eigs[idx], unstable_modes[idx]))
            # for idx in range(grid.x.device_modes.shape[0]):
            #     if idx == 0:
            #         continue
            #     f1[idx, :, :] = -self.charge_mass * 1.0e-10 * cp.exp(2j * np.pi * cp.random.random(1))

        else:
            f1 = 0

        # f1 = f1 / cp.amax(cp.absolute(f1))
        # print(cp.amax(cp.absolute(f1)))
        inverse = cp.fft.irfft(f1, axis=0, norm='forward')
        self.arr_nodal = inverse  # 1.0e-3 * inverse / cp.amax(inverse)
        print('Finished initialization...')

    def fourier_transform(self):
        # self.arr = cp.fft.fftshift(cp.fft.fft(self.arr_nodal, axis=0, norm='forward'), axes=0)
        self.arr = cp.fft.rfft(self.arr_nodal, axis=0, norm='forward')

    def inverse_fourier_transform(self):
        # self.arr_nodal = cp.real(cp.fft.ifft(cp.fft.fftshift(self.arr, axes=0), norm='forward', axis=0))
        self.arr_nodal = cp.fft.irfft(self.arr, axis=0, norm='forward')


class Scalar:
    def __init__(self, resolution, order):
        self.res = resolution
        self.order = order

        # arrays
        self.arr, self.grad = None, None
        self.grad2 = None
        self.arr_spectral, self.grad_spectral = None, None

    def initialize_maxwellian(self, grid, vt):
        self.arr = grid.v.compute_maxwellian(thermal_velocity=vt, drift_velocity=0)

    def initialize_bump_on_tail(self, grid, vt, u, chi, vb, vtb):
        maxwellian = grid.v.compute_maxwellian(thermal_velocity=vt, drift_velocity=u)
        bump = chi * grid.v.compute_maxwellian(thermal_velocity=vtb, drift_velocity=vb)
        self.arr = (maxwellian + bump) / (1 + chi)

    def compute_grad(self, grid):
        self.grad = cp.tensordot(self.arr,
                                 grid.local_basis.derivative_matrix, axes=([1], [0])) * grid.J[:, None]

    def compute_second_grad(self, grid):
        self.grad2 = cp.tensordot(self.grad,
                                  grid.local_basis.derivative_matrix, axes=([1], [0])) * grid.J[:, None]

    def fourier_transform(self, grid):
        self.arr_spectral = np.tensordot(self.arr, grid.fourier_quads, axes=([0, 1], [1, 2]))

    def fourier_grad(self, grid):
        # self.grad_spectral = np.tensordot(self.grad, grid.fourier_quads, axes=([0, 1], [1, 2]))
        self.grad_spectral = 1j * grid.modes * self.arr_spectral

    def hilbert_transform_grad(self, grid):
        analytic = cp.sum(2.0 * self.grad_spectral[None, None, :] * grid.grid_phases, axis=2)
        pv_integral = -1.0 * cp.pi * cp.imag(analytic)

        return pv_integral

    def zero_moment(self, grid):
        return cp.tensordot(self.arr,
                            grid.v.global_quads / grid.v.J[:, None], axes=([0, 1], [0, 1]))

    def second_moment(self, grid):
        return cp.tensordot(self.arr * (0.5 * grid.v.device_arr ** 2.0),
                            grid.v.global_quads / grid.v.J[:, None], axes=([0, 1], [0, 1]))


def trapz(y, dx):
    """ Custom trapz routine using cupy """
    return cp.sum(y[:-1] + y[1:]) * dx / 2.0


def trapz2(y, dx):
    return cp.sum(y[:-1, :] + y[1:, :], axis=0) * dx / 2.0
