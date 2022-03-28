# import numpy as np
import cupy as cp
import variables as var
from copy import deepcopy


def basis_product(flux, basis_arr, axis):
    return cp.tensordot(flux, basis_arr,
                        axes=([axis], [1]))


class MeanFlux:
    def __init__(self, resolution, order, charge_mass):
        self.v_res = resolution
        self.order = order

        # slices into the DG boundaries (list of tuples)
        self.boundary_slices = [(slice(self.v_res), 0),
                                (slice(self.v_res), -1)]
        self.boundary_slices_pad = [(slice(self.v_res + 2), 0),
                                    (slice(self.v_res + 2), -1)]
        self.boundary_slices_pad = [(slice(self.v_res + 2), 0),
                                    (slice(self.v_res + 2), -1)]

        self.num_flux_size = (self.v_res, 2)

        # arrays
        self.flux = var.Scalar(resolution=resolution, order=order)
        self.output = var.Scalar(resolution=resolution, order=order)

        # charge sign
        self.charge = charge_mass

    def semi_discrete_rhs(self, mean_distribution, fluctuating_distribution, elliptic, grid):
        """ Computes the semi-discrete equation for velocity flux only """
        # Compute the flux
        num_flux = self.compute_flux(mean_distribution=mean_distribution,
                                     fluctuating_distribution=fluctuating_distribution, elliptic=elliptic, grid=grid)
        self.output.arr = (grid.v.J[:, None] * self.v_flux(grid=grid, num_flux=num_flux))

    def compute_flux(self, mean_distribution, fluctuating_distribution, elliptic, grid):
        self.flux.arr = trapz2(y=(fluctuating_distribution.arr_nodal *
                                  self.charge * elliptic.field.arr_nodal[:, None, None]),
                               dx=grid.x.dx) / grid.x.length

        # return self.nodal_central_flux(flux=self.flux.arr)
        return self.nodal_lax_friedrichs_flux(mean_distribution=mean_distribution, flux=self.flux.arr)

    def nodal_central_flux(self, flux):
        # Allocate
        num_flux = cp.zeros((self.v_res, 2))

        # set padded flux
        padded_flux = cp.zeros((self.v_res + 2, self.order))
        padded_flux[1:-1, :] = flux

        # Central flux
        num_flux[self.boundary_slices[0]] = -0.5 * (cp.roll(padded_flux[self.boundary_slices_pad[1]],
                                                            shift=+1, axis=0)[1:-1] +
                                                    self.flux.arr[self.boundary_slices[0]])
        num_flux[self.boundary_slices[1]] = +0.5 * (cp.roll(padded_flux[self.boundary_slices_pad[0]],
                                                            shift=-1, axis=0)[1:-1] +
                                                    self.flux.arr[self.boundary_slices[1]])

        return num_flux

    def nodal_lax_friedrichs_flux(self, mean_distribution, flux):
        # Allocate
        num_flux = cp.zeros((self.v_res, 2))

        # set padded flux
        padded_flux = cp.zeros((self.v_res + 2, self.order))
        padded_flux[1:-1, :] = flux

        # Central flux
        num_flux[self.boundary_slices[0]] = -0.5 * (cp.roll(padded_flux[self.boundary_slices_pad[1]],
                                                            shift=+1, axis=0)[1:-1] +
                                                    self.flux.arr[self.boundary_slices[0]])
        num_flux[self.boundary_slices[1]] = +0.5 * (cp.roll(padded_flux[self.boundary_slices_pad[0]],
                                                            shift=-1, axis=0)[1:-1] +
                                                    self.flux.arr[self.boundary_slices[1]])

        # re-use padded_flux array for padded_distribution
        constant = cp.amax(cp.array([padded_flux[:-1, -1], padded_flux[1:, 0]]), axis=0)
        padded_flux[1:-1, :] = mean_distribution.arr

        # Additional flux central -> lax-friedrichs
        num_flux[self.boundary_slices[0]] += +0.5 * cp.multiply(constant[:-1],
                                                                (cp.roll(padded_flux[self.boundary_slices_pad[1]],
                                                                         shift=+1, axis=0)[1:-1] -
                                                                 mean_distribution.arr[self.boundary_slices[0]]))
        num_flux[self.boundary_slices[1]] += -0.5 * cp.multiply(constant[1:],
                                                                (cp.roll(padded_flux[self.boundary_slices_pad[0]],
                                                                         shift=-1, axis=0)[1:-1] -
                                                                 mean_distribution.arr[self.boundary_slices[1]]))

        return num_flux

    def v_flux(self, grid, num_flux):
        return (basis_product(flux=self.flux.arr, basis_arr=grid.v.local_basis.internal, axis=1) -
                basis_product(flux=num_flux, basis_arr=grid.v.local_basis.numerical, axis=1))


class FluctuationFlux:
    def __init__(self, resolutions, order, charge_mass, nu):
        self.x_ele, self.v_res = resolutions
        self.x_res = int(self.x_ele // 2 + 1)
        self.order = order

        # slices into the DG boundaries (list of tuples)
        self.boundary_slices = [(slice(self.x_res), slice(self.v_res), 0),
                                (slice(self.x_res), slice(self.v_res), -1)]
        self.boundary_slices_pad = [(slice(self.x_res), slice(self.v_res + 2), 0),
                                    (slice(self.x_res), slice(self.v_res + 2), -1)]
        self.boundary_slices_pad = [(slice(self.x_res), slice(self.v_res + 2), 0),
                                    (slice(self.x_res), slice(self.v_res + 2), -1)]
        self.num_flux_size = (self.x_res, self.v_res, 2)

        # for array padding
        self.pad_field, self.pad_spectrum = None, None

        # arrays
        self.flux = var.Distribution(resolutions=resolutions, order=order, charge_mass=None)
        self.output = var.Distribution(resolutions=resolutions, order=order, charge_mass=None)

        # species dependence
        self.charge = charge_mass
        self.nu = nu  # hyperviscosity

    def semi_discrete_rhs(self, mean_distribution, elliptic, grid):
        """ Computes the semi-discrete equation for velocity flux only """
        # Compute the flux
        num_flux = self.compute_flux(mean_distribution=mean_distribution, elliptic=elliptic, grid=grid)
        self.output.arr = (grid.v.J[None, :, None] * self.v_flux(grid=grid, num_flux=num_flux))
        # self.output.arr -= self.nu * grid.x.device_wavenumbers_fourth[:, None, None] * distribution.arr

    def initialize_zero_pad(self, grid):
        self.pad_field = cp.zeros((grid.x.modes + grid.x.pad_width)) + 0j
        # self.pad_spectrum = cp.zeros((grid.x.modes + grid.x.pad_width,
        #                               self.v_res, self.order)) + 0j

    def compute_flux(self, mean_distribution, elliptic, grid):
        """ Compute the flux convolution(field, distribution) using pseudospectral method """
        # Zero-pad
        self.pad_field[:-grid.x.pad_width] = elliptic.field.arr_spectral

        # Pseudospectral product
        field_nodal = cp.fft.irfft(self.pad_field, norm='forward')
        nodal_flux = self.charge * cp.multiply(field_nodal[:, None, None], mean_distribution.arr[None, :, :])

        # Compute upwind numerical flux
        num_flux = self.nodal_upwind_flux(flux=nodal_flux, field=self.charge * field_nodal)

        # Transform back
        self.flux.arr = cp.fft.rfft(nodal_flux, norm='forward', axis=0)[:-grid.x.pad_width, :, :]

        # return numerical flux
        return cp.fft.rfft(num_flux, norm='forward', axis=0)[:-grid.x.pad_width, :, :]

    def nodal_upwind_flux(self, flux, field):
        # Allocate
        # num_flux = cp.zeros(self.num_flux_size) + 0j
        num_flux = cp.zeros((flux.shape[0], self.v_res, 2))

        # Alternative:
        one_negatives = cp.where(condition=field < 0, x=1, y=0)
        one_positives = cp.where(condition=field >= 0, x=1, y=0)

        # set padded flux
        padded_flux = cp.zeros((num_flux.shape[0], self.v_res + 2, self.order))  # + 0j
        # print(flux.shape)
        # print(padded_flux.shape)
        # quit()
        padded_flux[:, 1:-1, :] = flux  # self.flux.arr
        # padded_flux[:, 0, -1] = 0.0  # -self.flux.arr[:, 0, 0]
        # padded_flux[:, -1, 0] = 0.0  # -self.flux.arr[:, -1, 0]

        self.boundary_slices = [(slice(num_flux.shape[0]), slice(self.v_res), 0),
                                (slice(num_flux.shape[0]), slice(self.v_res), -1)]
        self.boundary_slices_pad = [(slice(num_flux.shape[0]), slice(self.v_res + 2), 0),
                                    (slice(num_flux.shape[0]), slice(self.v_res + 2), -1)]

        # Upwind flux, left face
        num_flux[self.boundary_slices[0]] = -1.0 * (cp.multiply(cp.roll(padded_flux[self.boundary_slices_pad[1]],
                                                                        shift=+1,
                                                                        axis=1)[:, 1:-1],
                                                                one_positives[:, None]) +
                                                    cp.multiply(padded_flux[self.boundary_slices_pad[0]][:, 1:-1],
                                                                one_negatives[:, None]))
        # Upwind fluxes, right face
        num_flux[self.boundary_slices[1]] = (cp.multiply(padded_flux[self.boundary_slices_pad[1]][:, 1:-1],
                                                         one_positives[:, None]) +
                                             cp.multiply(cp.roll(padded_flux[self.boundary_slices_pad[0]],
                                                                 shift=-1,
                                                                 axis=1)[:, 1:-1],
                                                         one_negatives[:, None]))

        return num_flux

    def v_flux(self, grid, num_flux):
        return (basis_product(flux=self.flux.arr, basis_arr=grid.v.local_basis.internal, axis=2) -
                basis_product(flux=num_flux, basis_arr=grid.v.local_basis.numerical, axis=2))

    def spectral_advection(self, distribution_arr, grid):
        return -1.0j * cp.multiply(grid.x.device_wavenumbers[:, None, None],
                                   cp.einsum('ijk,mik->mij', grid.v.translation_matrix, distribution_arr))


def trapz2(y, dx):
    y_int = cp.zeros((y.shape[0] + 1, y.shape[1], y.shape[2]))
    y_int[:-1, :, :] = deepcopy(y)
    y_int[-1, :, :] = y[0, :, :]
    return cp.sum(y_int[:-1, :, :] + y_int[1:, :, :], axis=0) * dx / 2.0
