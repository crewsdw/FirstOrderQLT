import numpy as np
import cupy as cp
import basis as b
import tools.plasma_dispersion as pd
import matplotlib.pyplot as plt
import scipy.special as sp


class SpaceGrid:
    """ In this scheme, the spatial grid is uniform and transforms are accomplished by DFT """
    def __init__(self, low, high, elements):
        # grid limits and elements
        self.low, self.high = low, high
        self.elements = elements

        # domain and element widths
        self.length = self.high - self.low
        self.dx = self.length / self.elements

        # element Jacobian
        self.J = 2.0 / self.dx

        # arrays
        self.arr, self.device_arr = None, None
        self.create_grid()

        # spectral properties
        self.modes = elements // 2 + 1  # Nyquist frequency
        self.fundamental = 2.0 * np.pi / self.length
        # self.wavenumbers = self.fundamental * np.arange(-self.modes, self.modes)
        self.wavenumbers = self.fundamental * np.arange(self.modes)
        # print(self.wavenumbers)
        self.device_modes = cp.arange(self.modes)
        self.device_wavenumbers = cp.array(self.wavenumbers)
        self.device_wavenumbers_fourth = self.device_wavenumbers ** 4.0
        self.device_wavenumbers_fourth[self.device_wavenumbers < 0.5] = 0  # 1
        self.zero_idx = 0  # int(self.modes)
        # self.two_thirds_low = int((1 * self.modes)//3 + 1)
        # self.two_thirds_high = self.wavenumbers.shape[0] - self.two_thirds_low
        self.pad_width = int((1 * self.modes)//3 + 1)
        # print(self.two_thirds_low)
        # print(self.two_thirds_high)
        print(self.length)
        print(self.fundamental)

    def create_grid(self):
        """ Build evenly spaced grid, assumed periodic """
        self.arr = np.linspace(self.low, self.high - self.dx, num=self.elements)
        self.device_arr = cp.asarray(self.arr)


class VelocityGrid:
    """ In this experiment, the velocity grid is an LGL quadrature grid """
    def __init__(self, low, high, elements, order):
        self.low, self.high = low, high
        self.elements, self.order = elements, order
        self.local_basis = b.LGLBasis1D(order=self.order)

        self.element_idxs = np.arange(self.elements)

        # domain and element widths
        self.length = self.high - self.low
        self.dx = self.length / self.elements

        # arrays
        self.arr, self.device_arr = None, None
        self.mid_points = None
        self.create_even_grid()

        # stretch / transform elements
        self.pole_distance = 5
        self.dx_grid = None
        self.dx_grid = self.dx * np.ones(self.elements)
        # self.stretch_grid()
        # TS:
        # self.create_triple_grid(lows=np.array([self.low, -5, 5]),
        #                         highs=np.array([-5, 5, self.high]),
        #                         elements=np.array([20, 60, 20]))
        # BOT:
        self.create_pentic_grid(lows=np.array([self.low, -5, 3, 8, 12]),
                                highs=np.array([-5, 3, 8, 12, self.high]),
                                elements=np.array([10, 15, 35, 10, 10]))
        # TS:
        # self.create_pentic_grid(lows=np.array([self.low, -10, 3, 8, 12]),
        #                         highs=np.array([-5, 3, 8, 12, self.high]),
        #                         elements=np.array([5, 5, 30, 5, 5]))

        # jacobian
        self.J = cp.asarray(2.0 / self.dx_grid)
        self.J_host = self.J.get()
        self.min_dv = cp.amin(self.dx_grid)
        plt.figure()
        for i in range(self.elements):
            plt.plot(np.zeros_like(self.arr[i, :]), self.arr[i, :], 'ko')
        plt.show()

        # global quad weights
        self.global_quads = cp.tensordot(cp.ones(elements),
                                         cp.asarray(self.local_basis.weights), axes=0)

        # global translation matrix
        mid_identity = np.tensordot(self.mid_points, np.eye(self.local_basis.order), axes=0)
        self.translation_matrix = cp.asarray(mid_identity +
                                             self.local_basis.translation_matrix[None, :, :] /
                                             self.J[:, None, None].get())

        # quad matrix
        self.modes = 2.0 * np.pi / self.length * np.arange(int(2 * self.elements))  # only positive frequencies
        self.fourier_quads = (self.local_basis.weights[None, None, :] *
                              np.exp(-1j*self.modes[:, None, None]*self.arr[None, :, :]) /
                              self.J[None, :, None].get()) / self.length
        self.grid_phases = np.exp(1j * self.modes[None, None, :] * self.arr[:, :, None])

    def create_triple_grid(self, lows, highs, elements):
        """ Build a three-segment grid, each evenly-spaced """
        # translate to [0, 1]
        nodes_iso = (np.array(self.local_basis.nodes) + 1) / 2
        # element left boundaries (including ghost elements)
        dxs = (highs - lows) / elements
        xl0 = np.linspace(lows[0], highs[0] - dxs[0], num=elements[0])
        xl1 = np.linspace(lows[1], highs[1] - dxs[1], num=elements[1])
        xl2 = np.linspace(lows[2], highs[2] - dxs[2], num=elements[2])
        # construct coordinates
        self.arr = np.zeros((elements[0] + elements[1] + elements[2], self.order))
        for i in range(elements[0]):
            self.arr[i, :] = xl0[i] + dxs[0] * nodes_iso
        for i in range(elements[1]):
            self.arr[elements[0] + i, :] = xl1[i] + dxs[1] * nodes_iso
        for i in range(elements[2]):
            self.arr[elements[0] + elements[1] + i, :] = xl2[i] + dxs[2] * nodes_iso
        # send to device
        self.device_arr = cp.asarray(self.arr)
        self.mid_points = np.array([0.5 * (self.arr[i, -1] + self.arr[i, 0]) for i in range(self.elements)])
        self.dx_grid = self.device_arr[:, -1] - self.device_arr[:, 0]

    def create_pentic_grid(self, lows, highs, elements):
        """ Build a five-segment grid, each evenly-spaced """
        # translate to [0, 1]
        nodes_iso = (np.array(self.local_basis.nodes) + 1) / 2
        # element left boundaries (including ghost elements)
        dxs = (highs - lows) / elements
        xl0 = np.linspace(lows[0], highs[0] - dxs[0], num=elements[0])
        xl1 = np.linspace(lows[1], highs[1] - dxs[1], num=elements[1])
        xl2 = np.linspace(lows[2], highs[2] - dxs[2], num=elements[2])
        xl3 = np.linspace(lows[3], highs[3] - dxs[3], num=elements[3])
        xl4 = np.linspace(lows[4], highs[4] - dxs[4], num=elements[4])
        # construct coordinates
        self.arr = np.zeros((elements[0] + elements[1] + elements[2] + elements[3] + elements[4], self.order))
        for i in range(elements[0]):
            self.arr[i, :] = xl0[i] + dxs[0] * nodes_iso
        for i in range(elements[1]):
            self.arr[elements[0] + i, :] = xl1[i] + dxs[1] * nodes_iso
        for i in range(elements[2]):
            self.arr[elements[0] + elements[1] + i, :] = xl2[i] + dxs[2] * nodes_iso
        for i in range(elements[3]):
            self.arr[elements[0] + elements[1] + elements[2] + i, :] = xl3[i] + dxs[3] * nodes_iso
        for i in range(elements[4]):
            self.arr[elements[0] + elements[1] + elements[2] + elements[3] + i, :] = xl4[i] + dxs[4] * nodes_iso
        # send to device
        self.device_arr = cp.asarray(self.arr)
        self.mid_points = np.array([0.5 * (self.arr[i, -1] + self.arr[i, 0]) for i in range(self.elements)])
        self.dx_grid = self.device_arr[:, -1] - self.device_arr[:, 0]

    def create_even_grid(self):
        """ Build global grid """
        # translate to [0, 1]
        nodes_iso = (np.array(self.local_basis.nodes) + 1) / 2
        # element left boundaries (including ghost elements)
        xl = np.linspace(self.low, self.high - self.dx, num=self.elements)
        # construct coordinates
        self.arr = np.zeros((self.elements, self.order))
        for i in range(self.elements):
            self.arr[i, :] = xl[i] + self.dx * np.array(nodes_iso)
        # send to device
        self.device_arr = cp.asarray(self.arr)
        self.mid_points = np.array([0.5 * (self.arr[i, -1] + self.arr[i, 0]) for i in range(self.elements)])

    def stretch_grid(self):
        # Investigate grid mapping
        # two-stream
        alphas, betas = (np.array([1]), np.array([1]))
        # bump-on-tail
        # alphas, betas = (np.array([0.6, 0.6, 0.6, 0.6, 0.6]),  # 0.64]),  # , 0.705, 0.705, 0.705, 0.705]),
        #                  np.array([0.5, 0.5, 0.5, 0.5, 0.5]))
        # alphas, betas = (np.array([0.62, 0.62, 0.62, 0.62]),  # , 0.705, 0.705, 0.705, 0.705]),
        #                  np.array([0.5, 0.5, 0.5, 0.5]))  # , 0.6, 0.6, 0.6, 0.6]))
        # plt.figure()
        # plt.plot(self.arr.flatten(), self.iterate_map_asym(self.arr, alphas[:-2], betas[:-2]).flatten(),
        #          'k', label=r'Iteration 1')
        # plt.plot(self.arr.flatten(), self.iterate_map_asym(self.arr, alphas[:-1], betas[:-1]).flatten(),
        #          'k', label=r'Iteration 2')
        # plt.plot(self.arr.flatten(), self.iterate_map_asym(self.arr, alphas, betas).flatten(),
        #          'r', label=r'Iteration 3')
        # plt.xlabel('Input points'), plt.ylabel('Output points')
        # plt.grid(True), plt.legend(loc='best'), plt.tight_layout()
        # plt.show()
        # Map points
        # Map lows and highs
        # orders = [0.4, 0.5, 0.8, 0.8, 0.8]  # , 0.8, 0.8]
        # mapped_lows = self.iterate_map(self.arr[:, 0], orders=orders)
        # mapped_highs = self.iterate_map(self.arr[:, -1], orders=orders)
        # alphas, betas = np.array([0.3, 0.55]), np.array([0.35, 0.6])
        mapped_lows = self.iterate_map_asym(self.arr[:, 0], alphas=alphas, betas=betas)
        mapped_highs = self.iterate_map_asym(self.arr[:, -1], alphas=alphas, betas=betas)
        self.dx_grid = mapped_highs - mapped_lows
        # self.dx_grid = self.dx * self.grid_map(self.mid_points)  # mapped_highs - mapped_lows
        # print(self.dx_grid)
        # xl = np.linspace(self.low, self.high - self.dx, num=self.elements)
        # Overwrite coordinate array
        nodes_iso = (np.array(self.local_basis.nodes) + 1) / 2
        lows = np.zeros(self.elements+1)
        lows[0] = self.low
        for i in range(self.elements):
            self.arr[i, :] = lows[i] + self.dx_grid[i] * np.array(nodes_iso)
            lows[i+1] = self.arr[i, -1]
        # plt.figure()
        # plt.plot(self.arr.flatten(), 'o--')
        # plt.show()
        # send to device
        self.device_arr = cp.asarray(self.arr)
        self.mid_points = np.array([0.5 * (self.arr[i, -1] + self.arr[i, 0]) for i in range(self.elements)])

    def iterate_map(self, points, orders):
        for order in orders:
            points = self.grid_map(points, order)
        return points

    def iterate_map_asym(self, points, alphas, betas):
        for idx, alpha in enumerate(alphas):
            points = self.grid_map_asym(points, alpha, betas[idx])
        return points

    def grid_map_asym(self, points, alpha, beta):
        return (self.low * ((self.high - points) / self.length) ** alpha +
                self.high * ((points - self.low) / self.length) ** beta)

    def grid_map(self, points, order):
        return (self.low * ((self.high - points) / self.length) ** order +
                self.high * ((points - self.low) / self.length) ** order)

    def get_local_velocity(self, phase_velocity):
        # find out which element the phase velocity is in
        idx = self.element_idxs[(self.arr[:, 0] < np.real(phase_velocity)) &
                                (np.real(phase_velocity) <= self.arr[:, -1])]

        # get local velocity
        velocity = self.J_host[idx] * (phase_velocity - self.mid_points[idx])

        return idx, velocity

    def get_interpolant_on_point(self, velocity):
        # get interpolation polynomial on point
        vandermonde_on_point = np.array([sp.legendre(s)(velocity)
                                         for s in range(self.order)])
        interpolant_on_point = np.tensordot(self.local_basis.inv_vandermonde,
                                            vandermonde_on_point, axes=([0], [0]))
        return interpolant_on_point

    def get_interpolant_grad_on_point(self, velocity):
        # get interpolation polynomial on point
        vandermonde_grad_on_point = np.array([sp.legendre(s).deriv()(velocity)
                                              for s in range(self.order)])
        interpolant_grad_on_point = np.tensordot(self.local_basis.inv_vandermonde,
                                                 vandermonde_grad_on_point, axes=([0], [0]))
        return interpolant_grad_on_point

    def zero_moment(self, function, idx):
        return cp.tensordot(self.global_quads / self.J[:, None], function, axes=([0, 1], idx))

    def first_moment(self, function, idx):
        return cp.tensordot(self.global_quads / self.J[:, None], cp.multiply(self.device_arr[None, :, :], function),
                            axes=([0, 1], idx))

    def second_moment(self, function, idx):
        return cp.tensordot(self.global_quads / self.J[:, None], cp.multiply(self.device_arr[None, :, :] ** 2.0,
                                                           function),
                            axes=([0, 1], idx))

    def compute_maxwellian(self, thermal_velocity, drift_velocity):
        return cp.exp(-0.5 * ((self.device_arr - drift_velocity) /
                        thermal_velocity) ** 2.0) / (np.sqrt(2.0 * np.pi) * thermal_velocity)

    def compute_maxwellian_gradient(self, thermal_velocity, drift_velocity):
        return (-1.0*((self.device_arr - drift_velocity) / thermal_velocity ** 2.0) *
                self.compute_maxwellian(thermal_velocity=thermal_velocity, drift_velocity=drift_velocity))


class PhaseSpace:
    """ In this experiment, PhaseSpace consists of equispaced nodes and a finite LGL quad grid in velocity"""
    def __init__(self, lows, highs, elements, order):
        self.x = SpaceGrid(low=lows[0], high=highs[0], elements=elements[0])
        self.v = VelocityGrid(low=lows[1], high=highs[1], elements=elements[1], order=order)

    def eigenfunction(self, thermal_velocity, drift_velocity, eigenvalue, beams='two-stream'):
        if beams == 'one':
            df = self.v.compute_maxwellian_gradient(thermal_velocity=thermal_velocity,
                                                    drift_velocity=drift_velocity)
            zeta = eigenvalue / self.x.fundamental
            denom = zeta - self.v.device_arr
            denom_abs = np.absolute(denom)
            denom_ang = np.angle(denom)
            # denom = np.absolute(zeta - self.v.device_arr)
            # v_part = df / (eigenvalue - self.x.fundamental * self.v.device_arr) / self.x.fundamental
            v_part = df / denom_abs / (self.x.fundamental ** 2.0)  # * np.exp(-1j * denom_ang)
            xv_part = cp.cos(self.x.fundamental * self.x.device_arr[:, None, None] - denom_ang[None, :, :])
            return xv_part * v_part[None, :, :]

        if beams == 'two-stream':
            df1 = self.v.compute_maxwellian_gradient(thermal_velocity=thermal_velocity,
                                                     drift_velocity=drift_velocity[0])
            df2 = self.v.compute_maxwellian_gradient(thermal_velocity=thermal_velocity,
                                                     drift_velocity=drift_velocity[1])
            df = 0.5 * (df1 + df2)
            v_part = cp.divide(df, self.v.device_arr - eigenvalue)
            return cp.tensordot(cp.exp(1j * self.x.fundamental * self.x.device_arr), v_part, axes=0) * cp.exp(0.1j)
