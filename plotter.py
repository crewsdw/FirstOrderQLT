import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import variables as var
import elliptic as ell


# from mpl_toolkits.axes_grid1 import make_axes_locatable


class Plotter:
    def __init__(self, grid, colormap='RdPu'):
        self.colormap = colormap
        self.grid = grid
        self.length = grid.x.length
        # Build structured grid, nodal
        self.X, self.V = np.meshgrid(grid.x.arr.flatten(), grid.v.arr.flatten(), indexing='ij')
        self.x = grid.x.arr
        self.v = grid.v.arr.flatten()
        self.k = grid.x.wavenumbers / grid.x.fundamental
        self.fundamental = grid.x.fundamental
        # Build structured grid, global spectral
        self.FX, self.FV = np.meshgrid(grid.x.wavenumbers, grid.v.arr.flatten(),
                                       indexing='ij')  # / grid.x.fundamental

    def compute_two_time_correlation_function(self, distribution1, distribution2, time_delay):
        # Obtain delta_f's
        distribution1.fourier_transform()
        distribution1.arr[0, :] = 0
        distribution1.inverse_fourier_transform()
        distribution2.fourier_transform()
        vt_shift = -0.8 * time_delay
        cp.multiply(distribution2.arr, cp.exp(-1j * self.grid.x.device_wavenumbers * vt_shift)[:, None, None])
        distribution2.arr[0, :] = 0
        distribution2.inverse_fourier_transform()
        # Integrate across x
        mixed_arr = distribution1.arr_nodal * distribution2.arr_nodal
        return cp.sum(mixed_arr[:-1, :, :] + mixed_arr[1:, :, :], axis=0) * (self.x[1] - self.x[0]) / 2.0

    def plot_two_time_correlation_function(self, distribution_data, time_data, elements, order):
        two_time_corr = cp.zeros((time_data.shape[0], time_data.shape[0], elements[1], order))
        t1, t2 = cp.zeros((time_data.shape[0], time_data.shape[0])), cp.zeros((time_data.shape[0], time_data.shape[0]))
        for idx1, time1 in enumerate(time_data):
            distribution1 = var.Distribution(resolutions=elements, order=order, charge_mass=-1.0)
            distribution1.arr_nodal = cp.asarray(distribution_data[idx1])
            for idx2, time2 in enumerate(time_data):
                distribution2 = var.Distribution(resolutions=elements, order=order, charge_mass=-1.0)
                distribution2.arr_nodal = cp.asarray(distribution_data[idx2])
                two_time_corr[idx1, idx2] = self.compute_two_time_correlation_function(distribution1=distribution1,
                                                                                       distribution2=distribution2)
                t1[idx1, idx2] = time1
                t2[idx1, idx2] = time2

        print(self.grid.v.arr[20, 5])
        plt.figure()
        plt.contourf(t1.get(), t2.get(), two_time_corr[:, :, 20, 5].get())
        plt.axis('equal')
        plt.title(r'Two-time correlation '
                  r'$\langle f_1(x,v,t_1)f_1(x,v,t_2)\rangle_L$ at $v=${:0.2f}'.format(self.grid.v.arr[20, 5]))
        plt.xlabel(r'Time $t_1$'), plt.ylabel(r'Time $t_2$')
        plt.colorbar(), plt.tight_layout()
        plt.show()

    def plot_autocorrelation_function(self, distribution_data, time_data, elements, order):
        # Compute autocorrelation <f_1(x,v,t_sat)f_1(x - vg*tau,v,t_sat + tau)>_L
        t_idx = 125
        autocorr = cp.zeros((time_data.shape[0], elements[1], order))
        distribution1 = var.Distribution(resolutions=elements, order=order, charge_mass=-1.0)
        distribution1.arr_nodal = cp.asarray(distribution_data[t_idx])
        time_delay = cp.zeros((time_data.shape[0]))
        for idx, tau in enumerate(time_data):
            distribution2 = var.Distribution(resolutions=elements, order=order, charge_mass=-1.0)
            distribution2.arr_nodal = cp.asarray(distribution_data[idx])
            time_delay[idx] = tau - time_data[t_idx]
            autocorr[idx, :, :] = self.compute_two_time_correlation_function(distribution1=distribution1,
                                                                             distribution2=distribution2,
                                                                             time_delay=time_delay[idx])

        # autocorr = cp.log(1 + cp.abs(autocorr))
        # Plot it
        start_idx, stop_idx = 12, 27
        cb = np.linspace(np.amin(autocorr[:, start_idx:stop_idx, :].get()),
                         np.amax(autocorr[:, start_idx:stop_idx, :].get()), num=200)
        T, V = np.meshgrid(time_delay.get(), self.grid.v.arr[start_idx:stop_idx, :].flatten(), indexing='ij')
        plt.figure()
        # plt.contourf(T, V, autocorr[:, start_idx:stop_idx, :].reshape(time_data.shape[0],
        #                                                               (start_idx - stop_idx) * order).get(), cb)
        plt.pcolormesh(T, V, autocorr[:, start_idx:stop_idx, :].reshape(time_data.shape[0],
                                                                      (start_idx - stop_idx) * order).get(),
                       shading='gouraud', vmin=cb[0], vmax=cb[-1], rasterized=True)
        plt.xlabel(r'Time delay $\omega_p\tau$'), plt.ylabel(r'Velocity $v/v_t$')
        # plt.title(r'Autocorrelation '
        #           r'$\langle f_1(x,v,t)f_1(x-v_g\tau,v,t-\tau)\rangle_L$ at $t=${:0.0f}'.format(time_data[t_idx]))
        plt.colorbar(), plt.tight_layout()

        plt.figure()
        print('Velocity is {:0.3e}'.format(self.grid.v.arr[17, 2]))
        plt.plot(time_delay.get(), autocorr[:, 17, 2].get(), linewidth=3)
        plt.grid(True), plt.xlabel(r'Time delay $\omega_p\tau$'), plt.ylabel(r'Autocorrelation at $v=4.2 v_t$')
        plt.tight_layout()

        plt.show()

    def distribution_contourf(self, distribution, plot_spectrum=True, remove_average=False, max_cb=None, save=None):
        # distribution.average_on_boundaries()
        if distribution.arr_nodal is None:
            distribution.inverse_fourier_transform()
        if distribution.arr is None:
            distribution.fourier_transform()
        if remove_average:
            distribution.arr[0, :] = 0
            distribution.inverse_fourier_transform()

        cb = np.linspace(np.amin(distribution.arr_nodal.get()),
                         np.amax(distribution.arr_nodal.get()), num=100)  # * 0.1
        # cb = np.linspace(-1, 1, num=100)
        if remove_average:
            cb = cb * 0.05
        if max_cb:
            cb = cb * max_cb / np.amax(cb)

        # limits
        # xlim0, xlim1 = -310, -210
        # arg0, arg1 = np.where(self.x == xlim0)[0][0], np.where(self.x == xlim1)[0][0]
        # cb = np.linspace(np.amin(distribution.arr_nodal[arg0:arg1, :, :].get()),
        #                  np.amax(distribution.arr_nodal[arg0:arg1, :, :].get()), num=200)
        # cb = np.linspace(-10e-4, 10e-4, num=100)

        plt.figure(figsize=(6*1.618, 6))
        cf = plt.contourf(self.X, self.V, distribution.grid_flatten().get(), cb, extend='both')  # cmap=self.colormap,
        # plt.pcolormesh(self.X, self.V, distribution.grid_flatten().get(),
        #                shading='gouraud', vmin=cb[0], vmax=cb[-1], rasterized=True)
        plt.xlabel(r'Position $x/\lambda_D$', ), plt.ylabel(r'Velocity $v/v_t$')
        plt.colorbar(cf, format='%.0e')
        # plt.colorbar()
        # plt.xlim([xlim0, xlim1]), plt.ylim([-4, 6])
        plt.tight_layout()
        if save:
            plt.savefig(save + '.jpg', dpi=1000)

        if plot_spectrum:
            spectrum_to_plot = distribution.spectral_flatten()
            # spectrum_to_plot[self.grid.x.zero_idx, :] = 0.0
            spectrum = np.absolute(spectrum_to_plot.get())  # np.log(1.0 + np.absolute(spectrum_to_plot.get()))
            cb_s = np.linspace(np.amin(spectrum), np.amax(spectrum), num=100)
            # cb_s = np.linspace(0, 3e-4, num=100)

            plt.figure(figsize=(3*1.618, 3))
            # plt.contourf(self.FX, self.FV, spectrum, cb_s, extend='both')  # , cmap=self.colormap)
            plt.pcolormesh(self.FX, self.FV, spectrum, shading='gouraud', vmin=cb_s[0], vmax=cb_s[-1], rasterized=True)
            plt.xlabel(r'Wavenumber $k\lambda_D$'), plt.ylabel(r'Velocity $v/v_t$')
            plt.ylim([-5, 5])  # plt.xlim([0, 3.5])
            plt.colorbar(), plt.tight_layout()

    def plot_average_distribution(self, distribution):
        plt.figure()
        plt.plot(self.v, distribution.avg_dist.flatten(), 'o--')
        plt.xlabel('Velocity'), plt.ylabel('Average distribution')
        plt.grid(True), plt.tight_layout()

    def velocity_scalar_plot(self, scalar):
        plt.figure()
        plt.plot(self.v, scalar.arr.flatten().get(), 'o--')
        plt.xlabel('Velocity'), plt.ylabel(r'Distribution $\langle f\rangle_L$')
        plt.grid(True), plt.tight_layout()

    def velocity_scalar_plot_difference(self, scalar1, scalar2, save=False):
        difference = scalar1.arr.flatten().get() - scalar2.arr.flatten().get()
        plt.figure(figsize=(3*1.618, 3))
        plt.plot(self.v, difference)
        plt.xlabel('Velocity'), plt.ylabel(r'Difference from IC, $\langle f\rangle_L(t) - \langle f\rangle_L(0)$')
        plt.grid(True)
        plt.xlim([-15, 15])
        plt.tight_layout()
        if save:
            plt.savefig(save + '.png')

    def plot_many_velocity_averages(self, times, avg_dists, y_label):
        plt.figure()
        for idx in range(avg_dists.shape[0]):
            if np.amax(avg_dists[idx, :, :]) == 0:
                continue
            if idx == 0:
                continue
            plt.plot(self.grid.v.arr[:, 1:-1].flatten(), avg_dists[idx, :, 1:-1].flatten(),
                     linewidth=3, label='t = {:0.0f}'.format(times[idx]))
        plt.xlabel(r'Velocity $v/v_t$'), plt.ylabel(y_label)  # 'Average distribution')
        plt.legend(loc='best')
        plt.grid(True), plt.tight_layout()

    def plot_many_field_power_spectra(self, times, field_psd):
        plt.figure()
        for idx in range(field_psd.shape[0]):
            if idx == 0:
                continue
            if np.amax(field_psd[idx, :]) == 0:
                continue
            plt.plot(self.fundamental * self.k.flatten(), field_psd[idx, :], 'o',
                     label='t={:0.0f}'.format(times[idx]))
        plt.xlabel(r'Wavenumber $k\lambda_D$'), plt.ylabel('Energy spectrum $|E_k|^2$')
        plt.legend(loc='best'), plt.grid(True), plt.tight_layout()

    def plot_average_field_power_spectrum(self, times, field_psd, start_idx):
        average = 0 * field_psd[0, :]
        for idx in range(start_idx, field_psd.shape[0]):
            average += field_psd[idx, :]
        average = average / (field_psd.shape[0] - start_idx)
        plt.figure(figsize=(3*1.618, 3))
        plt.loglog(self.fundamental * self.k.flatten(), average, 'o', ms=1.111)
        plt.xlabel(r'Wavenumber $k\lambda_D$'), plt.ylabel('Field Power Spectral Density')
        plt.legend(loc='best'), plt.grid(True), plt.tight_layout()

    def compute_field_autocorrelation(self, elliptic1, elliptic2, time_delay):
        # Compute hilbert transforms
        hilbert1 = np.abs(elliptic1.field.compute_hilbert(vt_shift=0, grid=self.grid))
        hilbert2 = np.abs(elliptic2.field.compute_hilbert(vt_shift=-1.2 * time_delay, grid=self.grid))
        # group velocity in (0.8 - 1.2)
        # Integrate across x
        mixed_arr = hilbert1 * hilbert2
        return cp.sum(mixed_arr[:-1] + mixed_arr[1:], axis=0) * (self.x[1] - self.x[0]) / 2.0 / self.length

    def wavepacket_autocorrelation(self, field_data, time_data, elements):
        t_idx = 0
        autocorr_0 = cp.zeros((time_data.shape[0]))
        autocorr_s = cp.zeros((time_data.shape[0]))
        elliptic1 = ell.Elliptic(resolution=elements[0])
        elliptic1.field.arr_nodal = cp.asarray(field_data[t_idx])
        elliptic1.field.fourier_transform()
        time_delay = cp.zeros((time_data.shape[0]))

        for idx, tau in enumerate(time_data):
            elliptic2 = ell.Elliptic(resolution=elements[0])
            elliptic2.field.arr_nodal = cp.asarray(field_data[idx])
            elliptic2.field.fourier_transform()

            autocorr_0[idx] = self.compute_field_autocorrelation(elliptic1=elliptic1,
                                                                 elliptic2=elliptic2, time_delay=0)
            time_delay[idx] = tau - time_data[t_idx]
            autocorr_s[idx] = self.compute_field_autocorrelation(elliptic1=elliptic1,
                                                                 elliptic2=elliptic2, time_delay=time_delay[idx])

        plt.figure()
        plt.plot(time_delay.get(), autocorr_0.get(), 'o--', label='Unshifted autocorrelation')
        plt.plot(time_delay.get(), autocorr_s.get(), 'o--', label='Group-velocity-shifted autocorrelation')
        plt.xlabel(r'Time delay $\omega_p\tau$'), plt.ylabel(r'Wavepacket autocorrelation function')
        plt.ylim([0, 1.1 * np.amax(autocorr_0.get())])
        plt.grid(True), plt.legend(loc='best'), plt.tight_layout(), plt.show()

    def spatial_scalar_plot(self, scalar, y_axis, spectrum=True, quadratic=False, title=None, save=False):
        if scalar.arr_nodal is None:
            scalar.inverse_fourier_transform()

        # hilbert = np.abs(scalar.compute_hilbert(vt_shift=0, grid=self.grid))

        plt.figure()
        plt.plot(self.x.flatten(), scalar.arr_nodal.flatten().get(), 'o--')
        # plt.plot(self.x, hilbert, 'o--')
        plt.xlabel('x'), plt.ylabel(y_axis)
        # plt.xlim([0, 500]), plt.ylim([-0.25, 0.25])
        if title is not None:
            plt.title('Time is ' + title)
        plt.grid(True), plt.tight_layout()

        # wig, freq = scalar.compute_wigner_distribution(grid=self.grid)
        #
        # idx = [0, -1]
        # idxk = [2500, 3000]
        # X, K = np.meshgrid(self.x[idx[0]:idx[1]], freq[idxk[0]:idxk[1]].get() / 2, indexing='ij')
        #
        # spectrogram = np.real(wig[idx[0]:idx[1], idxk[0]:idxk[1]].get())  # np.log(1 + np.abs(wig.get()))
        # cb = np.linspace(np.amin(spectrogram), np.amax(spectrogram), num=100)
        # plt.figure()
        # plt.contourf(X, K, spectrogram, cb)
        # plt.xlabel(r'Position $x/\lambda_D$'), plt.ylabel(r'Wavenumber $k\lambda_D$')
        # plt.colorbar(), plt.tight_layout()
        # plt.show()

        # if save:
        #     plt.savefig(save + '.png')

        if spectrum:
            plt.figure(figsize=(3*1.618, 3))
            spectral_arr = scalar.arr_spectral.flatten().get()
            if not quadratic:
                plt.plot(self.k.flatten(), np.real(spectral_arr), 'ro', label='real')
                plt.plot(self.k.flatten(), np.imag(spectral_arr), 'go', label='imaginary')
                plt.legend(loc='best')
            if quadratic:
                plt.loglog(self.fundamental * self.k.flatten(), np.absolute(spectral_arr) ** 2.0, 'o', markersize=1.0)
            plt.xlabel(r'Wavenumber $k\lambda_D$'), plt.ylabel(y_axis + ' spectrum')
            plt.ylim([1.0e-42, 1.0e-5])  # plt.xlim([0.045, 6.1]),
            plt.grid(True), plt.tight_layout()
            if save:
                plt.savefig(save + '.png')

    def time_series_plot(self, time_in, series_in, y_axis, log=False, give_rate=False, numpy=False):
        if not numpy:
            time, series = time_in, series_in.get() / self.length
        else:
            time, series = time_in, series_in / self.length
        plt.figure()
        if log:
            plt.semilogy(time, series, linewidth=3)
        else:
            plt.plot(time, series, linewidth=3)
        plt.xlabel('Time')
        plt.ylabel(y_axis)
        plt.grid(True), plt.tight_layout()
        if give_rate:
            lin_fit = np.polyfit(time, np.log(series), 1)
            print('Numerical rate: {:0.10e}'.format(lin_fit[0]))
            print('cf. exact rate: {:0.10e}'.format(2 * 2.409497728e-01))
            print('The difference is {:0.10e}'.format(lin_fit[0] - 2 * 2.409497728e-01))

    def animate_line_plot(self, saved_array):
        fig, ax = plt.subplots()
        # ax.plot(self.x, saved_array[0, :], 'o--')
        # plt.show()
        plot, = ax.plot(self.x.flatten(), saved_array[0, :], 'o--')
        plt.grid(True)
        ax.set_ylim([0.995, 1.005])

        def animate_frame(idx):
            # clear existing contours
            # ax.collections = []
            # ax.patches = []
            # ax.clear()
            plot.set_data(self.x, saved_array[idx, :])

            # plot line-plot
            # ax.plot(self.x.flatten(), saved_array[idx, :], 'o--')

        anim = animation.FuncAnimation(fig, animate_frame, frames=len(saved_array))
        anim.save(filename='animation.mp4')

    def show(self):
        plt.show()
