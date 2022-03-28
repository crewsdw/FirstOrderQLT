import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
import scipy.optimize as opt
import tools.plasma_dispersion as pd
# import plasma_dispersion as pd


def dispersion_function(k, z, drift_one, vt_one, two_scale, drift_two, vt_two):
    """
    Computes two-species plasma dispersion function epsilon(zeta, k) = 0
    """
    sq2 = 2 ** 0.5
    k2 = k / sq2
    k_sq = k ** 2.0
    z_e_one = (z - drift_one) / vt_one / sq2
    z_e_two = (z - drift_two) / vt_two / sq2
    return 1 - 0.5 * (pd.Zprime(z_e_one) +
                      two_scale * pd.Zprime(z_e_two) * (vt_one / vt_two) ** 2) / k_sq / (1 + two_scale)  #


def analytic_jacobian(k, z, drift_one, vt_one, two_scale, drift_two, vt_two):
    sq2 = 2 ** 0.5
    k2 = k / sq2
    k_sq = k ** 2.0
    z_e_one = (z - drift_one) / vt_one / sq2
    z_e_two = (z - drift_two) / vt_two / sq2
    return -0.5 * (pd.Zdoubleprime(z_e_one) / vt_one +
                   two_scale * pd.Zdoubleprime(z_e_two) / vt_two) / k_sq / (1 + two_scale)


def dispersion_fsolve(z, k, drift_one, vt_one, two_scale, drift_two, vt_two):
    freq = z[0] + 1j * z[1]
    d = dispersion_function(k, freq, drift_one, vt_one, two_scale, drift_two, vt_two)
    return [np.real(d), np.imag(d)]


def jacobian_fsolve(z, k, drift_one, vt_one, two_scale, drift_two, vt_two):
    freq = z[0] + 1j * z[1]
    jac = analytic_jacobian(k, freq, drift_one, vt_one, two_scale, drift_two, vt_two)
    jr, ji = np.real(jac), np.imag(jac)
    return [[jr, -ji], [ji, jr]]


def dispersion_function_two_species(k, z, mass_ratio, temperature_ratio, electron_drift):
    """
    Computes two-species plasma dispersion function epsilon(zeta, k) = 0
    """
    thermal_velocity_ratio = np.sqrt(temperature_ratio / mass_ratio)
    k_sq = k ** 2.0
    z_e = z - electron_drift / np.sqrt(2)
    z_p = thermal_velocity_ratio * z
    return 1.0 - (pd.Zprime(z_e) + temperature_ratio * pd.Zprime(z_p)) / k_sq


def analytic_jacobian_two_species(k, z, mass_ratio, temperature_ratio, electron_drift):
    thermal_velocity_ratio = np.sqrt(temperature_ratio / mass_ratio)
    k_sq = k ** 2.0
    z_e = z - electron_drift / np.sqrt(2)
    z_p = thermal_velocity_ratio * z
    fe = 1
    fp = thermal_velocity_ratio
    return -0.5 * (pd.Zdoubleprime(z_e) / fe + temperature_ratio * pd.Zdoubleprime(z_p) / fp) / k_sq


def dispersion_fsolve_two_species(z, k, mass_ratio, temperature_ratio, electron_drift):
    freq = z[0] + 1j * z[1]
    d = dispersion_function_two_species(k, freq, mass_ratio, temperature_ratio, electron_drift)
    return [np.real(d), np.imag(d)]


def jacobian_fsolve_two_species(z, k, mass_ratio, temperature_ratio, electron_drift):
    freq = z[0] + 1j * z[1]
    jac = analytic_jacobian(k, freq, mass_ratio, temperature_ratio, electron_drift)
    jr, ji = np.real(jac), np.imag(jac)
    return [[jr, -ji], [ji, jr]]


if __name__ == '__main__':
    # parameters
    mr = 1 / 1836  # / 10  # 1836  # me / mp
    tr = 1.0  # Te / Tp
    k = 0.01

    e_d = 0  # 3.0

    # grid
    om_r = np.linspace(-0.1, 0.1, num=500)
    om_i = np.linspace(-0.1, 0.1, num=500)

    k_scale = k

    zr = om_r / k_scale
    zi = om_i / k_scale

    z = np.tensordot(zr, np.ones_like(zi), axes=0) + 1.0j * np.tensordot(np.ones_like(zr), zi, axes=0)

    X, Y = np.meshgrid(om_r, om_i, indexing='ij')

    # eps = dispersion_function(k, z, mr, tr, e_d)
    chi = 0.05
    vb = 5
    vtb = chi ** (1 / 3) * vb
    eps = dispersion_function(k, z, drift_one=0, vt_one=1, two_scale=chi, drift_two=vb, vt_two=vtb)
    eps = dispersion_function(k, z, drift_one=5, vt_one=1, two_scale=1, drift_two=-5, vt_two=1)
    cb = np.linspace(-1, 1, num=100)

    # plt.figure()
    # plt.contourf(X, Y, np.real(eps), cb, extend='both')

    plt.figure()
    plt.contour(X, Y, np.real(eps), 0, colors='r')
    plt.contour(X, Y, np.imag(eps), 0, colors='g')
    plt.grid(True)
    plt.show()

    guess_r, guess_i = 0.05 / k, -0.005 / k
    solution = opt.root(dispersion_fsolve, x0=np.array([guess_r, guess_i]),
                        args=(k, 0, 1, chi, vb, vtb), jac=jacobian_fsolve)
    print(solution.x * k_scale)

    # Loop over wavenumbers
    k0 = 2.0 * np.pi / 1000
    waves = k0 * np.arange(100)  # np.linspace(k, 0.75, num=300)
    print(waves)
    sols = np.zeros_like(waves) + 0j
    for idx, wave in enumerate(waves):
        solution = opt.root(dispersion_fsolve, x0=np.array([guess_r, guess_i]),
                            args=(wave, 0, 1, chi, vb, vtb), jac=jacobian_fsolve)
        guess_r, guess_i = solution.x
        sols[idx] = (guess_r + 1j * guess_i) * wave * (2 ** 0.5)

    # Lattice frequencies
    # k0 = 2.0 * np.pi / 1000
    # grid_waves = k0 * np.arange(50)

    plt.figure()
    plt.plot(waves, np.real(sols), 'r', label='real')
    plt.plot(waves, np.imag(sols), 'g', label='imag')
    plt.plot(waves, np.zeros_like(waves), 'ko')
    plt.xlabel(r'Wavenumber k'), plt.ylabel(r'Frequency $\omega_p$')
    plt.grid(True), plt.legend(loc='best'), plt.tight_layout()
    plt.show()

    # Visualize the modes
    x = np.linspace(-500, 500, num=2000)
    v = np.linspace(-3, 7, num=1000)
    X, V = np.meshgrid(x, v, indexing='ij')
    df = -V * np.exp(-0.5 * V ** 2.0) - chi * (V - vb) * np.exp(-0.5 * (V - vb) ** 2.0 / vtb ** 2.0) / (vtb ** 3.0)


    def eigenfunction(z, k):
        return df / (V - z) * np.exp(1j * k * X) / (k ** 2.0)


    def cb(f):
        return np.linspace(np.amin(f), np.amax(f), num=100)


    # modes = k0 * np.array([4, 5, 6, 7])
    # eigs = [(1.64+0.067j)/modes[0],
    #         (1.83+0.102j)/modes[1],
    #         (2.04+0.102j)/modes[2],
    #         (2.21+0.06j)/modes[3]]
    pi2 = 2 * np.pi

    unstable_modes = waves[np.imag(sols) > 0.002]
    unstable_eigs = sols[np.imag(sols) > 0.002]
    eig_sum = 0
    for idx in range(unstable_modes.shape[0]):
        eigenvalue = unstable_eigs[idx] / (2 ** 0.5 * unstable_modes[idx])
        eig = np.real(eigenfunction(eigenvalue, unstable_modes[idx]) * np.exp(1j * pi2 * np.random.random(1)))
        plt.figure()
        plt.contourf(X, V, eig, cb(eig))
        plt.xlabel(r'Position $x$'), plt.ylabel(r'Velocity $v$')
        plt.tight_layout(), plt.savefig('eig' + str(idx) + '.png')
        plt.close()
        eig_sum += eig

    # eig0 = np.real(eigenfunction(eigs[0], modes[0]))
    # eig1 = np.real(eigenfunction(eigs[1], modes[1]) * np.exp(1j * pi2 * np.random.random(1)))
    # eig2 = np.real(eigenfunction(eigs[2], modes[2]) * np.exp(1j * pi2 * np.random.random(1)))
    # eig3 = np.real(eigenfunction(eigs[3], modes[3]) * np.exp(1j * pi2 * np.random.random(1)))

    #
    # plt.figure()
    # plt.contourf(X, V, eig0, cb(eig0))
    # plt.xlabel(r'Position $x$'), plt.ylabel(r'Velocity $v$')
    # plt.tight_layout(), plt.savefig('eig0.png')
    #
    #
    # plt.figure()
    # plt.contourf(X, V, eig1, cb(eig1))
    # plt.xlabel(r'Position $x$'), plt.ylabel(r'Velocity $v$')
    # plt.tight_layout(), plt.savefig('eig1.png')
    #
    # plt.figure()
    # plt.contourf(X, V, eig2, cb(eig2))
    # plt.xlabel(r'Position $x$'), plt.ylabel(r'Velocity $v$')
    # plt.tight_layout(), plt.savefig('eig2.png')

    plt.figure()
    plt.contourf(X, V, eig_sum, cb(eig_sum))
    plt.xlabel(r'Position $x$'), plt.ylabel(r'Velocity $v$')
    plt.tight_layout(), plt.savefig('eig_sum.png')

    plt.show()
