import numpy as np
import grid as g
import variables as var
import elliptic as ell
import plotter as my_plt
import time as timer
import timestep as ts
# import data

# Geometry and grid parameters
elements, order = [2000, 80], 10  # 80, 10  # 1400
vt = 1
chi = 0.05
vb = 5
vtb = chi ** (1 / 3) * vb

# Grids
length = 5000  # 10000 # 2.0 * np.pi / 0.126  # 500  # 5000  # 1000
lows = np.array([-length / 2, -25 * vt])
highs = np.array([length / 2, 25 * vt])
grid = g.PhaseSpace(lows=lows, highs=highs, elements=elements, order=order)

# Build distribution
mean_distribution = var.Scalar(resolution=elements[1], order=order)
mean_distribution.initialize_bump_on_tail(grid=grid, vt=vt, u=0, chi=chi, vb=vb, vtb=vtb)
# mean_distribution.initialize_maxwellian(grid=grid, vt=vt)

initial_mean = var.Scalar(resolution=elements[1], order=order)
initial_mean.initialize_maxwellian(grid=grid, vt=vt)

fluctuating_distribution = var.Distribution(resolutions=elements, order=order, charge_mass=-1.0)
fluctuating_distribution.set_up_higher_quad(grid=grid)
fluctuating_distribution.initialize_bump_on_tail(grid=grid, vt=vt, u=0, chi=chi, vb=vb, vtb=vtb)
# fluctuating_distribution.initialize_maxwellian(grid=grid, vt=vt)
fluctuating_distribution.fourier_transform(), fluctuating_distribution.inverse_fourier_transform()

# Set up elliptic problem
Elliptic = ell.Elliptic(resolution=elements[0])
Elliptic.poisson_solve_single_species(distribution=fluctuating_distribution, grid=grid)

# Examine initial condition
Plotter = my_plt.Plotter(grid=grid)
Plotter.distribution_contourf(distribution=fluctuating_distribution, plot_spectrum=True, remove_average=False)
Plotter.spatial_scalar_plot(scalar=Elliptic.field, y_axis='Electric field', quadratic=True)
Plotter.velocity_scalar_plot(scalar=mean_distribution)
Plotter.show()

# Set up time integrator
# Time integration class and stepping information
t0 = timer.time()
time = 0
dt = 2.0e-3  # 4.7e-4
step = 2.0e-3  # 4.7e-4
final_time = 250.0e0  # 31  # 101  # 172  # 151  # 100  # 100  # 150  # 50
steps = int(np.abs(final_time // step))
dt_max_translate = 1.0 / (np.amax(grid.x.wavenumbers) * np.amax(grid.v.arr)) / (2 * order + 1)
cutoff_velocity = 1.0 / (np.amax(grid.x.wavenumbers) * dt) / (2 * order + 1)
print('Max dt translation is {:0.3e}'.format(dt_max_translate))
print('Cutoff velocity at max wavenumber is {:0.3e}'.format(cutoff_velocity))

# Set up stepper and execute main loop
Stepper = ts.StepperSingleSpecies(dt=dt, step=step, resolutions=elements, order=order,
                                  steps=steps, grid=grid)  #
Stepper.main_loop_adams_bashforth(mean_distribution=mean_distribution,
                                  fluctuating_distribution=fluctuating_distribution,
                                  elliptic=Elliptic, grid=grid, plotter=Plotter, mean_ic=initial_mean)
Elliptic.field.inverse_fourier_transform()
print('Done, it took {:0.3e}'.format(timer.time() - t0))

# Final visualize
fluctuating_distribution.inverse_fourier_transform()
Plotter.distribution_contourf(distribution=fluctuating_distribution, plot_spectrum=True, remove_average=False)
Plotter.spatial_scalar_plot(scalar=Elliptic.field, y_axis='Electric field', quadratic=True)
Plotter.velocity_scalar_plot(scalar=mean_distribution)

numpy_or_no = False
Plotter.time_series_plot(time_in=Stepper.time_array, series_in=Stepper.field_energy,
                         y_axis='Electric energy', log=True, give_rate=False, numpy=numpy_or_no)
Plotter.time_series_plot(time_in=Stepper.time_array, series_in=Stepper.thermal_energy-Stepper.thermal_energy[0],
                         y_axis='Kinetic energy electrons', log=False, numpy=numpy_or_no)
Plotter.time_series_plot(time_in=Stepper.time_array, series_in=Stepper.density_array,
                         y_axis='Total density electrons', log=False, numpy=numpy_or_no)
# Plotter.spatial_scalar_plot(scalar=Elliptic.field, y_axis='Field power spectral density', quadratic=True)
# plotter.animate_line_plot(saved_array=stepper.saved_density)
total_energy = Stepper.field_energy + Stepper.thermal_energy
Plotter.time_series_plot(time_in=Stepper.time_array, series_in=total_energy,
                         y_axis='Total energy', log=False, numpy=numpy_or_no)

Plotter.show()
