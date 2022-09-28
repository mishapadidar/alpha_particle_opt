from flux_integrator import FluxIntegrator

# plasma and bfield
vmec_input="../vmec_input_files/input.new_QA_scaling"
bs_path="../field/bs.new_QA_scaling"
# surface discretization
nphi = ntheta = 32
nvpar = 32
# classifier
nphi_classifier = ntheta_classifier = 512
eps_classifier = 0 
# tracing parameters
tmax = 1e-6
dt = 1e-8
ode_method = 'midpoint' # euler or midpoint

fint = FluxIntegrator(vmec_input=vmec_input,
                bs_path=bs_path,
                nphi=nphi,
                ntheta=ntheta,
                nvpar=nvpar,
                nphi_classifier=nphi_classifier,
                ntheta_classifier=ntheta_classifier,
                eps_classifier=eps_classifier,
                tmax=tmax,
                dt=dt,
                ode_method=ode_method
               )

loss_fraction = fint.solve()
print("")
print('loss fraction',loss_fraction)
