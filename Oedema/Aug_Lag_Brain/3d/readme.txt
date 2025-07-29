This repository includes the software needed to reproduce the research presented in the following publication: https://doi.org/10.1371/journal.pcbi.1012145
Guidance in the book: https://doi.org/10.1007/978-3-540-32609-0
 
Guide for execution

Five compartments, poroelastic cerebral blood, interstitial fluid flow and deformation for oedema and midline shift

1. prepare hemispheric stroke meshes in xml using your perfusion model. 
2. run "fullbrain_serial_opt_cop_conv_time_orig.py" to obtain the pressures at each time steps
3. run "aug_lag_brain_0.py" to start contact mechanics simulation to solve displacement for each time step. 

Please send google drive invitation to jozsait@gmail.com

My mistake, I solved them separately in the last version. We get the pressure first and then use the save pressure to calculate contact. This separates the contact mechanics solver, so I could test its performance. 

