# Learning emerges from pattern-induced circuit organization in deep neural networks
Deep neural networks have surpassed human-level performance across a wide range of complex tasks, yet the fundamental principles governing their learning dynamics remain poorly understood. Here, we introduce a dual theoretical framework that addresses both the dynamical and thermodynamic aspects of learning. From a dynamical perspective, a statistical inhomogeneous field theory classifies networks as random, critical, or quasi-critical according to the eigenspectrum of their connectivity tensor. Learning emerges through data pattern-induced circuit organization, culminating in a second-order topological phase transition. The converged state is necessarily quasi-critical to optimize collective computational capacity. From a geometric perspective, a random landscape theory posits flat, self-averaging basins that underpin generalization. We further postulate a form of free energy as the thermodynamic potential to quantify the emergent behavior of learning, enabling the identification of a convergence phase transition characterized by a phase boundary in hyperparameter space and driven by nonequilibrium probability flux in connectivity space. Numerical simulations corroborate these theoretical predictions and reveal striking parallels with brain dynamics. Together, these findings suggest a universal learning principle shared by both artificial and biological neural networks.

# Usage of the codes
## Recording of the training path
The script is contained in the file 'record_path.py' which allows you to modify the initialization methods (among Xavier, kaiming and common uniform), the minibatch size, the learning rate, the total iterations, and the neurons for a 3-layer MLP. 

Note that a learning rate larger than 1.0 does not converge, and training 10 x 10 samples in the hyperparameter space can take about 320 hours. 

You can also modify the 'test_point_indices' which indicates the sequence of steps recorded during the training. 

Other codes are pretty much self-explanatory.

## Analysis neural activity and distribution of loss landscapes
All the codes for analysis are contained in the single file 'analyzer.py' which is a giant class allowing convenient sampling of neural activities and training losses. 

Functions:

- 'set_hyperparam': set the hyperparameters, initialization, learning rate, minibatch size and number of neurons per-layer.
- 'init_dataset': initializes dataset for convenient sampling of MNIST data.
- 'init_model': prepare model for fast access.
- 'update_model': reload model weights to analysis neural activities in other training steps.
- 'get_activity': get neural activities using hooks, for the entire dataset (since it is small).
- 'get_spks': collect output activities after activation function, zscore them, and store results properly.
- 'get_activity_measures_minibatch': records the distribution of topology properties over minibatches, including connectivity_prob, number of edges, cluster size, clique size, and clique values which are the total activity for nodes in a clique.
- 'get_activity_key_measures': calculates topological properties for the neural correlation graph, which is obtained by threshold the neural Pearson correlation matrix, including largest component fraction, susceptibility, global efficiency, small worldness, clustering coefficient, heterogeneity, spectral radius, algebraic connectivity, kcore, SIS indicator, Kuramoto Kc, diffusion mixing time, branching ratio, etc.
- 'get_activity_measures': also calculates topological properties, besides the key measures, also including cumulative explained variance, principle direction indices, SVD transition matrix, normalized eigenvalues, and fitting of singular values to the power-law spectrum.
- 'get_grad_cov_parallel': computes the covariance of loss gradients using parallel distributed methods, where we have used vmap on jacrev for gathering the loss gradients, and compute the full covariance matrix using chunks on a single GPU. The outputs are the average loss gradients 'gm', the outer product of averaged gradients 'sm' and the covariance of gradients 'cov'.
- 'get_hessian_block': calculates the full hessian matrix by dividing the matrix into small blocks, using the hessian-vector product for fast computation, the accuracy is checked and ensured by an independent testing script 'verify_hessian.py'.
- 'get_critical_connections': obtains critical connections for the function of the neural networks, these connections mostly correspond to the active mode of the network, forming a circuit for prediction.
- 'get_landscape': get the loss landscape in different directions.
- 'get_per_data_loss': identifies the distribution of loss landscapes for some training steps.
- 'get_entropies': computes the quasi-thermal entropy (defined over large deviation) by the distribution of loss values.
- 'get_dS_energy': computes the lagrangian (free energy) of entropy and energy (mean loss), as defined in the paper.
- 'get_ccg_performance': computes the performance of the network for three circumstances: all neurons, only critical neurons in the circuit, only non-critical neurons in the circuit.
- 'get_loss_flux': computes the marginal flux of loss as defined in the paper for identifying the transition point of training phases.
- 'get_weight_flux': computes the marginal flux of weight.
- 'weight_evolution_sim_single': simulates the self-reinforced redistribution model.
- 'get_ccg_coupling_energy_evolve': computes the effective circuit potential.

## Visualization of the results
Plotting of the analysis results can be done using the 'NNVisualizer' class in file 'visualizer.py'. The functions and codes are pretty much self-explanatory. 

An example script of usage can be found in the file 'test_anchor.py'.
