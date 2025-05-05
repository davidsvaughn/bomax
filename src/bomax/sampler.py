import warnings
# For the FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
# For the UserWarnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*torch.cuda.*DtypeTensor.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*torch.sparse.SparseTensor.*")

import numpy as np
from matplotlib import pyplot as plt
import torch
import gpytorch
import logging
import math
import time
from scipy.stats import gaussian_kde
from functools import partial

from botorch.models import MultiTaskGP
from gpytorch.priors import LKJCovariancePrior, SmoothedBoxPrior, NormalPrior
from botorch.fit import fit_gpytorch_mll_torch

# import linear_operator.settings as linop_settings
# linop_settings._fast_log_prob._default = True
# linop_settings._fast_solves._default = True

from .utils import adict, display_fig, to_numpy, log_h, clear_cuda_tensors
from .stopping import StoppingCondition, StoppingConditions
from .normalize import Transform
from .degree import degree_metric

torch.set_default_dtype(torch.float64)

#----------------------------------------------------------------------

"""
Bayesian Optimization with Multi-Task Gaussian Processes

see: https://botorch.readthedocs.io/en/latest/models.html#botorch.models.multitask.MultiTaskGP

"""

class MultiTaskSampler:
    def __init__(self,
                 X_feats,
                 Y_obs, *,
                 Y_test=None,
                 max_iterations=1000, # max iterations for MLE fit
                 lr=0.1, # learning rate for MLE fit
                 lr_gamma=0.98, # learning rate decay for scheduler
                 max_sample=0.2, # fraction of total points to sample
                 rank_fraction=0.5,
                 eta=0.25,
                 eta_gamma=0.99,
                 ei_beta=0.5,
                 ei_gamma=0.9925,
                 max_retries=10,
                 verbosity=1,
                 use_cuda=False,
                 run_dir=None,
                 ): 
        
        #--------------------------------------------------------------------------
        self.X_feats = X_feats # 1D : original input feature space
        self.Y_obs = Y_obs # 2D : observed y values (np.nan everywhere else)
        self.Y_test = Y_test # 2D : ALL y values (optional)
        
        # normalize X_feats to unit interval [0..1]
        self.X_test, self.X_norm = Transform.normalize(X_feats)
        
        # create X_inputs (d x 2) that includes task indices
        all_idx = np.where(np.ones_like(Y_obs))
        self.X_inputs = torch.tensor([ [self.X_test[i], j] for i, j in  zip(*all_idx) ], dtype=torch.float64)
        
        # create X_train (d x 2) and Y_train (d x 1) from observed data
        sample_idx = np.where(self.S)
        self.X_train = torch.tensor([ [self.X_test[i], j] for i, j in  zip(*sample_idx) ], dtype=torch.float64)
        self.Y_train = torch.tensor( Y_obs[sample_idx], dtype=torch.float64 ).unsqueeze(-1)

        #--------------------------------------------------------------------------
        self.lr = lr
        self.lr_gamma = lr_gamma
        self.max_iterations = max_iterations
        self.max_sample = max_sample
        self.rank_fraction = rank_fraction
        self.eta = eta
        self.eta_gamma = eta_gamma
        self.ei_beta = ei_beta
        self.ei_gamma = ei_gamma
        self.ei_decay = 1.0
        self.max_retries = max_retries
        self.verbosity = verbosity
        self.run_dir = run_dir
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        
        if use_cuda:
            self.log(f'Using CUDA: {torch.cuda.is_available()}')
        else:
            self.log('Using CPU')
        self.log(f'Using device: {self.device}')
    
        self.X_inputs = self.X_inputs.to(self.device)
        self.round = 0
        self.reset()
        
        # get dimensions of Y_obs (n x m): n = number of x_features, m = number of tasks
        self.n, self.m = Y_obs.shape
        
    def log(self, msg, verbosity_level=1):
        if self.verbosity >= verbosity_level:
            self.logger.info(msg)
            
    def reset(self):
        self.num_retries = -1
    
    @property
    def S(self):
        """
        S is a boolean matrix of size (n, m) storing True for 
        sampled (observed) points and False for unobserved points.
        """
        return ~np.isnan(self.Y_obs)
    
    @property
    def sample_fraction(self):
        """
        sample_fraction is the fraction of the total number 
        of data points that have been sampled so far.
        """
        return np.mean(self.S)
    
    # fit model and recompute posterior predictions
    def update(self, use_large=False):
        if use_large:
            self._fit_large()
        else:
            self.fit_loop()
        return self.predict()
    
    # repeatedly attempt to fit model
    def fit_loop(self):
        # increment round
        self.round += 1
        
        # reset next sample indices
        self.next_i, self.next_j = None, None
        
        # clear CUDA tensors
        if self.device.type == 'cuda':
            clear_cuda_tensors(log = partial(self.log, verbosity_level=2))
        
        # run MLE fit loop
        for i in range(self.max_retries):
            if self.fit():
                return True
            else:
                if i+1 < self.max_retries:
                    self.log('-'*110)
                    self.log(f'FAILED... ATTEMPT {i+2}')
        raise Exception('ERROR: Failed to fit model - max_retries reached')
    
    # fit model inner loop
    def fit(self):
        self.num_retries += 1
        x_train = self.X_train
        y_train = self.Y_train
        m = self.S.shape[1]
        
        # standardize y_train
        y_train, self.Y_stand = Transform.standardize(x_train, y_train)
        
        # init retry-adjusted parameters
        rank = int(self.rank_fraction * m) if self.rank_fraction > 0 else None
        eta = self.eta
        
        #---------------------------------------------------------------------
        # if fit failed previously... adjust parameters
        if self.num_retries > 0:
            
            # rank adjustment...
            if self.rank_fraction > 0:
                w = self.num_retries * (m-rank)//self.max_retries
                rank = min(m, rank + w)
                self.log(f'[ROUND-{self.round}]\tFYI: rank adjusted to {rank}', 2)
                
            # eta adjustment... ??
            if eta is not None:
                eta = eta * (self.eta_gamma ** max(0, self.num_retries - self.max_retries//2))
                self.log(f'[ROUND-{self.round}]\tFYI: eta adjusted to {eta:.4g}', 2)
                
        #---------------------------------------------------------------------
        # Initialize multitask model
        
        # define task_covar_prior (IMPORTANT!!! with sparse data, nothing works without this!)
        # see: https://archive.botorch.org/v/0.9.2/api/_modules/botorch/models/multitask.html
        if eta is None:
            task_covar_prior = None
        else:
            task_covar_prior = LKJCovariancePrior(n=m, 
                                                  eta=torch.tensor(eta).to(self.device),
                                                  sd_prior=SmoothedBoxPrior(math.exp(-6), math.exp(1.25), 0.05),
                                                  ).to(self.device)
        
        # define multi-task model
        self.model = MultiTaskGP(x_train, y_train, task_feature=-1, 
                                 rank=rank,
                                 task_covar_prior=task_covar_prior,
                                 outcome_transform=None,
                                 ).to(self.device)

        x_train, y_train = x_train.to(self.device), y_train.to(self.device)
        
        # Set the model and likelihood to training mode
        self.model.train()
        
        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood=self.model.likelihood, model=self.model)
        
        # use botorch optimization with Adam
        fit_gpytorch_mll_torch(mll, 
                                optimizer=partial(torch.optim.Adam, lr=self.lr),
                                scheduler=partial(torch.optim.lr_scheduler.StepLR, step_size=10, gamma=self.lr_gamma),
                                step_limit=self.max_iterations)
        self.reset()
        return True
    #---------------------------------------------------------------------
    
    # experimental -- manually fit model when number of observations is large 
    def _fit_large(self):
        self.num_retries += 1
        x_train = self.X_train
        y_train = self.Y_train
        m = self.S.shape[1]
        
        # standardize y_train
        y_train, self.Y_stand = Transform.standardize(x_train, y_train)
        
        # compute rank
        rank = int(self.rank_fraction * m) if self.rank_fraction > 0 else None
        
        # init retry-adjusted parameters
        eta = self.eta
        patience = 5
        min_iterations = 100
        loss_thresh=0.00025
        degree_thresh=6
        degree_stat='max'
        log_interval=2
                
        #---------------------------------------------------------------------
        # define task_covar_prior (IMPORTANT!!! with sparse data, nothing works without this!)
        # see: https://archive.botorch.org/v/0.9.2/api/_modules/botorch/models/multitask.html
        if eta is None:
            task_covar_prior = None
        else:
            task_covar_prior = LKJCovariancePrior(n=m, 
                                                  eta=torch.tensor(eta).to(self.device),
                                                  sd_prior=SmoothedBoxPrior(math.exp(-6), math.exp(1.25), 0.05),
                                                  ).to(self.device)
            
        #---------------------------------------------------------------------
        # Initialize multitask model
        
        self.model = MultiTaskGP(x_train, y_train, task_feature=-1, 
                                 rank=rank,
                                 task_covar_prior=task_covar_prior,
                                 outcome_transform=None,
                                 ).to(self.device)

        x_train, y_train = x_train.to(self.device), y_train.to(self.device)
        
        # Set the model and likelihood to training mode
        self.model.train()
        
        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood=self.model.likelihood, model=self.model)
        
        # Use the adam optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        #---------------------------------------------------------
        # Stopping Criteria...
        cond_list = []
        
        # relative change stopping criterion
        if loss_thresh is not None:
            loss_condition = StoppingCondition(
                value="loss",
                condition="0 < (x[-2] - x[-1])/abs(x[-2]) < t",
                t=loss_thresh,
                alpha=5,
                min_iterations=min_iterations,
                patience=patience,
                lr_steps=5, # learning rate lowered 5 times before termination
                lr_gamma=0.8,
                optimizer=optimizer,
                verbosity=self.verbosity,
            )
            cond_list.append(loss_condition)
        
        # max-degree stopping criterion ( if there has been at least one previous failure )
        if degree_thresh is not None:
        
            # function closure for degree metric
            def degree_func(**kwargs):
                return degree_metric(model=self.model, X_inputs=self.X_inputs, m=self.m, num_trials=100)[degree_stat]
            
            degree_condition = StoppingCondition(
                value=degree_func,
                condition="x[-1] > t",
                t=degree_thresh,
                interval=5, # check every 5 iterations
                min_iterations=min_iterations,
                verbosity=self.verbosity,
            )
            cond_list.append(degree_condition)
        
        #------------------------------------
        # combine stopping conditions (at least one must be satisfied)
        stop_conditions = StoppingConditions(cond_list, mode='any') # mode = 'any' or 'all'
        
        #---------------------------------------------------------
        
        report = {}
        start_time = time.time()
        
        # train loop
        for i in range(self.max_iterations):
            optimizer.zero_grad()
            output = self.model(x_train)
            
            try:
                loss = -mll(output, self.model.train_targets)
            except Exception as e:
                if self.num_retries+1 == self.max_retries:
                    self.logger.exception("An exception occurred")
                else:
                    self.logger.error(f'Error in loss calculation at iteration {i}:\n{e}')
                return False
            
            loss.backward()
            
            if stop_conditions.check(loss=loss.item(), report=report):
                break
            
            optimizer.step()
            
            if i % log_interval == 0:
                iter_per_sec = log_interval/(time.time() - start_time)
                start_time = time.time()
                self.log(f'[ROUND-{self.round}]\tITER-{i}/{self.max_iterations}\t{iter_per_sec:.2g} it/s\t' + '\t'.join([f'{k}: {v:.4g}' for k,v in report.items()]))  
                     
        #---- end train loop --------------------------------------------------
        
        if stop_conditions.last:
            # fit has succeeded
            self.reset()
            return True
        
        # max iterations reached without convergence
        return False
    #--------------------------------------------------------------------------
            
    def predict(self, x=None):
        n,m = self.n, self.m
        if x is None:
            x = self.X_inputs
        self.model.eval()
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = self.model.likelihood(self.model(x))
        
        # retrieve estimates for each (input, task) pair
        y_means = predictions.mean
        y_vars = predictions.variance
        y_covar = predictions.covariance_matrix
        
        # reshape
        y_means = to_numpy(y_means.reshape(n, m))
        y_vars = to_numpy(y_vars.reshape(n, m))
        y_covar = to_numpy(y_covar.reshape((n, m, n, m)))
        
        # inverse standardize...
        y_means = self.Y_stand.inv(y_means)
        sigma = self.Y_stand.params['sigma']
        y_sigmas = np.sqrt(y_vars) * sigma
        
        # average across tasks
        y_mean = y_means.mean(axis=1)
        y_covar = y_covar * sigma**2
        y_sigma = np.array([np.sqrt(y_covar[i,:,i].sum()) for i in range(n)]) / m
        
        # current max estimate
        self.current_best_idx = i = np.argmax(y_mean)
        self.current_best_checkpoint = self.X_feats[i]
        # current_y_mean = y_mean[i] # current y_mean estimate at peak_idx of y_mean estimate
        
        #-------------------------------------------------
        # fetch inter-task correlation estimates from task kernel Kt
        # with torch.no_grad():
        #     task_covar = self.model.task_covar_module.covar_matrix.to_dense()
        # # Compute correlations from covariances
        # diag_std = task_covar.diag().sqrt()
        # self.task_corr = (task_covar / (diag_std.unsqueeze(-1) * diag_std.unsqueeze(0))).cpu().numpy()
        #-------------------------------------------------
        
        self.y_means = y_means
        self.y_sigmas = y_sigmas
        self.y_mean = y_mean
        self.y_sigma = y_sigma
        return y_mean, y_sigma, y_means, y_sigmas
    
    def get_r2(self, Y_ref, plot=True):
        Y_ref_corr = np.corrcoef(Y_ref.T)
        Y_ref_corr = np.tril(Y_ref_corr, k=-1).flatten()
        Y_est_corr = np.corrcoef(self.y_means.T)
        Y_est_corr = np.tril(Y_est_corr, k=-1).flatten()
        r2 = np.corrcoef(Y_ref_corr, Y_est_corr)[0, 1]**2
        
        # make scatterplot of Y_ref_corr vs Y_est_corr
        if plot:
            plt.figure(figsize=(10, 10))
            plt.scatter(Y_ref_corr, Y_est_corr, alpha=0.5)
            plt.xlabel('Reference Correlation')
            plt.ylabel('Estimated Correlation')
            # plt.title(f'Round {self.round}  R^2 {r2:.4f}')
            # make R superscript 2 in title
            plt.title(f'Round {self.round} : R$^2$={r2:.4f}')
            plt.plot([-1, 1], [-1, 1], 'r--')
            plt.xlim(-1, 1)
            plt.ylim(-1, 1)
            plt.grid()
            plt.gca().set_aspect('equal', adjustable='box')
            self.display(prefix='Corr')
        
        return r2
    
    # ---------------------------------------------------------------------
    # compare current model to reference and gold standard
    def compare(self, Y_ref, Y_gold=None):
        # get R^2 for reference
        Tr2 = self.get_r2(Y_ref)
        
        # get reference mean across tasks
        Y_ref_mean = Y_ref.mean(axis=1)
        i = np.argmax(Y_ref_mean)
        y_ref_max = Y_ref_mean[i]
        # ref_best_checkpoint = self.X_feats[i]
        
        current_y_val = Y_ref_mean[self.current_best_idx]
        self.current_err = err = abs(current_y_val - y_ref_max)/y_ref_max
        
        # self.log('-'*110)
        # if Y_gold is not None:
        #     # get R^2 for gold standard
        #     Gr2 = self.get_r2(Y_gold, plot=False)
        #     self.log(f'[ROUND-{self.round}]\tSTATS\t{self.round}\t{self.current_best_checkpoint}\t{Tr2:.4g}\t{Gr2:.4g}\t{err:.4g}\t{self.sample_fraction:.4g}')
        # else:
        #     self.log(f'[ROUND-{self.round}]\tSTATS\t{self.round}\t{self.current_best_checkpoint}\t{Tr2:.4g}\t{err:.4g}\t{self.sample_fraction:.4g}')
        
        self.log(f'[ROUND-{self.round}]\tCURRENT BEST:\tCHECKPOINT-{self.current_best_checkpoint}\tR^2={Tr2:.4f}\tY_PRED={current_y_val:.4f}\tY_ERR={100*err:.4g}%\t({100*self.sample_fraction:.2f}% sampled)')
        
    #-------------------------------------------------------------------------
    
    # dampen the acquisition function in highly sampled regions
    def sample_damper(self, decay=1.0, bw_mult=25.0):
        # bandwidth for kde smoother
        bw = bw_mult * decay / self.n
        # x-axis possible sample locations
        X = self.X_test
        # matrix to hold kde estimates
        K = np.ones_like(self.S, dtype=np.float64) * np.nan
        
        # for each task compute kde
        for j in range(K.shape[1]):
            y = self.S[:, j]
            # extract sampled and unsampled points
            x, o = X[y], X[~y]
            # compute kde from sampled points
            kde = gaussian_kde(x, bw_method=bw)
            # evaluate kde at unsampled points
            v = kde(o)
            # normalize kde...
            v = v/np.sum(v)
            # and scale each task by number of sampled points in that task
            v *= len(x)/K.shape[0]
            # assign back to K
            K[~y, j] = v
        return K
    
    # compute expected improvement
    def max_expected_improvement(self, beta=0.5, decay=1.0, debug=True):
        S_mu = self.y_means.sum(axis=-1)
        S_max_idx = np.argmax(S_mu)
        S_max = S_mu[S_max_idx]
        
        # get unsampled indices
        i_indices, j_indices = np.where(np.ones_like(self.S))
        mask = (~self.S).reshape(-1)
        
        # Vectorized computation of all EI components
        valid_i, valid_j = i_indices[mask], j_indices[mask]
            
        # Initialize EI matrix with -inf for invalid entries
        EI = np.full(len(mask), -math.inf)

        # Vectorized computation of all EI components
        # mu = self.y_means[valid_i, valid_j]
        sig = self.y_sigmas[valid_i, valid_j]

        # Get row sums for each valid i
        Sx = np.sum(self.y_means[valid_i, :], axis=1)
        
        # improvement vector, z-scores
        imp = Sx - S_max - beta
        z = imp/sig

        # Expected Improvement (EI) computation
        # if use_logei: # logEI computation (for stability)
        logh = log_h(torch.tensor(z, dtype=torch.float64)).numpy()
        ei_values = np.log(sig) + logh
        # else: # standard EI
        #     ei_values = imp * norm.cdf(z) + sig * norm.pdf(z)
        #     ei_values = np.log(ei_values)
        
        if debug:
            # retain original EI matrix for
            # comparison to dampened EI matrix
            EI0 = EI.copy()
            EI0[mask] = ei_values
            
        # dampen highly sampled regions
        D = self.sample_damper(decay=decay)
        d = D[valid_i, valid_j]
        ei_min = ei_values.min()
        # shift (so non-negative) -> apply dampening -> unshift
        ei_values = (ei_values-ei_min) * (1 - decay * d**0.5) + ei_min

        # Assign computed values to valid sampling positions and return optimum
        EI[mask] = ei_values
        k = np.argmax(EI)
        next_i, next_j = i_indices[k], j_indices[k]
        
        #-----------------------------------------------------------
        # debugging...
        if debug:
            self.plot_task(next_j, '- NO DAMPER', EI0.reshape(self.n, self.m)[:, next_j])
        self.plot_task(next_j, '- BEFORE', EI.reshape(self.n, self.m)[:, next_j])
        #------------------------------------------------------------
        
        # decay EI parameters
        self.ei_decay = self.ei_decay * self.ei_gamma
        self.ei_beta = self.ei_beta * self.ei_gamma
        self.log(f'[ROUND-{self.round}]\tFYI: EI beta: {self.ei_beta:.4g}', 2)
        
        return next_i, next_j
    
    #-------------------------------------------------------------------------
    
    # report task with most samples
    def report_most_sampled_task(self):
        task_counts = np.sum(self.S, axis=0)
        max_task = np.argmax(task_counts)
        max_count = task_counts[max_task]
        self.log(f'[ROUND-{self.round}]\tFYI: TASK-{max_task} has most samples: {max_count}', 1)
    
    # choose next sample
    def get_next_sample_point(self):
        
        # Maximize Expected Improvement acquisition function
        next_i, next_j = self.max_expected_improvement(beta=self.ei_beta, decay=self.ei_decay)
        self.next_i, self.next_j = next_i, next_j
        
        # convert to original X feature space
        next_checkpoint = self.X_feats[next_i]
        self.log(f'[ROUND-{self.round}]\tNEXT SAMPLE:\tCHECKPOINT-{next_checkpoint}\tTASK-{next_j}')
        
        # self.report_most_sampled_task()
        self.log('='*110)
        
        # return next sample point
        return next_i, next_j
    
    # add next sample to training set
    def add_next_sample(self, Y=None):
        
        # check if Y is provided
        if Y is None:
            Y = self.Y_test
        if Y is None:
            raise ValueError('Y is None, no data source for next sample')
        
        if self.next_i is None or self.next_j is None:
            # check if Y is scalar value
            if np.isscalar(Y):
                raise ValueError('Y is a scalar value, but next sample point is not chosen yet')
            self.get_next_sample_point()
            
        # get next sample indices
        next_i, next_j = self.next_i, self.next_j
        
        # acquire next observation
        if np.isscalar(Y):
            y = Y
        elif callable(Y):
            # if y is callable function, then call it with next_i, next_j
            y = Y(next_i, next_j)
        else:
            # else, access Y directly with next_i, next_j
            try:
                y = Y[next_i][next_j]
            except Exception as e:
                raise ValueError(f'Error accessing Y with next_i, next_j: {e}')
            
        # add new sample to training set (observe) and update mask
        self.X_train = torch.cat([self.X_train, torch.tensor([ [self.X_test[next_i], next_j] ], dtype=torch.float64)])
        self.Y_train = torch.cat([self.Y_train, torch.tensor([y], dtype=torch.float64).unsqueeze(-1)])
        self.Y_obs[next_i, next_j] = y
        
        # return next sample point
        return next_i, next_j
    
    #-------------------------------------------------------------------------
    # Plotting functions
    
    def display(self, fig=None, fn=None, prefix='fig'):
        display_fig(self.run_dir, fig=fig, fn=fn, prefix=prefix)
    
    def plot_posterior_mean(self, y_ref=None, y_gold=None, ref_color='g', gold_color='r', prefix='posterior_mean'):
        legend = []
        plt.figure(figsize=(15, 10))
        plt.plot(self.X_feats, self.y_mean, 'b')
        legend.append('GP Posterior Mean')
        plt.fill_between(self.X_feats, self.y_mean - 2*self.y_sigma, self.y_mean + 2*self.y_sigma, alpha=0.5)
        legend.append('GP 2$\sigma$ Confidence')
        plt.axvline(self.current_best_checkpoint, color='b', linestyle='--')
        legend.append('GP Optimum')
        
        if y_ref is None and y_gold is not None:
            y_ref = y_gold
            y_gold = None
            ref_color = gold_color
        
        # compare to reference
        if y_ref is not None:
            i = np.argmax(y_ref)
            ref_best_input = self.X_feats[i]
            plt.plot(self.X_feats, y_ref, ref_color)
            legend.append('Target Mean')
            
            # draw vertical dotted line at best input
            plt.axvline(ref_best_input, color=ref_color, linestyle='--')
            legend.append('Target Optimum')
            
        # compare to gold standard
        if y_gold is not None:
            i = np.argmax(y_gold)
            gold_best_input = self.X_feats[i]
            plt.plot(self.X_feats, y_gold, gold_color)
            legend.append('True Mean')
            
        plt.legend(legend, loc='best')
        
        if y_ref is not None:
            y_min, y_max = plt.ylim()
            padding = (y_max - y_min) * 0.05
            y_min -= padding
            y_max += padding
            plt.ylim(y_min, y_max)
            plt.fill_betweenx([y_min, y_max], self.current_best_checkpoint, ref_best_input, color='red', alpha=0.1)
        
        plt.title(f'Round: {self.round-1} - Estimated Max: {self.current_best_checkpoint}')
        self.display(prefix=prefix)
        
    def plot_task(self, j, msg='', fvals=None, prefix='task'):
        fig, ax1 = plt.subplots(figsize=(15, 10))
        x = self.X_feats
        legend = []
        
        # Plot all data as black stars (optional?)
        if self.Y_test is not None:
            ax1.plot(x, self.Y_test[:, j], 'k*')
            legend.append('Unobserved')
        
        # Plot training (observed) data as red circles
        idx = np.where(to_numpy(self.X_train)[:,1] == j)
        xx = to_numpy(self.X_train[idx][:,0])
        # transform to original feature space
        xx = self.X_norm.inv(xx)
        yy = to_numpy(self.Y_train[idx])
        ax1.plot(xx, yy, 'ro')
        legend.append('Observed')
        
        # Plot predictive means as blue line
        ax1.plot(x, self.y_means[:, j], 'b')
        legend.append('Posterior Mean')
        
        # confidences
        win = 2 * self.y_sigmas[:, j]
        ax1.fill_between(x, self.y_means[:, j] - win, self.y_means[:, j] + win, alpha=0.5)
        legend.append('Confidence')
        
        # Set up primary y-axis labels
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        
        if fvals is not None:
            # Create secondary y-axis for fvals
            ax2 = ax1.twinx()
            
            # Plot fvals as green dashed line on secondary axis
            ax2.plot(x, fvals, 'g--')
            ax2.set_ylabel('log(EI)', color='g')
            ax2.tick_params(axis='y', labelcolor='g')
            legend.append('Expected Improvement')
            fig.legend(legend, loc='upper left')
        else:
            ax1.legend(legend, loc='best')
        
        plt.title(f'Round: {self.round} - Task: {j} {msg}')
        plt.tight_layout()  # Adjust layout to make room for the second y-axis label
        self.display(prefix=prefix)
    
    def plot_all(self, max_fig=None):
        self.plot_posterior_mean()
        for j in range(self.m):
            if max_fig is not None and j > max_fig:
                break
            self.plot_task(j)
