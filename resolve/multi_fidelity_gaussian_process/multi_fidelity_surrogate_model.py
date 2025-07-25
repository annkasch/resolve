import numpy as np
np.random.seed(123)
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
import GPy
from emukit.multi_fidelity import kernels
from emukit.model_wrappers.gpy_model_wrappers import GPyMultiOutputWrapper
from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel
from emukit.multi_fidelity.convert_lists_to_array import (
    convert_x_list_to_array,
    convert_xy_lists_to_arrays
)
from emukit.core import ParameterSpace, ContinuousParameter, InformationSourceParameter
from emukit.core.optimization import GradientAcquisitionOptimizer
from emukit.core.acquisition import Acquisition
from emukit.experimental_design.acquisitions import IntegratedVarianceReduction, ModelVariance
from emukit.bayesian_optimization.acquisitions.entropy_search import MultiInformationSourceEntropySearch
from emukit.core.loop.candidate_point_calculators import SequentialPointCalculator
from emukit.core.loop.loop_state import create_loop_state
from emukit.core.optimization.multi_source_acquisition_optimizer import MultiSourceAcquisitionOptimizer
import copy

from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans

# Ensure reproducibility
np.random.seed(123)

class Normalizer:
    def __init__(self):
        self.scaler_x = None
        self.scaler_y = None

    def fit(self, trainings_data, y_scaler=None):
        """
        trainings_data: dict like {0: (X_lf, Y_lf), 1: (X_hf, Y_hf)}
        Fits one global scaler for X and one for Y across all fidelities.
        """
        # Collect all X and Y
        x_list = []
        y_list = []
        for key in trainings_data.keys():
            x, y = trainings_data[key]
            x = np.atleast_2d(x)
            y = np.atleast_2d(y).reshape(-1, 1)  # Ensure (N, 1)
            x_list.append(x)
            y_list.append(y)

        # Stack together and fit scalers
        x_all = np.vstack(x_list)
        y_all = np.vstack(y_list)

        self.scaler_x = StandardScaler().fit(x_all)
        self.scaler_y=StandardScaler()
        if y_scaler == None: self.scaler_y = StandardScaler().fit(y_all)
        else: self.scaler_y.scale_=y_scaler

    def transform_x(self, x):
        x = np.atleast_2d(x)
        return self.scaler_x.transform(x)

    def transform_y(self, y):
        y = np.atleast_2d(y).reshape(-1, 1)
        return self.scaler_y.transform(y)

    def transform_noise(self, noise):
        """Normalize noise (standard deviation) according to y-scaling."""
        sigma_y = self.scaler_y.scale_[0]  # Only one output dimension
        return noise / sigma_y

    def inverse_transform_x(self, x_norm):
        """Undo normalization for predictions (mean values)."""
        return self.scaler_x.inverse_transform(np.atleast_2d(x_norm))
    
    def inverse_transform_y(self, y_norm):
        """Undo normalization for predictions (mean values)."""
        return self.scaler_y.inverse_transform(np.atleast_2d(y_norm))

    def inverse_transform_noise(self, noise_norm):
        """Undo normalization for predictive noise (std)."""
        sigma_y = self.scaler_y.scale_[0]
        return noise_norm * sigma_y

    def inverse_transform_variance(self, var_norm):
        """Undo normalization for predictive variance."""
        sigma_y = self.scaler_y.scale_[0]
        return var_norm * (sigma_y ** 2)

class MFGPModel():
    def __init__(self, trainings_data, noise, normalize=False,y_scaler=None, inequality_constraints=None):
        self.normalizer = Normalizer()
        self.trainings_data = copy.deepcopy(trainings_data)
        self.noise = noise
        if normalize==True:
            self.normalizer.fit(trainings_data, y_scaler)
            for d in self.trainings_data:
                X,Y = self.trainings_data[d]
                X = self.normalizer.transform_x(X)
                #Y = np.atleast_2d(Y).reshape(-1, 1)
                Y = self.normalizer.transform_y(Y)
                self.noise[d] = self.normalizer.transform_noise(self.noise[d])
                self.trainings_data[d]=[X,Y]
        else:
            for d in self.trainings_data:
                X,Y = self.trainings_data[d]
                X = np.atleast_2d(X)
                Y = np.atleast_2d(Y).reshape(-1, 1)
                self.trainings_data[d]=[X,Y]


        self.fidelities = list(self.trainings_data.keys())
        self.nfidelities = len(self.fidelities)
                
        self.model = None
        if inequality_constraints==None:
            self.inequality_constraints=MFGPInequalityConstraints()
        else:
            self.inequality_constraints=inequality_constraints

    def set_traings_data(self, trainings_data):
        if self.normalizer.scaler_x != None:
            self.normalizer = Normalizer()
            self.normalizer.fit(trainings_data)
            for d in trainings_data:
                X,Y = trainings_data[d]
                X = self.normalizer.transform_x(X)
                #Y = np.atleast_2d(Y).reshape(-1, 1)
                Y = self.normalizer.transform_y(Y)
                trainings_data[d]=[X,Y]
        self.trainings_data = trainings_data

    def build_model(self,n_restarts=10, custom_lengthscale=None):
        """
        Constructs and trains a linear multi-fidelity model using Gaussian processes.
        """
        x_train = []
        y_train = []

        for fidelity in self.fidelities:
            x_tmp=np.atleast_2d(self.trainings_data[fidelity][0])
            y_tmp=np.atleast_2d(self.trainings_data[fidelity][1])
            x_train.append(x_tmp)
            y_train.append(y_tmp)
        
        X_train, Y_train = convert_xy_lists_to_arrays(x_train, y_train)
        
        # Define kernels for each fidelity
        kernels_list = []

        for f in range(self.nfidelities - 1):
            rbf1 = GPy.kern.RBF(input_dim=X_train[0].shape[0] - 1, name=f"RBF_rho_{f}")
            #rbf1 = GPy.kern.Matern32(input_dim=X_train[0].shape[0] - 1, ARD=True, name=f"RBF_rho_{f}")
            #input_dim=X_train[0].shape[0] - 1
            #rbf1=(GPy.kern.RBF(input_dim) +
            #    GPy.kern.Bias(input_dim) +
            #    GPy.kern.Matern32(input_dim))
            #rbf2 = GPy.kern.RBF(input_dim=1, ARD=True, name=f"RBF_delta_{f}")
            rbf2 = GPy.kern.Matern32(1, name=f"RBF_delta_{f}")

            

            # Set custom lengthscale if provided
            if custom_lengthscale is not None:
                rbf1.lengthscale = custom_lengthscale
                rbf2.lengthscale = custom_lengthscale

            #
            rbf1.lengthscale.unconstrain()
            #rbf1.lengthscale.constrain_bounded(1,250)
            #rbf1.lengthscale.set_prior(GPy.priors.Gaussian(mu=200, sigma=np.sqrt(200)))
            #rbf1.variance.constrain_bounded(1e-5, 0.006)

            kernels_list.append(rbf1)
            kernels_list.append(rbf2)


        lin_mf_kernel = kernels.LinearMultiFidelityKernel(kernels_list)
        gpy_lin_mf_model = GPyLinearMultiFidelityModel(X_train, Y_train, lin_mf_kernel, n_fidelities=len(self.fidelities))

        # Fix noise terms for each fidelity
        for i,fidelity in enumerate(self.fidelities):
            # Construct the attribute name dynamically
            noise_attr = f"Gaussian_noise" if i == 0 else f"Gaussian_noise_{i}"
            try:
                getattr(gpy_lin_mf_model.mixed_noise, noise_attr).fix(self.noise[fidelity])
            except AttributeError:
                print(f"Error: Attribute '{noise_attr}' not found in the model.")
                raise



        # Wrap and optimize the model
        self.model = GPyMultiOutputWrapper(
            gpy_lin_mf_model, len(self.fidelities), n_optimization_restarts=n_restarts, verbose_optimization=True
        )

        self.model.optimize()
        return self.model

    def set_data(self,trainings_data_new):
        x_train = []
        y_train = []
        for fidelity in self.fidelities:
            x = trainings_data_new[fidelity][0]
            y = trainings_data_new[fidelity][1]
            if self.normalizer.scaler_x != None:
                x = self.normalizer.transform_x(x)
                #y = np.atleast_2d(y).reshape(-1, 1)
                y = self.normalizer.transform_y(y)
            self.trainings_data[fidelity][0].extend(x)
            self.trainings_data[fidelity][1].extend(y)
            x_tmp=np.atleast_2d(self.trainings_data[fidelity][0])
            y_tmp=np.atleast_2d(self.trainings_data[fidelity][1])
            x_train.append(x_tmp)
            y_train.append(y_tmp)
        
        X_train, Y_train = convert_xy_lists_to_arrays(x_train, y_train)
        self.model.set_data(X_train, Y_train)

    def max_acquisition_integrated_variance_reduction(self, parameters):
        ## Here we run a gradient-based optimizer over the acquisition function to find the next point to attempt. 
        spaces_tmp = []
        for i in parameters:
            spaces_tmp.append(ContinuousParameter(i, parameters[i][0], parameters[i][1]))
        
        spaces_tmp.append(InformationSourceParameter(self.nfidelities))
        parameter_space = ParameterSpace(spaces_tmp)

        optimizer = GradientAcquisitionOptimizer(parameter_space)
        multi_source_acquisition_optimizer = MultiSourceAcquisitionOptimizer(optimizer, parameter_space)
        #acquisition = ModelVariance(mf_model) * inequality_constraints
        acquisition = IntegratedVarianceReduction(self.model, parameter_space, num_monte_carlo_points=2000) * self.inequality_constraints

        # Create batch candidate point calculator
        sequential_point_calculator = SequentialPointCalculator(acquisition, multi_source_acquisition_optimizer)
        loop_state = create_loop_state(self.model.X, self.model.Y)
        x_next = sequential_point_calculator.compute_next_points(loop_state)

        return x_next, acquisition
    
    def max_acquisition_multisource(self, parameters):
        ## Here we run a gradient-based optimizer over the acquisition function to find the next point to attempt. 
        spaces_tmp = []
        for i in parameters:
            spaces_tmp.append(ContinuousParameter(i, parameters[i][0], parameters[i][1]))
        
        spaces_tmp.append(InformationSourceParameter(self.nfidelities))
        parameter_space = ParameterSpace(spaces_tmp)

        optimizer = GradientAcquisitionOptimizer(parameter_space)
        us_acquisition = MultiInformationSourceEntropySearch(self.model, parameter_space) * self.inequality_constraints
        x_new, _ = optimizer.optimize(us_acquisition)
        return x_new, us_acquisition
    
    def max_acquisition_model_variance(self, parameters):
        ## Here we run a gradient-based optimizer over the acquisition function to find the next point to attempt. 
        spaces_tmp = []
        for i in parameters:
            spaces_tmp.append(ContinuousParameter(i, parameters[i][0], parameters[i][1]))
        
        spaces_tmp.append(InformationSourceParameter(self.nfidelities))
        parameter_space = ParameterSpace(spaces_tmp)


        optimizer = GradientAcquisitionOptimizer(parameter_space)
        multi_source_acquisition_optimizer = MultiSourceAcquisitionOptimizer(optimizer, parameter_space)
        acquisition = ModelVariance(self.model) * self.inequality_constraints

        # Create batch candidate point calculator
        sequential_point_calculator = SequentialPointCalculator(acquisition, multi_source_acquisition_optimizer)
        loop_state = create_loop_state(self.model.X, self.model.Y)
        x_next = sequential_point_calculator.compute_next_points(loop_state)
        
        return x_next, acquisition

    def evaluate_model(self, x, fidelity=2):
        x_eval=np.array([x])
        SPLIT = 1
        X_eval = convert_x_list_to_array([x_eval , x_eval, x_eval])
        return self.model.predict(X_eval[int(fidelity)*SPLIT:int(fidelity+1)*SPLIT])[0][0][0]

    def evaluate_model_gradient(self, x, fidelity=2):
        x_eval=np.array([x])
        SPLIT = 1
        X_eval = convert_x_list_to_array([x_eval , x_eval, x_eval])
        return self.model.get_prediction_gradients(X_eval[int(fidelity)*SPLIT:int(fidelity+1)*SPLIT])[0][0]

    def evaluate_model_uncertainty(self, x, fidelity=2):
        x_eval=np.array([x])
        SPLIT = 1
        X_eval = convert_x_list_to_array([x_eval , x_eval, x_eval])
        _, var = self.model.predict(X_eval[int(fidelity)*SPLIT:int(fidelity+1)*SPLIT])
        var=var[0][0]
        var=np.sqrt(var)
        return var

    def get_min(self, parameters, x0=None, fidelity=2):

        def f(x):
            self.evaluate_model(x, fidelity)

        bnds=[]
        for i in parameters:
            bnds.append((parameters[i][0],parameters[i][1]))
            if x0==None:
                x0.append((parameters[i][1]-parameters[i][0])/2.)
        x0=np.array(x0)
        
        res = minimize(f, x0,bounds=bnds)
        return res.x, res.fun
    
    def get_min_constrained(self, parameters, fidelity=2):
        spaces_tmp = []
        for i in parameters:
            spaces_tmp.append(ContinuousParameter(i, parameters[i][0], parameters[i][1]))
        
        spaces_tmp.append(InformationSourceParameter(self.nfidelities))
        parameter_space = ParameterSpace(spaces_tmp)

        model = MFGPAuxilaryModel(self, fidelity)

        optimizer = GradientAcquisitionOptimizer(parameter_space)
        acquisition = model
        x_min, _ = optimizer.optimize(acquisition)
        x_min=[x for x in x_min[0]]
        return x_min, self.evaluate_model(x_min, fidelity)
    

class MFGPAuxilaryModel(Acquisition):
    def __init__(self, mf_model, fidelity):
        self.mf_model = mf_model
        self.fidelity = fidelity
        self.inequality=self.mf_model.inequality_constraints

    def evaluate(self, x):
        delta_inequ=self.inequality.evaluate(x)
        delta_inequ[delta_inequ == 0] = np.inf
        delta_x = np.ones(len(x))
        for i,xi in enumerate(x[:,:]):
            delta_x[i] = -1.*self.mf_model.evaluate_model(xi, self.fidelity)
            if self.mf_model.evaluate_model(xi, self.fidelity) <= 0.:
                delta_x[i] = -0.00001
        return delta_x[:, None]*delta_inequ[:,None]
    
    @property
    def has_gradients(self):
        return True
    
    def get_gradients(self,x):
        delta_x = np.ones(len(x))
        for i,xi in enumerate(x[:,:]):
            delta_x[i] = self.mf_model.evaluate_model_gradient(xi,self.fidelity)[0][0]
        return delta_x[:, None]

    def evaluate_with_gradients(self, x):
        return self.evaluate(x), np.zeros(x[:,:].shape)

# --- Custom Acquisitions ---
class Cost(Acquisition):
    def __init__(self, costs):
        self.costs = costs

    def evaluate(self, x):
        fidelity_index = x[:, -1].astype(int)
        return np.array([self.costs[i] for i in fidelity_index])[:, None]

    @property
    def has_gradients(self):
        return True

    def evaluate_with_gradients(self, x):
        return self.evaluate(x), np.zeros(x.shape)


class MFGPInequalityConstraints(Acquisition):
    def __init__(self):
        pass

    def evaluate(self, x):
        delta_x = np.ones(len(x))

        return delta_x[:, None]


    @property
    def has_gradients(self):
        return True

    def evaluate_with_gradients(self, x):
        return self.evaluate(x), np.zeros(x[:, :-1].shape)
