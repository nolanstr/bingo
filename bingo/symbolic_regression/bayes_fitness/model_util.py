import numpy as np
from copy import deepcopy

class Utilities:

    def __init__(self):
        pass
    
    def estimate_cred_pred(self, ind, step_list, subset=None, bounds=0.05,
                                                            sort_x=True,
                                                            linspace=True, 
                                                            step_list_term=-1):

        ind = deepcopy(ind)
        n_params = ind.get_number_local_optimization_params()
        x, y = self.get_dataset(subset=subset)
        x_sort = np.argsort(x[:,0])
        x = x[x_sort, :]
        y = y[x_sort, :]
        raw_data = [x, y]
        
        if linspace:
            empty_data = np.empty((linspace, x.shape[1]))
            empty_data[:, 0] = np.linspace(x[:,0].min(), x[:,0].max(), linspace)
            empty_data[:, 1] = x[0,1]
            empty_data[:, 2] = x[0,2]
            x = empty_data

        ind.set_local_optimization_params(step_list[step_list_term].params.T)
        model_outputs = ind.evaluate_equation_at(x).T
        weights = step_list[step_list_term].weights
        if subset != None:
            subset += n_params
            noise_stds = step_list[step_list_term].params[:, subset].reshape((-1,1))
        else:
            noise_stds = step_list[step_list_term].params[:, n_params].reshape((-1,1))
        
        cred = self._estimate_interval(model_outputs, weights, bounds=bounds)
        pred = self._estimate_interval(model_outputs, weights, noise=noise_stds,
                                                                    bounds=bounds)

        return x, raw_data, cred, pred
         
    def base_estimate_cred_pred(self, ind, step_list, subset=None, bounds=0.05,
                                                            sort_x=True,
                                                            linspace=True, 
                                                            step_list_term=-1):

        ind = deepcopy(ind)
        n_params = ind.get_number_local_optimization_params()
        x, y = self.get_dataset(subset=subset)
        x_sort = np.argsort(x[:,0])
        raw_data = [x, y]
        
        if linspace:
            empty_data = np.empty((linspace, x.shape[1]))
            empty_data[:, 0] = np.linspace(x[:,0].min(), x[:,0].max(), linspace)
            x = empty_data

        ind.set_local_optimization_params(step_list[step_list_term].params.T)
        model_outputs = ind.evaluate_equation_at(x).T
        weights = step_list[step_list_term].weights

        if subset != None:
            subset += n_params
            noise_stds = step_list[step_list_term].params[:, subset].reshape((-1,1))
        else:
            noise_stds = step_list[step_list_term].params[:, n_params].reshape((-1,1))
        
        cred = self._estimate_interval(model_outputs, weights, bounds=bounds)
        pred = self._estimate_interval(model_outputs, weights, noise=noise_stds,
                                                                    bounds=bounds)

        return x, raw_data, cred, pred

    def _estimate_interval(self, model_outputs, weights, noise=None, bounds=0.05):
        """
        if noise is None --> credible interval
        else --> predicted interval
        """
        if noise is None:
            noise = np.zeros(model_outputs.shape[0]).reshape((-1,1))

        means = np.zeros(model_outputs.shape)
        model_outputs = model_outputs + np.random.normal(means, abs(noise))

        model_outputs_sort = np.argsort(model_outputs, axis=0)
        model_outputs_weights = np.cumsum(weights[model_outputs_sort].squeeze(), axis=0)
        model_out_sort = np.sort(model_outputs, axis=0)
        model_y = np.empty((model_outputs.shape[1], 2))

        for i, weight_col in enumerate(model_outputs_weights.T):
            model_y[i] = np.interp([bounds, 1-bounds], weight_col,
                                                model_out_sort[:,i])

        return model_y

