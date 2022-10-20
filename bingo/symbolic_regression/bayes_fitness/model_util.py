import numpy as np

class utilities:

    def __init__(self):
        pass
    
    def estimate_cred_pred(self, ind, step_list, subset=None, bounds=0.05):

        n_params = ind.get_number_local_optimization_params()
        x, y = self.subset_data.get_dataset(subset=subset)

        ind.set_local_optimization_params(step_list[-1].params.T)
        model_outputs = \
               ind.evaluate_equation_at(x).T
        weights = step_list[-1].weights

        if subset != None:
            subset += n_params
            noise_stds = step_list[-1].params[:, subset].reshape((-1,1))
        else:
            noise_stds = step_list[-1].params[:, n_params].reshape((-1,1))
        
        cred = self._estimate_interval(model_outputs, weights)
        pred = self._estimate_interval(model_outputs, weights, noise=noise_stds)

        return x, cred, pred
         
    def _estimate_interval(self, model_output, weights, noise=None):
        """
        if noise is None --> credible interval
        else --> predicted interval
        """
        if noise == None:
            noise = np.zeros(model_outputs.shape[0])

        if subset != None:
            subset += n_params
            noise_stds = step_list[-1].params[:, subset].reshape((-1,1))
        else:
            noise_stds = step_list[-1].params[:, n_params].reshape((-1,1))

        means = np.zeros(model_outputs.shape)
        model_outputs = model_outputs + np.random.normal(means, abs(noise_stds))

        model_outputs_sort = np.argsort(model_outputs, axis=0)
        model_outputs_weights = np.cumsum(weights[model_sort].squeeze(), axis=0)
        model_out_sort = np.sort(model_outputs, axis=0)
        model_y = np.array((2, model_output.shape[1]))

        for i, weight_col in enumerate(model_output_weights.T):
            model_y[i] = np.interp([bounds, 1-bounds], weight_col,
                                                model_out_sort[:,i])

        return model_y

