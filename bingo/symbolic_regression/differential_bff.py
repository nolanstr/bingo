import torch

import numpy as np



class DifferentialRegression_TF(VectorBasedFunction):

    def __init__(self, bff, df_order=1, differential_weight=1.0,
                 detect_const_solutions=False):

        super().__init__(None, metric)

        self._clo_type = clo_type

        self.X = [torch.tensor(X[:, i], dtype=torch.float64) 
                  for i in range(X.shape[1])]
        self.U = U  # Keep as a numpy array
        self.X_df = [torch.tensor(X_df[:, i], dtype=torch.float64)
                      for i in range(X_df.shape[1])]
        for X in self.X_df:
            X.requires_grad = True

        self.differential_weight = differential_weight
        self.detect_const_solutions = detect_const_solutions
        self.df_err = df_err

    def build_torch_graph_from_agraph(self, individual, X):
        
        commands = individual._simplified_command_array
        constants = individual._simplified_constants
        import pdb;pdb.set_trace()
        for i in range(commands.shape[0]):
            node = commands[i, 0]
            if node == -1:
                ad_stack[i] = torch.ones_like(
                    X[0], dtype=torch.float64) * commands[i,1]
            elif node == 0:
                column_idx = commands[i, 1]
                ad_stack[i] = X[column_idx]
            elif node == 1:
                const_idx = commands[i, 1]
                # IMPORTANT: We need to first create the constant using numpy
                # and then convert to tensor to avoid a memory leak. This bypasses
                # one of tensorflows caching mechanisms for constants which causes the leak.
                ad_stack[i] = torch.ones_like(
                    X[0], dtype=torch.float64) * constants[const_idx]
            elif node == 2:
                t1_idx, t2_idx = commands[i, 1], commands[i, 2]
                ad_stack[i] = ad_stack[t1_idx] + ad_stack[t2_idx]
            elif node == 3:
                t1_idx, t2_idx = commands[i, 1], commands[i, 2]
                ad_stack[i] = ad_stack[t1_idx] - ad_stack[t2_idx]
            elif node == 4:
                t1_idx, t2_idx = commands[i, 1], commands[i, 2]
                ad_stack[i] = ad_stack[t1_idx] * ad_stack[t2_idx]
            elif node == 5:
                t1_idx, t2_idx = commands[i, 1], commands[i, 2]
                ad_stack[i] = ad_stack[t1_idx] / ad_stack[t2_idx]
            elif node == 6:
                t1_idx = commands[i, 1]
                ad_stack[i] = torch.sin(ad_stack[t1_idx])
            elif node == 7:
                t1_idx = commands[i, 1]
                ad_stack[i] = torch.cos(ad_stack[t1_idx])
            elif node == 8:
                t1_idx = commands[i, 1]
                ad_stack[i] = torch.exp(ad_stack[t1_idx])
            elif node == 9:
                t1_idx = commands[i, 1]
                ad_stack[i] = torch.log(tf.abs(ad_stack[t1_idx]))
            elif node == 10:
                t1_idx, t2_idx = commands[i, 1], commands[i, 2]
                ad_stack[i] = torch.pow(
                    ad_stack[t1_idx], ad_stack[t2_idx])
            elif node == 11:
                t1_idx = commands[i, 1]
                ad_stack[i] = torch.abs(ad_stack[t1_idx])
            elif node == 12:
                t1_idx = commands[i, 1]
                ad_stack[i] = torch.sqrt(torch.abs(ad_stack[t1_idx]))
            elif node == 13:
                t1_idx, t2_idx = commands[i, 1], commands[i, 2]
                ad_stack[i] = torch.pow(
                    torch.abs(ad_stack[t1_idx]), ad_stack[t2_idx])
            elif node == 14:
                t1_idx = commands[i, 1] 
                ad_stack[i] = torch.sinh(ad_stack[t1_idx])
            elif node == 15:
                t1_idx = commands[i, 1]
                ad_stack[i] = torch.cosh(ad_stack[t1_idx])
            else:
                raise IndexError(f"Node value {node} unrecognized") 

        return ad_stack[-1]

    def evaluate_fitness_vector(self, individual):

        self.eval_count += 1
        ad_graph_function = self.build_torch_graph_from_agraph(individual)

        # evaluate model at collocation points
        U_df = ad_graph_function(self.X_df)

        # translate pdefn output to numpy
        diff_eqn_errs_pt = self.df_err(self.X_df, U_df)

        for diff_eqn_err_pt in diff_eqn_errs_pt:
            diff_eqn_err = diff_eqn_err_pt.detach().numpy()

            if self._clo_type == "optimize":
                fitness += self.differential_weight * self._metric(diff_eqn_err)
            elif self._clo_type == "root":
                fitness = np.concatenate((fitness, diff_eqn_err), axis=0)

        if np.isnan(fitness).any():
            fitness *= np.inf
 
        return fitness
