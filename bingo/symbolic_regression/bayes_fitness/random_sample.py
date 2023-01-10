import numpy as np
import math

class RandomSample:

    def __init__(self, training_data, multisource_info, random_sample_info):
        
        self._set_subsample_parameters(random_sample_info)
        self._set_multisource_info(multisource_info)
        self._set_random_sample_info(random_sample_info)
        self._set_subset_idxs()
        
        self.randomize_subsets()

    def randomize_subsets(self):

        for i, subset_size in enumerate(self._multisource_num_pts):
            
            if subset_size > self.full_idxs[i].shape[0]:
                replace=True
            else:
                replace = False
            self.subset_idxs[i] = np.sort(np.random.choice(self.full_idxs[i],
                                                subset_size, replace=replace))
        
        self._x = np.vstack([self.training_data.x[subset_idxs,:] for \
                                        subset_idxs in self.subset_idxs])
        self._y = np.vstack([self.training_data.y[subset_idxs,:] for \
                                        subset_idxs in self.subset_idxs])
    
    @property
    def x_subset(self):
        return self._x

    @property
    def y_subset(self):
        return self._y
    
    def _set_subsample_parameters(self, random_sample_info):
        
        if random_sample_info == None:
            self.multisource_num_points = (self.training_data.x.shape[0])
       
    def _set_subset_idxs(self):
        all_idxs = np.arange(sum(self._full_multisource_num_pts))
        idx_markers = np.cumsum([0] + list(self._full_multisource_num_pts))
        self.full_idxs = [all_idxs[idx_markers[i]:idx_markers[i+1]] for i in \
                                                            range(len(idx_markers)-1)]
        self.subset_idxs = [np.empty(num_pts).astype(np.int) for num_pts \
                                    in self._multisource_num_pts]

    def get_dataset(self, subset=None):
        
        if subset is None:
            return self._x, self._y

        else:
            return self.training_data.x[self.subset_idxs[subset]], \
                                self.training_data.y[self.subset_idxs[subset]]
    
    def _set_multisource_info(self, multisource_info):
    
        if multisource_info is None:
            multisource_info = \
                    tuple([self._cont_local_opt.training_data.x.shape[0]])
    
        self._multisource_num_pts = tuple(multisource_info)
        self._full_multisource_num_pts = tuple(multisource_info)

    def _update_multisource(self, random_sample_info):
        
        if not isinstance(random_sample_info, np.ndarray):
            random_sample_info = np.array(random_sample_info)
        assert(len(random_sample_info)==len(self._multisource_num_pts)), \
                'length of random sample subsets must match multisource num pts'

        if np.all(random_sample_info>=1):
            random_sample_info = np.minimum(random_sample_info, 
                                            np.array(self._full_multisource_num_pts))
            self._multisource_num_pts = tuple(random_sample_info.astype(int).tolist())
        
        elif np.all(random_sample_info<=1):
            self._multisource_num_pts = tuple([math.ceil(subset_size * subset_percent) \
                                      for subset_size, subset_percent in \
                                      zip(self._multisource_num_pts, random_sample_info)])
        else:
            raise ValueError(\
                    'random sample info needs to be all greater than 1 or all less than 1')

        assert(sum(self._multisource_num_pts) < \
               sum(self._full_multisource_num_pts), 
               'random sample subset smaller than full subsets')

    def _set_random_sample_info(self, random_sample_info):
        if np.any([isinstance(random_sample_info, float),
                  isinstance(random_sample_info, int)]):
            self._random_sample_subsets = random_sample_info
            self._update_multisource(
                    len(self._multisource_num_pts) * [random_sample_info])

        elif np.any([isinstance(random_sample_info, list), 
                     isinstance(random_sample_info, tuple),
                     isinstance(random_sample_info, np.ndarray)]):
            self._random_sample_subsets = random_sample_info
            self._update_multisource(random_sample_info, uneven_sampling=True)

        else:
            self._random_sample_subsets = 1.0
