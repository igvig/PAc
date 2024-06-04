# %% [markdown]
# ### Finding optimal $x_\text{low}$ and $x_\text{high}$
# For a given lookup array size $n$, and lower and upper bounds ($x_\text{low}$ and $x_\text{high}$), the absolute difference $\varepsilon$ between the approximation function $\tilde{a}(x)$ and its source activation function $a(x)$ can be computed.  Finding the best values for these parameters is a simple optimization problem.  Given resource constraints, it is likely that one would start from an array size $n$ and look for optimal array bounds to match that array size.  While it should be possible to derive a closed form equation of this error $\varepsilon$ for any given closed form activation $a(x)$, a computational approach may yield more automated results. This is the approach taken here.
# 
# Multiple optimization goals could be aimed for. The present work aims for two different goals: (a) minimizing the maximum point error $\varepsilon(x)$ so that the deviation from actual activation is localized to a particular set of trigger values, and (b) minimizing the absolute area under the curve of $\varepsilon(x)$ to ensure the overall faithfulness of the approximation $\tilde{a}(x)$ to its activation $a(x)$.  Given the scale difference between the two goals and the lack of an objective preference or weights of the two, the product of these will be used as the objective function to minimize.  In other words, the problem can be stated as finding
# $$
# \min \left(\varepsilon_\text{max}\times AUC\right)
# $$
# 
# -----
# 
# The total $AUC$ of a pre-computed activation function $\tilde{a}(x)$ with respect to its corresponding activation function $a(x)$ is:
# $$
# \begin{align}
# \varepsilon &= \int \left|\tilde{a}(x) - a(x)\right| dx\\
#             &= \int_{-\infty}^{x_\text{low}} \left|\alpha_\text{low}(x) - a(x)\right| dx +
#                \int_{x_\text{low}}^{x_\text{high}} \left|\tilde{a}(x) - a(x)\right| dx +
#                \int_{x_\text{high}}^{\infty} \left|\alpha_\text{high}(x) -a(x)\right| dx\\
#             &\approx \int_{x_\text{min}}^{x_\text{low}} \left|\alpha_\text{low}(x) - a(x)\right| dx +
#                \int_{x_\text{low}}^{x_\text{high}} \left|\tilde{a}(x) - a(x)\right| dx +
#                \int_{x_\text{high}}^{x_\text{max}} \left|\alpha_\text{high}(x) -a(x)\right| dx
# \end{align}
# $$
# where $[x_\text{min}; x_\text{max}]$ is a large enough interval that the absolute error outside of this interval is negligible or zero (i.e., $\int_{-\infty}^{x_\text{min}}\left|\alpha_\text{low}(x)-a(x)\right|dx + \int_{x_\text{max}}^{\infty}\left|\alpha_\text{high}(x)-a(x)\right|dx \approx 0$).
# 
# For the tail integrals (the first and third terms in the last equation above), over any very small interval $[x_1; x_2]$, the error term can be approximated by a straight line; that is:
# $$
# \begin{align}
# \int_{x_1}^{x_2} \left|\alpha(x)-a(x)\right|dx &\approx\int_{x_1}^{x_2} \left|mx+b\right|dx \\
#                                  &= \frac{|mx_2+b|(mx_2+b) - |mx_1+b|(mx_1+b)}{2m} \\
#                                  &= \frac{|y_2|y_2 - |y_1|y_1}{2\frac{y_2-y_1}{x_2-x_1}}\\
#                                  &= \frac{|\alpha(x_2)-a(x_2)|(\alpha(x_2)-a(x_2)) - |\alpha(x_1)-a(x_1)|(\alpha(x_1)-a(x_1))}{2\frac{\alpha(x_2)-a(x_2)-\alpha(x_1)+a(x_1)}{x_2-x_1}}
# \end{align}
# $$

# %% [markdown]
# ## Utility Functions & Environment Initialization

# %%
import os, sys
import datetime as dt
import gc
import pandas as pd

def get_env_type() -> str:
    '''
    Get the environment type where the code is running.

    Returns:
    - 'kaggle' if running on Kaggle
    - 'google.colab' if running on Google Colab
    - 'local' if running on local environment
    '''
    if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
        return 'kaggle'
    elif 'google.colab' in sys.modules:
        if 'COLAB_TPU_ADDR' in os.environ:  # Google Colab w/ TPU
            # Connect to TPU
            import tensorflow
            tpu = tensorflow.distribute.cluster_resolver.TPUClusterResolver()
            tensorflow.config.experimental_connect_to_cluster(tpu)
            tensorflow.tpu.experimental.initialize_tpu_system(tpu)
        # Connect to Drive
        from google.colab import drive
        drive.mount('/content/drive')
        return 'google.colab'
    else:   # Running on local environment
        return 'local'

def print_versions_and_GPU() -> None:
    '''
    Prints version numbers for various modules and GPU information (if available).
    '''
    import sys, tensorflow, sklearn
    print(f'Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')
    print(f'TensorFlow: {tensorflow.__version__}')
    try:
        print(f'Keras: {tensorflow.keras.version()}')
    except:
        print(f'Keras: Unknown version')
    print(f'Scikit-learn: {sklearn.__version__}')
    if get_env_type() == 'google.colab':
      if os.environ.keys().__contains__('TPU_ACCELERATOR_TYPE'):
          print(f"Running on TPU {os.environ['TPU_ACCELERATOR_TYPE']}")
      else:
        GPU_info = "" #!nvidia-smi -L
        GPU_info = '\n'.join(GPU_info)
        if GPU_info.find('command not found') >= 0:
          print("Running on CPU")
        else:
          print(f"Running on GPU {GPU_info} ({os.environ['CUDA_VERSION']})")

def log(msg: str|None = None, end: str|None = '\n'):
    print(f"{dt.datetime.now().isoformat().replace('T',' ')} {msg}", end=end)



# %% [markdown]
# ## Core Object Definitions

# %% [markdown]
# ### Precomputed Activation (PAc) Approximator

# %%
import math
from typing import Callable, Iterable, Tuple, List
import numpy as np

class PAc_Approximator:

    def __init__(self, func: Callable, func_low: Callable, func_high: Callable, xmin: float = -20., xmax: float = 20., w: float = 0.01):
        """
        Initialize the PAc_Approximator class.

        Parameters:
        - func (Callable): The activation function to be used to initialize the PAF lookup table.
        - func_low (Callable): The activation function to be used for x <= x_low.
        - func_high (Callable): The activation function to be used for x >= x_high.
        - xmin (float): The lower bound for the low asymptotic tail.
            Default: -20.
        - xmax (float): The upper bound for the high asymptotic tail.
            Default: 20.
        - w (float): The width of the intervals on the asymptotic tails.
            Default: 0.01
        """
        self._xmin = xmin
        self._xmax = xmax
        self._w = w
        self._func = func
        self._func_low = func_low
        self._func_high = func_high
        self._init_low_tail()
        self._init_high_tail()

    @staticmethod
    def _local_auc(func_a: Callable, func_b: Callable, x1: float, x2: float) -> float:
        """
        Calculate the absolute area under the curve (AUC) between two functions, func_a
        and func_b, within the range of x1 and x2.

        Parameters:
        - func_a (Callable): The first function.
        - func_b (Callable): The second function.
        - x1 (float): The lower bound of the range.
        - x2 (float): The upper bound of the range.

        Returns:
        - float: The local AUC between func_a and func_b within the range of x1 and x2.
        """
        if x1 == x2:
            return 0
        elif x1 > x2:
            x1, x2 = x2, x1

        y2 = func_a(x2) - func_b(x2)
        y1 = func_a(x1) - func_b(x1)
        if (y2 == y1):
            return abs(y2 * (x2 - x1))
        else:
            m = (y2 - y1) / (x2 - x1)
            return ((abs(y2) * y2) - (abs(y1) * y1)) / (2 * m)


    def _init_low_tail(self):
        """
        Initializes the cumulative absolute area under the curve for the low asymptotic tail.

        Returns:
        - None
        """
        self.priv_low_tail_max = [0.]
        self.priv_low_tail = [0.]
        current_max = -1000000.0
        x1 = self._xmin
        x_high = 0 - self._w
        x2 = min(x1 + self._w, x_high)
        auc = 0
        while x1 < x_high:
            current_max = max(current_max, abs(self._func(x1) - self._func_low(x1)))
            self.priv_low_tail_max.append(current_max)
            auc += PAc_Approximator._local_auc(self._func, self._func_low, x1, x2)
            self.priv_low_tail.append(auc)
            x1 = x2
            x2 = min(x2 + self._w, x_high)
        # print(priv_low_tail)

    def _init_high_tail(self):
        """
        Initializes the cumulative absolute area under the curve for the high asymptotic tail.

        Returns:
        - None
        """
        self.priv_high_tail_max = [0.]
        self.priv_high_tail = [0.]
        high_tail_inorder = [0.]
        high_tail_max_inorder = [0.]
        x1 = 0 + self._w
        x_high = self._xmax
        x2 = min(x1 + self._w, x_high)
        auc = 0
        while x1 < x_high:
            local_auc = PAc_Approximator._local_auc(self._func, self._func_high, x1, x2)
            high_tail_inorder.append(local_auc)
            local_max = abs(self._func(x1) - self._func_high(x1))
            high_tail_max_inorder.append(local_max)
            x1 = x2
            x2 = min(x2 + self._w, x_high)
        # Now we reverse the lists and accumulate the values (in reverse order)
        high_tail_inorder.reverse()
        high_tail_max_inorder.reverse()
        auc = 0
        for l in high_tail_inorder:
            auc += l
            self.priv_high_tail.append(auc)
        max = 0
        for m in high_tail_max_inorder:
            max = m if m > max else max
            self.priv_high_tail_max.append(max)
        # Reverse the results
        self.priv_high_tail.reverse()
        self.priv_high_tail_max.reverse()
        #print(f"priv_high_tail: {self.priv_high_tail}")


    def _approx_abs_auc(self, func_a: Callable, func_b: Callable, x_low: float, x_high: float, w:float=0.01, is_low_tail: bool=False, is_high_tail: bool=False):
        """Approximate the absolute area under the curve of the difference between the two functions.

        Parameters:
            func_a: Callable, the first function.
            func_b: Callable, the second function.
            x_low: float, the lower bound of the integration.
            x_high: float, the upper bound of the integration.
            w: float, the width of the intervals.
                Default: 0.01
        """
        assert x_low <= x_high, f"x_low ({x_low}) must be less than x_high ({x_high})"
        assert w > 0, f"w ({w}) must be greater than 0"

        if is_low_tail:
            x_idx = int(math.floor((x_high - x_low) / w))
            if x_idx == len(self.priv_low_tail):
                x_idx -= 1
            return self.priv_low_tail[x_idx] + self._local_auc(func_a, func_b, x_idx * w + x_low, x_high), \
                max(self.priv_low_tail_max[x_idx], abs(func_a(x_high) - func_b(x_high)))
        elif is_high_tail:
            x_idx = int(math.floor(x_low / w))
            return self.priv_high_tail[x_idx] + self._local_auc(func_a, func_b, x_low, x_idx * w), \
                max(self.priv_high_tail_max[x_idx], abs(func_a(x_low) - func_b(x_low)))
        else:
            retval = 0
            x1 = x_low
            x2 = min(x_low + w, x_high)
            y1 = func_a(x1) - func_b(x1)
            current_max = abs(y1)
            while x1 < x_high:
                y2 = func_a(x2) - func_b(x2)
                current_max = max(current_max, abs(y2))
                if (y2 == y1):
                    retval += abs(y2 * (x2 - x1))
                else:
                    m = (y2 - y1) / (x2 - x1)
                    retval += ((abs(y2) * y2) - (abs(y1) * y1)) / (2 * m)
                x1 = x2
                x2 = min(x2 + w, x_high)
                y1 = y2
            return retval, current_max

    def _approx_paf_abs_auc(self, x_low: float, x_high: float, n=64):
        """Approximate the absolute area under the curve of the difference between the activation and the PAc.

        Args:
            x_low: float, the lower bound of the integration.
            x_high: float, the upper bound of the integration.
            n: int, the number of points in the lookup table.
                Default: 64
        """
        assert x_low >= self._xmin, f"x_low ({x_low}) must be greater than xmin ({self._xmin})"
        assert x_high <= self._xmax, f"x_high ({x_high}) must be less than xmax ({self._xmax})"
        assert n > 1, f"n ({n}) must be greater than 1"

        if (x_low == x_high):
            return 0, 0

        paf_table = [self._func(x_low + (i + 0.5) * (x_high - x_low) / n) for i in range(n+1)]
        width = (x_high - x_low) / n
        mult = n / (x_high - x_low)
        add = (x_low * n) / (x_low - x_high)
        return self._approx_abs_auc(self._func, lambda x: paf_table[np.floor(x * mult + add).astype(int)], x_low, x_high, w=width)

    def approx_error(self, x_low: float, x_high: float, n: int = 64) -> Tuple[float, float, float]:
        """Compute the approximate error of a PAF layer.

        Args:
            x_low: float, the lower bound of the lookup table.
            x_high: float, the upper bound of the lookup table.
            n: int, the number of points in the lookup table.
                Default: 64

        Returns:
            Tuple[float, float, float]: The compound, AUC and max error values for the approximation.
        """
        assert x_low >= self._xmin, f"x_low ({x_low}) must be greater than xmin ({self._xmin})"
        assert x_high <= self._xmax, f"x_high ({x_high}) must be less than xmax ({self._xmax})"
        assert x_low <= x_high, f"x_low ({x_low}) must be less than x_high ({x_high})"
        assert n > 1, f"n ({n}) must be greater than 1"

        auc_low, max_low = self._approx_abs_auc(self._func, self._func_low, self._xmin, x_low, is_low_tail=True)
        auc_int, max_int = self._approx_paf_abs_auc(x_low, x_high, n)
        auc_high, max_high = self._approx_abs_auc(self._func, self._func_high, x_high, self._xmax, is_high_tail=True)

        auc = auc_low + auc_int + auc_high
        max_error = max(max_low, max(max_int, max_high))

        return auc * max_error, auc, max_error


    def approx_swarm_errors(self, swarm: Iterable, n: int = 64) -> List[float]:
        """
        Compute the approximate error for a swarm of particles.
        """
        return [self.approx_error(particle[0], particle[1], n)[0] for particle in swarm]
    

    def approx_max_e_swarm(self, swarm: Iterable, n: int = 64) -> List[float]:
        """
        Compute the maximum error for a swarm of particles.
        """
        return [self.approx_error(particle[0], particle[1], n)[2] for particle in swarm]

# %% [markdown]
# ### Particle Swarm Optimization (PSO)

# %%
from pyswarms.single.global_best import GlobalBestPSO
import numpy as np

class PSOEngine2D:

    def __init__(self, swarm_size: int = 10, max_iter: int = 100,
                 c1: float = 0.5, c2: float = 0.3, w: float = 0.9,
                 min_x: float = -20., max_x: float = 0., min_y: float = 0., max_y: float = 20.):
        self._swarm_size = swarm_size
        self._max_iter = max_iter
        self._c1 = c1
        self._c2 = c2
        self._w = w
        self.optimizer = GlobalBestPSO(n_particles=swarm_size,
                                       dimensions=2,
                                       options={"c1": c1, "c2": c2, "w": w},
                                       bounds=np.array([[min_x, max_x], [min_y, max_y]]))
        self.best_cost = 1000000.
        self.best_pos = (-100., -100.)

    def optimize(self, monitor: Callable, max_iter: int = 100, sig_digits: int = 4, patience: int = 10, **kwargs):

        def _check_sig_digits(value1: float, value2: float, sig_digits: int) -> bool:
            significand1, exp1 = math.frexp(value1)
            significand2, exp2 = math.frexp(value2)
            if (exp1 == exp2):
                if (round(significand1, sig_digits) == round(significand2, sig_digits)):
                    return True
            return False

        patience_left = patience

        log(f"Starting optimization {max_iter}", end=' ')
        for i in range(max_iter):
            cost, pos = self.optimizer.optimize(monitor, 1, verbose=False, **kwargs)
            if _check_sig_digits(self.best_cost, cost, sig_digits):
                patience_left -= 1
                if patience_left == 0:
                    print(f" - Early stopping: {self.best_cost} @ {self.best_pos}")
                    break
                else:
                    print(f".", end='')
            else:
                self.best_cost = cost
                self.best_pos = pos
                patience_left = patience
                print(f"{cost:.5e} ", end='')

        return self.optimizer.cost_history[-1]

# %% [markdown]
# ### Multi-Experiment Runner

# %%
import pandas as pd
import os

def multi_opt_runner(data_path: str, name: str, monitor: Callable, approx: PAc_Approximator,
                     num_runs: int=5, max_iter: int = 100, sig_digits: int = 4,
                     patience: int = 10, **kwargs) -> pd.DataFrame:
    """
    Runs a multiple runs of an optimization algorithm, or reads them from file.

    Parameters:
    - data_path (str): The path to store the results.
    - name (str): The name of the experiment.
    - monitor (Callable): The function to be optimized.
    - approx (PAc_Approximator): The approximator to be used to calculate the error.
    - num_runs (int): The number of runs to be performed.
        Default: 5
    - max_iter (int): The maximum number of iterations.
        Default: 100
    - sig_digits (int): The number of significant digits to be used to consider
    when comparing results for early stopping.
        Default: 4
    - patience (int): The number of iterations to wait before early stopping.
        Default: 10
    - kwargs: Additional keyword arguments to be passed to the optimization
    algorithm.
    """
    csv_file_name = f'{data_path}{name}.csv'
    if (os.path.isfile(csv_file_name)):
        df = pd.read_csv(csv_file_name)
        number_read = df.shape[0]
        log(f"{name} loaded ({number_read})", end='\n')
        if number_read < num_runs:
            todo = num_runs - number_read
        else:
            return df
    else:
        todo = num_runs
        df = pd.DataFrame({
            'x_min': pd.Series(dtype=float), 
            'x_max': pd.Series(dtype=float), 
            'error': pd.Series(dtype=float), 
            'error_auc': pd.Series(dtype=float), 
            'error_max': pd.Series(dtype=float)})
    log(f"Starting {name}, {todo} runs",end='\n')
    for i in range(todo):
        pso = PSOEngine2D()
        cost =pso.optimize(monitor, max_iter, sig_digits, patience, **kwargs)
        pos = pso.best_pos
        compound, auc, emax = approx.approx_error(pos[0], pos[1], **kwargs)
        df.loc[len(df.index)]=[pos[0], pos[1], cost, auc, emax]
        df.to_csv(csv_file_name, index=False)
    return df



def multi_opt_max_runner(data_path: str, name: str, monitor: Callable, approx: PAc_Approximator,
                         num_runs: int=5, max_iter: int = 100, sig_digits: int = 4,
                         patience: int = 10, **kwargs) -> pd.DataFrame:
    """
    Runs a multiple runs of an optimization algorithm, or reads them from file.

    Parameters:
    - data_path (str): The path to store the results.
    - name (str): The name of the experiment.
    - monitor (Callable): The function to be optimized.
    - approx (PAc_Approximator): The approximator to be used to calculate the error.
    - num_runs (int): The number of runs to be performed.
        Default: 5
    - max_iter (int): The maximum number of iterations.
        Default: 100
    - sig_digits (int): The number of significant digits to be used to consider
    when comparing results for early stopping.
        Default: 4
    - patience (int): The number of iterations to wait before early stopping.
        Default: 10
    - kwargs: Additional keyword arguments to be passed to the optimization
    algorithm.
    """
    csv_file_name = f'{data_path}{name}.csv'
    if (os.path.isfile(csv_file_name)):
        df = pd.read_csv(csv_file_name)
        number_read = df.shape[0]
        log(f"{name} loaded ({number_read})", end='\n')
        if number_read < num_runs:
            todo = num_runs - number_read
        else:
            return df
    else:
        todo = num_runs
        df = pd.DataFrame({
            'x_min': pd.Series(dtype=float), 
            'x_max': pd.Series(dtype=float), 
            'error': pd.Series(dtype=float), 
            'error_auc': pd.Series(dtype=float), 
            'error_max': pd.Series(dtype=float)})
    log(f"Starting {name}, {todo} runs",end='\n')
    for i in range(todo):
        pso = PSOEngine2D()
        cost = pso.optimize(monitor, max_iter, sig_digits, patience, **kwargs)
        pos = pso.best_pos
        compound, auc, emax = approx.approx_error(pos[0], pos[1], **kwargs)
        df.loc[len(df.index)]=[pos[0], pos[1], compound, auc, emax]
        df.to_csv(csv_file_name, index=False)
    return df


# %% [markdown]
# ### Error Plotting (3D)

# %%
import os
from matplotlib import colormaps
import matplotlib.pyplot as plt

class Error3D:
    _pf_x = '_x'
    _pf_y = '_y'
    _pf_loss = '_loss'
    _pf_auc = '_auc'
    _pf_max_error = '_max_error'
    _postfixes = [_pf_x, _pf_y, _pf_loss, _pf_auc, _pf_max_error]

    def __init__(self, data_path: str, name: str, label: str, best_x: float, best_y: float, approx: PAc_Approximator, pac_table_size: int, xmin:float|None=None, xmax:float|None=None, ymin:float|None=None, ymax:float|None=None, grid_size:int|None=None):
        self.data_path = data_path
        self.name = name
        self.label = label
        self._xmin = xmin
        self._xmax = xmax
        self._ymin = ymin
        self._ymax = ymax
        self._grid_size = grid_size
        self.best_x = best_x
        self.best_y = best_y
        if not self.load_data():
            assert xmin is not None, "xmin must be specified for initial computation"
            assert xmax is not None, "xmax must be specified for initial computation"
            assert ymin is not None, "ymin must be specified for initial computation"
            assert ymax is not None, "ymax must be specified for initial computation"
            assert grid_size is not None, "grid_size must be specified for initial computation"
            self._x = np.linspace(xmin, xmax, grid_size)
            self._y = np.linspace(ymin, ymax, grid_size)
            self.loss = np.empty((grid_size, grid_size))
            self.auc = np.empty((grid_size, grid_size))
            self.max_error = np.empty((grid_size, grid_size))
            for i, x in enumerate(self._x):
                log(f"Building mesh row {i+1}/{grid_size} ({100 * (i+1) / grid_size:.2f}%)")
                for j, y in enumerate(self._y):
                    self.loss[i, j], self.auc[i, j], self.max_error[i, j] = approx.approx_error(x, y, pac_table_size)
            self.save_data(data_path)
        self.x, self.y = np.meshgrid(self._x, self._y)
        self.best_loss, self.best_auc, self.best_max = approx.approx_error(self.best_x, self.best_y, pac_table_size)

    def load_data(self):
        if all([os.path.isfile(f"{self.data_path}{self.name}{postfix}.npy") for postfix in Error3D._postfixes]):
            self._x = np.load(f"{self.data_path}{self.name}{Error3D._pf_x}.npy")
            self._y = np.load(f"{self.data_path}{self.name}{Error3D._pf_y}.npy")
            self.loss = np.load(f"{self.data_path}{self.name}{Error3D._pf_loss}.npy")
            self.auc = np.load(f"{self.data_path}{self.name}{Error3D._pf_auc}.npy")
            self.max_error = np.load(f"{self.data_path}{self.name}{Error3D._pf_max_error}.npy")

            xmin=self._x[0]
            xmax=self._x[-1]
            ymin=self._y[0]
            ymax=self._y[-1]
            grid_size = len(self._x)

            if (self._xmin == xmin) and (self._xmax == xmax) and (self._ymin == ymin) and (self._ymax == ymax) and (self._grid_size == grid_size):
                log("Data Loaded")
                return True
            else:
                log("Parameters do not match previously computed data")
                return False
        else:
            return False

    def save_data(self, data_path: str):
        np.save(f"{data_path}{self.name}{Error3D._pf_x}.npy", self._x)
        np.save(f"{data_path}{self.name}{Error3D._pf_y}.npy", self._y)
        np.save(f"{data_path}{self.name}{Error3D._pf_loss}.npy", self.loss)
        np.save(f"{data_path}{self.name}{Error3D._pf_auc}.npy", self.auc)
        np.save(f"{data_path}{self.name}{Error3D._pf_max_error}.npy", self.max_error)

    def _plot(self, z:np.ndarray, zbest:float, title:str, zlabel:str, width:float=6, height:float=4, cmap:str='gist_ncar'):
        my_cmap = colormaps.get_cmap(cmap)
        fig = plt.figure(figsize=(width, height))
        ax = plt.axes(projection="3d")
        ax.plot_surface(self.x, self.y, z, cmap=my_cmap)
        zmax=math.ceil(np.max(z))
        ax.plot([self.best_x, self.best_x], [self.best_y, self.best_y], [zbest, zmax], color='red', marker=None, alpha=1)
        ax.plot([self.best_x], [self.best_y], [zbest], marker="v", color='red', alpha=1)
        ax.set_xlabel(r'$x_{low}$')
        ax.set_ylabel(r'$x_{high}$')
        ax.set_zlabel(zlabel)
        ax.set_zlim((0,zmax))
        ax.set_xlim((self._xmin, self._xmax))
        ax.set_ylim((self._ymin, self._ymax))
        ax.zaxis.labelpad=-0.9
        fig.suptitle(title, fontsize=10)
        plt.tight_layout()
        plt.show()

    def plot_combined_error(self, **kwargs):
        self._plot(self.loss, self.best_loss, f'Combined Error for {self.label}', r'$\varepsilon_{max} \times AUC$', **kwargs)

    def plot_max_error(self, **kwargs):
        self._plot(self.max_error, self.best_max, f'Max Error for {self.label}', r'$\varepsilon_{max}$', **kwargs)

    def plot_auc(self, **kwargs):
        self._plot(self.auc, self.best_auc, f'Absolute AUC for {self.label}', r'$AUC$', **kwargs)



