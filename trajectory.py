import numpy as np
from scipy.integrate import solve_ivp
from pyDOE import lhs
import torch
from torch.utils.data import Dataset


class SequenceGenerator:
    def __init__(self, rng: np.random.Generator = None):
        self._rng = rng if rng else np.random.default_rng()

    def sample(self, time_range, delta):
        return self._sample_impl(time_range, delta)


class TrajectoryGenerator:
    def __init__(self,
                 state_dimension,
                 dynamics,
                 control_delta,
                 control_generator: SequenceGenerator,
                 method='RK45'):
        self._n = state_dimension
        self._ode_method = method
        self._dyn = dynamics
        self._rng = np.random.default_rng()
        self._delta = control_delta  # control sampling time
        self._seq_gen = control_generator

        self._init_time = 0.

    def get_example(self, time_horizon, n_samples):
        y0 = self._rng.standard_normal(size=self._n)

        control_seq = self._seq_gen.sample(time_range=(self._init_time,
                                                       time_horizon),
                                           delta=self._delta)

        def f(t, y):
            n_control = int(np.floor((t - self._init_time) / self._delta))
            u = control_seq[n_control]  # get u(t)

            return self._dyn(y, u)

        t_samples = np.sort(self._init_time +
                            (time_horizon - self._init_time) *
                            lhs(1, n_samples).ravel())

        traj = solve_ivp(
            f,
            (self._init_time, time_horizon),
            y0,
            t_eval=t_samples,
            method=self._ode_method,
            rtol=1e-9,
        )

        y = traj.y.T
        t = traj.t.reshape(-1, 1)

        return y0, t, y, control_seq


class TrajectoryDataset(Dataset):
    def __init__(self, generator: TrajectoryGenerator, n_trajectories,
                 n_samples, time_horizon):

        examples = [
            generator.get_example(time_horizon, n_samples)
            for _ in range(n_trajectories)
        ]

        delta = generator._delta
        t0 = generator._init_time
        seq_len = examples[0][-1].size
        self.len = n_samples * n_trajectories

        self.init_state = np.empty((self.len, generator._n))
        self.time = np.empty((self.len, 1))
        self.control_seq = np.empty((self.len, seq_len, 1))
        self.state = np.empty((self.len, generator._n))

        k = 0

        for example in examples:
            x0, t, y, u = example

            for x_, t_ in zip(y, t):
                self.init_state[k] = x0
                self.time[k] = t_
                self.state[k] = x_

                self.control_seq[k] = u
                # pad control sequence
                n_control = int(np.floor((t_ - t0) / delta))
                self.control_seq[k][n_control + 1:] = 0.

                k += 1

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return (self.init_state[index], self.time[index], self.state[index],
                self.control_seq[index])


def pack_model_inputs(x0, t, u, delta):
    t = torch.Tensor(t.reshape((-1, 1)))
    x0 = torch.Tensor(x0.reshape((1, -1))).repeat(t.shape[0], 1)
    u_unrolled = torch.empty((t.shape[0], u.size, 1))

    for t_, u_ in zip(t, u_unrolled):
        u_[:] = torch.Tensor(u)
        n_control = int(np.floor(t_ / delta))
        u_[n_control + 1:] = 0.

    return x0, t, u_unrolled


class GaussianSequence(SequenceGenerator):
    def __init__(self, mean=0., std=1., rng=None):
        super(GaussianSequence, self).__init__(rng)

        self._mean = mean
        self._std = std

    def _sample_impl(self, time_range, delta):
        n_control_vals = int(1 +
                             np.floor((time_range[1] - time_range[0]) / delta))
        control_seq = self._rng.normal(loc=self._mean,
                                       scale=self._std,
                                       size=(n_control_vals, 1))

        return control_seq


class SinusoidalSequence(SequenceGenerator):
    def __init__(self, rng=None):
        super(SinusoidalSequence, self).__init__(rng)

        self._f_mean = 1.0
        self._f_std = 1.0
        self._amp_mean = 1.0
        self._amp_std = 1.0

    def _sample_impl(self, time_range, delta):
        amplitude = self._rng.lognormal(mean=self._amp_mean,
                                        sigma=self._amp_std)
        frequency = self._rng.lognormal(mean=self._f_mean, sigma=self._f_std)

        n_control_vals = int(1 +
                             np.floor((time_range[1] - time_range[0]) / delta))
        time = np.linspace(time_range[0], time_range[1], n_control_vals)

        return (amplitude * np.sin(frequency * time)).reshape((-1, 1))
