import numpy as np
import torch
from torch.utils.data import Dataset


class RawTrajectoryDataset(Dataset):

    def __init__(self,
                 data,
                 state_dim,
                 control_dim,
                 output_dim,
                 delta,
                 output_mask,
                 noise_std=0.,
                 **kwargs):
        self.__dict__.update(kwargs)

        n_traj = len(data)
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.output_dim = output_dim
        self.delta = delta
        self.mask = output_mask

        self.init_state = torch.empty(
            (n_traj, self.state_dim)).type(torch.get_default_dtype())
        self.init_state_noise = torch.empty(
            (n_traj, self.state_dim)).type(torch.get_default_dtype())

        self.time = []
        self.state = []
        self.state_noise = []
        self.control_seq = []
        self.U = [] # left singular matrix 
        self.S = [] # Singular values
        self.V = [] # right singular matrix
        self.a_initial = [] # a(0)
        self.A = [] # time coefficients A = [a(0)...a(tend)] with a(0) = [a0(0)...aN(0)]
        self.control_seq_projected = [] # control sequences projected with spatial modes
        self.W = [] 

        ## sample represents trajectory
        for k, sample in enumerate(data):
            self.init_state[k] = torch.from_numpy(sample["init_state"].reshape(
                (1, self.state_dim)))
            self.init_state_noise[k] = 0.
            self.time.append(
                torch.from_numpy(sample["time"]).type(
                    torch.get_default_dtype()).reshape((-1, 1)))

            self.state.append(
                torch.from_numpy(sample["state"]).type(
                    torch.get_default_dtype()).reshape((-1, self.state_dim)))

            self.state_noise.append(
                torch.normal(mean=0.,
                             std=noise_std,
                             size=self.state[-1].size()))

            self.control_seq.append(
                torch.from_numpy(sample["control"]).type(
                    torch.get_default_dtype()).reshape((-1, self.control_dim)))
            
            # Compute SVD (spatial modes)
            state_space_time = sample["state"].reshape(self.state_dim,-1)
            time_length = len(state_space_time[0])
            U_, S_, V_ = np.linalg.svd(state_space_time,full_matrices=False)
            
            self.U.append(
                torch.from_numpy(U_).type(torch.get_default_dtype()))
            
            self.S.append(
                torch.from_numpy(S_).type(torch.get_default_dtype()))
            
            self.V.append(
                torch.from_numpy(V_).type(torch.get_default_dtype()))
            
            # Compute projection (temporal coefficients)
            A_ = np.transpose(U_) @ state_space_time
            A_ = A_.reshape(time_length,-1) # reshape to time,space from space,time
            self.a_initial.append(
                torch.from_numpy(A_[0]).type(torch.get_default_dtype())) # A0
            self.A.append(
                torch.from_numpy(A_).type(torch.get_default_dtype()))
            
            # projection of input U
            Br = np.transpose(U_) @ np.ones((state_dim,self.control_dim)) # mxr -> r x control dim
            self.control_seq_projected.append(
                torch.from_numpy(np.transpose(Br @ np.transpose(sample["control"]))).type(torch.get_default_dtype()))
            
            # Projection back to W (original space)
            self.W.append(
                torch.from_numpy((U_@(A_.reshape(-1,time_length))).reshape(time_length,-1)).type(torch.get_default_dtype()))
            
        self.dim_galerkin = U_.shape[1] # amount of columns  
        self.galerkin = True
        if self.galerkin: # change dimensions x -> a
            self.state_dim = self.dim_galerkin
            self.control_dim = self.dim_galerkin
            self.output_dim = self.dim_galerkin

    @classmethod
    def generate(cls, generator, time_horizon, n_trajectories, n_samples,
                 noise_std):

        def get_example():
            x0, t, y, u = generator.get_example(time_horizon, n_samples)
            return {
                "init_state": x0,
                "time": t,
                "state": y,
                "control": u,
            }

        data = [get_example() for _ in range(n_trajectories)]

        return cls(data,
                   *generator.dims(),
                   delta=generator._delta,
                   output_mask=generator._dyn.mask,
                   generator=generator,
                   noise_std=noise_std)

    def __len__(self):
        return len(self.init_state)

    def __getitem__(self, index):
        return (self.init_state[index], self.init_state_noise[index],
                self.time[index], self.state[index], self.state_noise[index],
                self.control_seq[index],self.U[index],self.S[index],self.V[index],self.a_initial[index],self.A[index],self.control_seq_projected[index],self.W[index]) 


class TrajectoryDataset(Dataset):

    def __init__(self,
                 raw_data: RawTrajectoryDataset,
                 max_seq_len=-1,
                 n_samples=1):
       
        self.control_dim = raw_data.control_dim
        self.state_dim = raw_data.state_dim 
        self.output_dim = raw_data.state_dim
        mask = tuple(bool(v) for v in raw_data.mask)

        self.delta = raw_data.delta


        init_state = []
        init_a = [] # a(0)
        state = []
        a = [] 
        rnn_input_data = []
        rnn_input_data_projected = []
        seq_len_data = []
        seq_len_data_projected = []

        Phi = []

        rng = np.random.default_rng()
        self.galerkin = raw_data.galerkin
        k_tr = 0
        for (x0, x0_n, t, y, y_n, u,phi,S,V,a_initial,A,u_proj,W) in raw_data:
            y += y_n
            x0 += x0_n

            if max_seq_len == -1:
                for k_s, (a_s, y_s) in enumerate(zip(A,y)):
                    rnn_input, rnn_input_len = self.process_example(
                        0, k_s, t, u, self.delta)
                    rnn_input_projected, rnn_input_len_projected = self.process_example(
                        0, k_s, t, u_proj, self.delta)

                    mask_galerkin = mask[raw_data.dim_galerkin]
                    a_ = a_s.view(1, -1)[:, mask_galerkin].reshape(-1)
                    s = y_s.view(1, -1)[:, mask].reshape(-1)

                    init_state.append(x0)
                    init_a.append(a_initial)
                    state.append(s)
                    a.append(a_)
                    seq_len_data.append(rnn_input_len)
                    seq_len_data_projected.append(rnn_input_len_projected)
                    rnn_input_data.append(rnn_input)
                    rnn_input_data_projected.append(rnn_input_projected)

                    Phi.append(phi)

            else:
                for k_s, y_s in enumerate(y):
                    # find index of last relevant state sample
                    times = (t - t[k_s] - max_seq_len * self.delta)
                    times[times > 0] = 0.
                    k_l = times.argmax().item()

                    if k_l == k_s:
                        end_idxs = (0, )
                    else:
                        end_idxs = rng.choice(k_l - k_s,
                                              size=min(n_samples, k_l - k_s),
                                              replace=False)

                    for k_e in end_idxs:
                        rnn_input, rnn_input_len = self.process_example(
                            k_s, k_s + k_e, t, u, self.delta)

                        init_state.append(y_s)
                        state.append(y[k_s + k_e, mask])
                        seq_len_data.append(rnn_input_len)
                        rnn_input_data.append(rnn_input)

        self.init_state = torch.stack(init_state).type(
            torch.get_default_dtype())
        self.init_a = torch.stack(init_a).type(
            torch.get_default_dtype())
        self.state = torch.stack(state).type(torch.get_default_dtype())
        self.a = torch.stack(a).type(torch.get_default_dtype())
        self.rnn_input = torch.stack(rnn_input_data).type(
            torch.get_default_dtype())
        self.rnn_input_projected = torch.stack(rnn_input_data_projected).type(
            torch.get_default_dtype())
        self.seq_lens = torch.tensor(seq_len_data, dtype=torch.long)
        self.seq_lens_projected = torch.tensor(seq_len_data_projected, dtype=torch.long)

        # basis vectors 
        self.Phi = torch.stack(Phi).type(
            torch.get_default_dtype())
        
        self.len = len(init_state)
        self.len_projected = len(init_a)

    @staticmethod
    def process_example(start_idx, end_idx, t, u, delta):
        init_time = 0.

        u_start_idx = int(np.floor((t[start_idx] - init_time) / delta))
        u_end_idx = int(np.floor((t[end_idx] - init_time) / delta))
        u_sz = 1 + u_end_idx - u_start_idx

        u_seq = torch.zeros_like(u)
        u_seq[0:u_sz] = u[u_start_idx:(u_end_idx + 1)]

        deltas = torch.ones((u_seq.shape[0], 1))
        t_u_end = init_time + delta * u_end_idx
        t_u_start = init_time + delta * u_start_idx

        if u_sz > 1:
            deltas[0] = (1. - (t[start_idx] - t_u_start) / delta).item()
            deltas[u_sz - 1] = ((t[end_idx] - t_u_end) / delta).item()
        else:
            deltas[0] = ((t[end_idx] - t[start_idx]) / delta).item()

        deltas[u_sz:] = 0.

        rnn_input = torch.hstack((u_seq, deltas))

        return rnn_input, u_sz

    def __len__(self):
        if self.galerkin:
            return self.len_projected
        else:
            return self.len

    def __getitem__(self, index):
        return (self.init_state[index], self.init_a[index],self.state[index],self.a[index],
                self.rnn_input[index],self.rnn_input_projected[index], self.seq_lens[index],self.seq_lens_projected[index],self.Phi[index])
