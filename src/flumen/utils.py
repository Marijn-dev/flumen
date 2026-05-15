import torch
import datetime


def get_loss(which):
    if which == "mse":
        return torch.nn.MSELoss()
    elif which == "l1":
        return torch.nn.L1Loss()
    else:
        raise ValueError(f"Unknown loss {which}.")


def print_header():
    header_msg = (
        f"{'Epoch':>5} :: {'Loss (Train)':>16} :: "
        f"{'Loss (Val)':>16} :: {'Best (Val)':>16}"
    )

    print(header_msg)
    print("=" * len(header_msg))


def print_losses(epoch: int, train: float, val: float, best_val_yet: float):
    print(
        f"{epoch + 1:>5d} :: {train:>16e} :: {val:>16e} :: {best_val_yet:>16e}"
    )


def get_timestamp() -> str:
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    ts = now.strftime("%y%m%d_%H%M")

    return ts


def print_gpu_info():
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print(f"CUDA is available, {n_gpus} devices can be used.")
        current_dev = torch.cuda.current_device()

        for id in range(n_gpus):
            msg = f"Device {id}: {torch.cuda.get_device_name(id)}"

            if id == current_dev:
                msg += " [Current]"

            print(msg)


def get_batch_inputs(x0, t, u, delta: float):
    x0, u, skips, tau = pack_model_inputs(x0, t, u, delta)

    tau_seq = torch.ones((u.shape[0], u.shape[1], 1))
    tau_seq[range(tau_seq.shape[0]), skips] = tau
    rnn_input = torch.cat((u, tau_seq), dim=-1)

    lengths = skips.add(1)
    if lengths.ndim == 0:
        lengths.unsqueeze(0)

    return x0, rnn_input, tau, lengths


def get_batch_inputs_packed(x0, t, u, delta: float):
    x0, rnn_input, tau, lengths = get_batch_inputs(x0, t, u, delta)

    rnn_input = torch.nn.utils.rnn.pack_padded_sequence(
        rnn_input, lengths, batch_first=True, enforce_sorted=False
    )

    return x0, rnn_input, tau


def pack_model_inputs(x0, t, u, delta: float, parameter=None):
    t = torch.Tensor(t)
    x0 = torch.Tensor(x0)
    u = torch.Tensor(u)
    parameter = (
        torch.Tensor(parameter).unsqueeze(0) if parameter is not None else None
    )
    if x0.ndim < 2:
        x0 = x0.unsqueeze(0)
        u = u.unsqueeze(0)

    skips = torch.floor(t / delta).long()
    tau = (t - delta * skips) / delta

    return x0, u, skips.squeeze(), tau, parameter
