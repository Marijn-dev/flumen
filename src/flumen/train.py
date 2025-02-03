import torch
import time
from .experiment import Experiment
import wandb 

def prep_inputs(x0, a0,y,a, u, lengths,Phi,time, device):
    sort_idxs = torch.argsort(lengths, descending=True)

    a0 = a0[sort_idxs]
    a = a[sort_idxs]
    u = u[sort_idxs]
    lengths = lengths[sort_idxs]
    y
    
    deltas = u[:, :lengths[0], -1].unsqueeze(-1)

    u = torch.nn.utils.rnn.pack_padded_sequence(u,
                                                lengths,
                                                batch_first=True,
                                                enforce_sorted=True)

    a0 = a0.to(device)
    a = a.to(device)
    u = u.to(device)
    deltas = deltas.to(device)

    return a0, a, u, deltas

def prep_inputs_(x0,a0, y,a, u, lengths,PHI,time, device):
    sort_idxs = torch.argsort(lengths, descending=True)

    a0 = a0[sort_idxs]
    a = a[sort_idxs]
    u = u[sort_idxs]
    y = y[sort_idxs]
    lengths = lengths[sort_idxs]
    PHI = PHI[sort_idxs]
    deltas = u[:, :lengths[0], -1].unsqueeze(-1)
    time = time[sort_idxs]

    u = torch.nn.utils.rnn.pack_padded_sequence(u,
                                                lengths,
                                                batch_first=True,
                                                enforce_sorted=True)

    a0 = a0.to(device)
    a = a.to(device)
    y = y.to(device)
    u = u.to(device)
    deltas = deltas.to(device)
    PHI = PHI.to(device)
    time = time.to(device)

    return a0, a, u, deltas, y,PHI,time


def validate(data, loss_fn, model, device):
    vl_inner = 0.
    vl_outer = 0.

    with torch.no_grad():
        for example in data:
            a0, a, u, deltas,y,PHI_true,time = prep_inputs_(*example, device)

            a_pred = model(a0, u, deltas)
            vl_inner += loss_fn(a, a_pred).item()

            # y_true = PHI_true[0] @ a[0]
            # y_true = y_true.view(1,100)
                
            y_pred = PHI_true[0] @ a_pred[0]
            y_pred = y_pred.view(1,100)

            vl_outer += loss_fn(y, y_pred).item()

    return model.state_dim * vl_inner / len(data), model.state_dim * vl_outer / len(data)


def train_step(example, loss_fn, model, optimizer, device):
    x0, y, u, deltas = prep_inputs(*example, device)

    optimizer.zero_grad()

    y_pred = model(x0, u, deltas)
    loss = model.state_dim * loss_fn(y, y_pred)

    loss.backward()
    optimizer.step()

    return loss.item()


class EarlyStopping:

    def __init__(self, es_patience, es_delta=0.):
        self.patience = es_patience
        self.delta = es_delta

        self.best_val_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        self.best_model = False

    def step(self, val_loss):
        self.best_model = False

        if self.best_val_loss - val_loss > self.delta:
            self.best_val_loss = val_loss
            self.best_model = True
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True


def train(experiment: Experiment, model, loss_fn, optimizer, sched,
          early_stop: EarlyStopping, train_dl, val_dl, test_dl, device,
          max_epochs):
    
    header_msg = f"{'Epoch':>5} :: {'Loss (Train)':>16} :: " \
        f"{'Loss (Val)':>16} :: {'Loss (Test)':>16} :: {'Best (Val)':>16}"


    # Log the architecture to wandb.config
    wandb.watch(model, log="all", log_graph=True)
    wandb.summary["model"] = model

    print(header_msg)
    print('=' * len(header_msg))
    # Evaluate initial loss
    model.eval()
    train_loss_inner, train_loss_outer = validate(train_dl, loss_fn, model, device)
    val_loss_inner, val_loss_outer = validate(val_dl, loss_fn, model, device)
    test_loss_inner, test_loss_outer = validate(test_dl, loss_fn, model, device)

    wandb.log({"inner/train_loss": train_loss_inner,
                  "inner/val_loss": val_loss_inner,
                  "inner/test_loss": test_loss_inner}, step=0)
    
    wandb.log({"outer/train_loss": train_loss_outer,
                  "outer/val_loss": val_loss_outer,
                  "outer/test_loss": test_loss_outer}, step=0)
    
    early_stop.step(val_loss_inner)
    experiment.register_progress(train_loss_inner, val_loss_inner, test_loss_inner,
                                 early_stop.best_model)
    print(
        f"{0:>5d} :: {train_loss_inner:>16e} :: {val_loss_inner:>16e} :: " \
        f"{test_loss_inner:>16e} :: {early_stop.best_val_loss:>16e}"
    )

    start = time.time()

    for epoch in range(max_epochs):
        model.train()
        for example in train_dl:
            train_step(example, loss_fn, model, optimizer, device)

        model.eval()
        train_loss_inner, train_loss_outer = validate(train_dl, loss_fn, model, device)
        val_loss_inner, val_loss_outer = validate(val_dl, loss_fn, model, device)
        test_loss_inner, test_loss_outer = validate(test_dl, loss_fn, model, device)

        sched.step(val_loss_inner)
        early_stop.step(val_loss_inner)

        print(
            f"{epoch + 1:>5d} :: {train_loss_inner:>16e} :: {val_loss_inner:>16e} :: " \
            f"{test_loss_inner:>16e} :: {early_stop.best_val_loss:>16e}"
        )

        wandb.log({"inner/train_loss": train_loss_inner,
                  "inner/val_loss": val_loss_inner,
                  "inner/test_loss": test_loss_inner,
                  "inner/best_val_loss":early_stop.best_val_loss}, step=epoch+1)
         
        wandb.log({"outer/train_loss": train_loss_outer,
                  "outer/val_loss": val_loss_outer,
                  "outer/test_loss": test_loss_outer}, step=epoch+1)

        if early_stop.best_model:
            experiment.save_model(model)

        experiment.register_progress(train_loss_inner, val_loss_inner, test_loss_inner,
                                     early_stop.best_model)

        if early_stop.early_stop:
            break

    train_time = time.time() - start
    experiment.save(train_time)

    wandb.summary["test loss"] = test_loss_inner
    wandb.summary["train time"] = train_time
    return train_time
