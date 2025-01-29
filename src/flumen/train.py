import torch
import time
from .experiment import Experiment
import wandb 

def prep_inputs(x0,a0, y,a, u,u_projected, lengths,lengths_projected,Phi, device):
    sort_idxs = torch.argsort(lengths, descending=True)

    x0 = x0[sort_idxs]
    a0 = a0[sort_idxs]
    y = y[sort_idxs]
    a = a[sort_idxs]
    u = u[sort_idxs]
    u_projected = u_projected[sort_idxs]
    lengths = lengths_projected[sort_idxs]
    Phi = Phi[sort_idxs]
    deltas = u[:, :lengths[0], -1].unsqueeze(-1)

    u = torch.nn.utils.rnn.pack_padded_sequence(u,
                                                lengths,
                                                batch_first=True,
                                                enforce_sorted=True)
    u_projected = torch.nn.utils.rnn.pack_padded_sequence(u_projected,
                                                lengths,
                                                batch_first=True,
                                                enforce_sorted=True)

    x0 = x0.to(device)
    a0 = a0.to(device)
    y = y.to(device) 
    a = a.to(device) 
    u = u.to(device)
    u_projected = u_projected.to(device)
    deltas = deltas.to(device)

    Phi = Phi.to(device)

    return x0, a0,y,a, u,u_projected, deltas, Phi


def validate(data, loss_fn, model, device):
    loss_projected = 0. # inner loss
    loss = 0.           # outer loss

    with torch.no_grad():
        for example in data:
            x0, a0,y,a, u,u_projected, deltas, Phi = prep_inputs(*example, device)

            a_pred = model(a0, u_projected, deltas)
            y_pred = torch.matmul(Phi, a_pred.unsqueeze(-1)).squeeze(-1)  # (102, 100, 51) @ (102, 51, 1) -> (102, 100, 1) -> (102, 100)
            loss_projected += loss_fn(a, a_pred).item()
            loss += loss_fn(y, y_pred).item()
            break
    return model.state_dim * loss / len(data),  model.state_dim * loss_projected / len(data) 


def train_step(example, loss_fn, model, optimizer, device):
    x0, a0,y,a, u,u_projected, deltas, Phi = prep_inputs(*example, device)
    optimizer.zero_grad()

    a_pred = model(a0, u_projected, deltas)
    loss = model.state_dim * loss_fn(a, a_pred)

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

    # inner loss is loss with predicted temporal coefficient a
    # outer loss is loss with respect to data 
    train_loss_outer, train_loss_inner = validate(train_dl, loss_fn, model, device)
    val_loss_outer, val_loss_inner = validate(val_dl, loss_fn, model, device)
    test_loss_outer,test_loss_inner = validate(test_dl, loss_fn, model, device)

    wandb.log({"inner/train_loss": train_loss_inner,
                  "inner/val_loss": val_loss_inner,
                  "inner/test_loss": test_loss_inner,
                  "outer/train_loss": train_loss_outer,
                  "outer/val_loss": val_loss_outer,
                  "outer/test_loss": test_loss_outer}, step=0)
    
    early_stop.step(train_loss_inner)
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
        train_loss_outer,train_loss_inner = validate(train_dl, loss_fn, model, device)
        val_loss_outer,val_loss_inner= validate(val_dl, loss_fn, model, device)
        test_loss_outer,test_loss_inner = validate(test_dl, loss_fn, model, device)

        sched.step(val_loss_inner)
        early_stop.step(val_loss_inner)

        print(
            f"{epoch + 1:>5d} :: {train_loss_inner:>16e} :: {val_loss_inner:>16e} :: " \
            f"{test_loss_inner:>16e} :: {early_stop.best_val_loss:>16e}"
        )

        # log results of epoch to wandb
        wandb.log({"inner/train_loss": train_loss_inner,
                  "inner/val_loss": val_loss_inner,
                  "inner/test_loss": test_loss_inner,
                  "outer/train_loss": train_loss_outer,
                  "outer/val_loss": val_loss_outer,
                  "outer/test_loss": test_loss_outer,
                  "best/val loss":early_stop.best_val_loss},step=epoch + 1)


        if early_stop.best_model:
            experiment.save_model(model)

        experiment.register_progress(train_loss_inner, val_loss_inner, test_loss_inner,
                                     early_stop.best_model)

        if early_stop.early_stop:
            break

    train_time = time.time() - start
    experiment.save(train_time)

    wandb.summary["inner/test loss"] = test_loss_inner
    wandb.summary["outer/test loss"] = test_loss_outer
    wandb.summary["train time"] = train_time
    return train_time
