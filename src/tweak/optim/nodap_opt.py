"""Optimizer module."""

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup
)

from optim.custom_opt import CustomOpt


class NodapOpt(CustomOpt):
    """Optim wrapper that implements rate."""

    def __init__(self, model_size, factor, warmup, optimizer, step=0):
        """
        Construct an NodapOpt object.
        model_size: hidden size
        factor: coefficient
        #warmup: warm up steps(step ** (-0.5) == step * warmup ** (-1.5) holds when warmup equals step)
        """
        self.optimizer = optimizer
        self._step = step
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

        batch = 32
        self.warmup = 200000 / batch
        self.train_steps = int(200000 / batch)

        self.init_l_rate = 0.002
        self.unit_l_rate = init_l_rate / warmup

    def step(self):
        """Update parameters and rate."""
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        """Implement `lrate` above."""
        if step is None:
            step = self._step
        # return self.factor * \
        #     (self.model_size ** (-0.5) *
        #     min(step ** (-0.5), step * self.warmup ** (-1.5)))
        if step < self.warmup:
            l_rate = step * self.unit_l_rate
        else:
            l_rate = step % self.train_steps ** (-0.98)
        return l_rate

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        state_dict = self.optimizer.state_dict()
        state_dict.update({'step': self._step})
        return state_dict

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict=state_dict)



class TNodapOpt(CustomOpt):
    """Optim wrapper that implements rate."""

    def __init__(self, model, weight_decay, num_train_steps=200000):
        """
        Construct an NodapOpt object.
        model_size: hidden size
        factor: coefficient
        """
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0
            },
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5, eps=1e-9)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=250, num_training_steps=num_train_steps
        )
        self._step = 0

    def step(self):
        """Update parameters and rate."""
        self._step += 1
        self.optimizer.step()
        self.scheduler.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        state_dict = self.optimizer.state_dict()
        state_dict.update({'step': self._step})
        return state_dict

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict=state_dict)

