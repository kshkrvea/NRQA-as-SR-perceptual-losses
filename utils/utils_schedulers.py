import math
import warnings
from torch.optim.lr_scheduler import _LRScheduler

class CosineCycleAnnealingWarmRestarts(_LRScheduler):

    def __init__(
        self, 
        optimizer, 
        start_value: float,
        end_value: float,
        cycle_size: int,
        cycle_mult: float = 1.0,
        start_value_mult: float = 1.0,
        end_value_mult: float = 1.0,
        last_epoch=-1,
        verbose=False
    ):
        self.start_value = start_value
        self.end_value = end_value
        self.cycle_size = int(cycle_size)  # Ensure cycle_size is integer
        self.cycle_mult = cycle_mult

        self.start_value_mult = start_value_mult
        self.end_value_mult = end_value_mult
        self.event_index = 0

        if self.cycle_size < 2:
            raise ValueError(f"Argument cycle_size should be positive and larger than 1, but given {cycle_size}")
        
        super(CosineCycleAnnealingWarmRestarts, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        
        cycle_progress = self.event_index / self.cycle_size
        return [self.start_value + ((self.end_value - self.start_value) / 2) * (1 - math.cos(math.pi * cycle_progress))
                for base_lr in self.base_lrs]

    def step(self, epoch=0):

        if self.event_index != 0 and self.event_index % self.cycle_size == 0:
            self.event_index = 0
            self.cycle_size = int(self.cycle_size * self.cycle_mult)
            self.start_value *= self.start_value_mult
            self.end_value *= self.end_value_mult
                
        class _enable_get_lr_call:

            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        with _enable_get_lr_call(self):
            for i, data in enumerate(zip(self.optimizer.param_groups, self.get_lr())):
                param_group, lr = data
                param_group['lr'] = lr
                self.print_lr(self.verbose, i, lr)

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
        
        self.event_index += 1
        
        
