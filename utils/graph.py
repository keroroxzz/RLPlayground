from torch.utils.tensorboard import SummaryWriter

class GraphPoint:
    """
    A (or multiple) data point(s) to be recorded in a graph on TensorBoard.
        - name: {section name}/{graph name}, ex: Train/loss, to create a graph named "loss" under "Train" section 
        - iter: should be a int to specify the x position of the value
        - value: could be a int/float value or a dict[str, int|float] to create a single/multiple data lines
    """
    def __init__(
            self, 
            name:str, 
            iter:int|float, 
            value:int|float|dict[str, int|float]):
        self.name = name
        self.iter = iter
        self.value = value
    
    def addTo(self, writer: SummaryWriter):
        if isinstance(self.value, int) or isinstance(self.value, float):
            writer.add_scalar(self.name, self.value, self.iter)
        elif isinstance(self.value, dict):
            writer.add_scalars(self.name, self.value, self.iter)