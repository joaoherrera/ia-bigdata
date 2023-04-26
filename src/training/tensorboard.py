from torch.utils.tensorboard.writer import SummaryWriter


class TrainingRecorder:
    def __init__(self, summary_filepath: str) -> None:
        self.writer = SummaryWriter(summary_filepath)

    def record_scalar(self, tag: str, value: float, step=None) -> None:
        self.writer.add_scalar(tag, value, step)

    def close(self) -> None:
        self.writer.close()
