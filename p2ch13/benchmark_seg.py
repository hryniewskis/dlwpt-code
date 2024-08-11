import torch
import torch.optim
from torch.utils.data import DataLoader

from p2ch13.dsets import TrainingLuna2dSegmentationDataset
from p2ch13.train_seg import LunaTrainingApp
from util.logconf import logging

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)


class BenchmarkLuna2dSegmentationDataset(TrainingLuna2dSegmentationDataset):
    def __len__(self):
        # return 500
        return 5000
        return 1000


class LunaBenchmarkApp(LunaTrainingApp):
    def initTrainDl(self):
        train_ds = BenchmarkLuna2dSegmentationDataset(
            val_stride=10,
            isValSet_bool=False,
            contextSlices_count=3,
            # augmentation_dict=self.augmentation_dict,
        )

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return train_dl

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        train_dl = self.initTrainDl()

        for epoch_ndx in range(1, 2):
            log.info(
                "Epoch {} of {}, {}/{} batches of size {}*{}".format(
                    epoch_ndx,
                    self.cli_args.epochs,
                    len(train_dl),
                    len([]),
                    self.cli_args.batch_size,
                    (torch.cuda.device_count() if self.use_cuda else 1),
                )
            )

            self.doTraining(epoch_ndx, train_dl)


if __name__ == "__main__":
    LunaBenchmarkApp().main()
