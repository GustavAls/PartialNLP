import argparse
import trainer
import os
from MapNN import MapNN
from toy_datasets import ToyRegressionDDataModule


def main(args):
    data = ToyRegressionDDataModule(batch_size=1)
    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()
    map_model = MapNN(input_size=1, width=1, size=3, output_size=1, non_linearity="silu")

    trained_model = trainer.train(
        map_model,
        dataloader_train=train_dataloader,
        dataloader_val=val_dataloader,
        device=args.device,
        save_path=os.path.join(args.output_path, "map_1d_net.pt"),
        epochs=args.num_epochs,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run training and or evaluation of Semantic Similarity Model"
    )
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_epochs", type=int, default=50)

    args = parser.parse_args()
    main(args)
