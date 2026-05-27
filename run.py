from main import main
import argparse


def get_params():
    parser = argparse.ArgumentParser(description="RSCD_PyTorch")
    parser.add_argument("--batch_size", type=int, default=32, metavar="N",
                        help="input batch size for training (default: 32)")
    parser.add_argument("--lr", type=float, default=0.0001, metavar="LR",
                        help="learning rate (default: 0.0001)")

    parser.add_argument("--epochs", type=int, default=200,
                        help="number of training epochs (default: 200)")
    parser.add_argument("--input_h", type=int, default=256,
                        help="input height for FLOPs calculation (default: 256)")
    parser.add_argument("--input_w", type=int, default=256,
                        help="input width for FLOPs calculation (default: 256)")

    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    params = vars(get_params())
    main(params)
