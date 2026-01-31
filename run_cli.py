import argparse
from gigp.pipeline import run_pipeline

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--make_splits", action="store_true")
    ap.add_argument("--full_csv", type=str, default="train.csv")

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--epochs_linear", type=int, default=10)
    ap.add_argument("--epochs_nograph", type=int, default=10)
    ap.add_argument("--epochs_gigp", type=int, default=10)

    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--enc_layers", type=int, default=4)
    ap.add_argument("--dec_layers", type=int, default=4)
    ap.add_argument("--ffn_dim", type=int, default=2048)
    ap.add_argument("--dropout", type=float, default=0.2)

    ap.add_argument("--eval_max_len", type=int, default=64)
    ap.add_argument("--lat_warmup", type=int, default=30)
    ap.add_argument("--lat_runs", type=int, default=200)

    args = ap.parse_args()

    run_pipeline(
        data_dir=args.data_dir,
        use_cuda=args.cuda,
        make_splits=args.make_splits,
        full_csv=args.full_csv,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        epochs_linear=args.epochs_linear,
        epochs_nograph=args.epochs_nograph,
        epochs_gigp=args.epochs_gigp,
        d_model=args.d_model,
        nhead=args.nhead,
        enc_layers=args.enc_layers,
        dec_layers=args.dec_layers,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        eval_max_len=args.eval_max_len,
        lat_warmup=args.lat_warmup,
        lat_runs=args.lat_runs,
    )

if __name__ == "__main__":
    main()
