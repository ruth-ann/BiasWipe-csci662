import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import os

def main(input_file, train_file, dev_file, test_file, train_size=0.6, dev_size=0.2, test_size=0.2):
    assert abs(train_size + dev_size + test_size - 1.0) < 1e-6, "Sizes must sum to 1.0"

    # Load Dataframe 
    ext = os.path.splitext(input_file)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(input_file)
    elif ext == ".tsv":
        df = pd.read_csv(input_file, sep="\t")
    else:
        raise ValueError(f"Unknown extension {ext}")

    # Split train / temp
    temp_size = dev_size + test_size
    train_df, temp_df = train_test_split(
        df, test_size=temp_size, stratify=df['is_toxic'], random_state=42
    )

    # Split temp into dev / test
    test_frac_of_temp = test_size / temp_size
    dev_df, test_df = train_test_split(
        temp_df, test_size=test_frac_of_temp, stratify=temp_df['is_toxic'], random_state=42
    )

    # Save files
    train_df.to_csv(train_file, sep="\t", index=False)
    dev_df.to_csv(dev_file, sep="\t", index=False)
    test_df.to_csv(test_file, sep="\t", index=False)

    print(f"Saved {len(train_df)} train rows to {train_file}")
    print(f"Saved {len(dev_df)} dev rows to {dev_file}")
    print(f"Saved {len(test_df)} test rows to {test_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a TSV dataset into stratified train/dev/test sets.")
    parser.add_argument("--input", required=True, help="Path to input TSV file")
    parser.add_argument("--train_out", required=True, help="Path to output train TSV")
    parser.add_argument("--dev_out", required=True, help="Path to output dev TSV")
    parser.add_argument("--test_out", required=True, help="Path to output test TSV")
    parser.add_argument("--train_size", type=float, default=0.6, help="Fraction for train set (default 0.6)")
    parser.add_argument("--dev_size", type=float, default=0.2, help="Fraction for dev set (default 0.2)")
    parser.add_argument("--test_size", type=float, default=0.2, help="Fraction for test set (default 0.2)")

    args = parser.parse_args()

    main(
        args.input, args.train_out, args.dev_out, args.test_out,
        train_size=args.train_size, dev_size=args.dev_size, test_size=args.test_size
    )
