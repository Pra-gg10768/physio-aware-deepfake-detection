from src.preprocessing import run_preprocessing

if __name__ == "__main__":
    run_preprocessing(
    raw_root="data/DFD_original_sequence",
    out_root="data/signals_dfd_test"
)

    print("Pipeline completed.")
