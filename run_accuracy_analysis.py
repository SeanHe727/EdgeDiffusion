from rknn.api import RKNN

RKNN_PATH = "/mnt/d/chenh/model-compression-toolkit/unet_hybrid_v1.rknn"
DATASET_PATH = "/mnt/d/chenh/model-compression-toolkit/calib_npy_b2/dataset_wsl_small.txt"


def main() -> None:
    rknn = RKNN(verbose=True)
    ret = rknn.load_rknn(RKNN_PATH)
    if ret != 0:
        raise SystemExit(f"[ERROR] load_rknn failed: {ret}")

    print("[INFO] accuracy_analysis ...")
    rknn.accuracy_analysis(inputs=DATASET_PATH)

    rknn.release()
    print("[OK] accuracy_analysis done.")


if __name__ == "__main__":
    main()
