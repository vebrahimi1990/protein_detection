CFG = {
    "data": {
        "image_dr_s1": r"D:\Projects\multicolor_multiple_instance\patch64\tub.tif",
        "image_dr_s1234": r"D:\Projects\multicolor_multiple_instance\patch64\tub_nuc_er_oth.tif",
        "image_dr_s234": r"D:\Projects\multicolor_multiple_instance\patch64\nuc_er_oth.tif",
        "patch_size": 64,
        "threshold": 0.4,
        "num_bag": 10000,
        "bag_size": 6,
        "ratio": 0.9,
    },
    "data_test": {
        "image_dr": r"\test\Average.tif",
        "save_dr": r"\Denoising-STED",
        "patch_size": 64,
        "num_bag": 4096
    },
    "model": {
        "lr": 0.0001,
        "n_epochs": 200,
        "batch_size": 32,
        "dropout": 0.3,
        "save_dr": r"\model.h5",
        "save_config": r"\Denoising-STED"
    }
}
