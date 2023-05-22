import os
import openml

dataset_name = "BTS_Mini"  # BTS_Extended

# Get dataset by name
dataset = openml.datasets.get_dataset(
    f"Meta_Album_{dataset_name}"
)  # download_data=True, download_all_files=True

# Get MARVEL data as a dataframe
marvel_df, _, _, _ = dataset.get_data(dataset_format="dataframe")
print(marvel_df)

# Compute directory where images are stored
img_dir = os.path.join(os.path.dirname(dataset.data_file), dataset_name, "images")

for idx, row in marvel_df.iterrows():
    img_file = os.path.join(img_dir, row["FILE_NAME"])
    img_catg = row["CATEGORY"]

    print(img_file, img_catg)
