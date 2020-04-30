# dataset_downloads
#    | coco
#    |    | zips
#    |    |    | train2017
#    |    |    | test2017.zip
#    |    |    | val2017.zip
#    |    |    | annotations_trainval2017.zip

unzip dataset_downloads/coco/zips/"*".zip -d dataset_downloads/coco
python -m ssd.scripts.create_coco_tfrecords -d dataset_downloads/coco -o ssd/data/dataset_files/coco