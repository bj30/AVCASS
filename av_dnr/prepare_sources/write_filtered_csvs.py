import os
import csv
from tqdm import tqdm
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--metadata_dir', required=True, default='/path/to/VGGSound/metadata' type=str, help='Path to metadata directory')
parser.add_argument('--output_dir', required=True, default='vggsound_filtered' type=str, help='Path to output directory')
parser.add_argument('--filtered_data', default='vgg_stat_filtered.csv' type=str, help='Path to filtered data csv file')
args = parser.parse_args()

filtered_classes = []
with open(args.filtered_data) as f:
    stat_csv = csv.reader(f)
    for row in stat_csv:
        filtered_classes.append(row[0])


splits = ['test', 'train']
for split in splits:
    new_list = []
    with open(os.path.join(args.metadata_dir, split+'.csv')) as csvfile:
        meta_csv = csv.reader(csvfile)
        for row in tqdm(meta_csv):
            if row[1] in filtered_classes:
                new_list.append(row)
    f = open(f"{args.output_dir}/{split}.csv", "w")
    writer = csv.writer(f)
    writer.writerows(new_list) 
    f.close()