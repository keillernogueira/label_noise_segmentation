import pandas as pd
import numpy as np
import skimage
import shapely.wkt
import cv2
import tqdm

ids = ["AOI_2_Vegas", "AOI_3_Paris", "AOI_4_Shanghai", "AOI_5_Khartoum"]

csv_paths = [
    f"data/spacenet2/{id}/SN2_buildings_train_{id}_solutions.csv" for id in ids
]

print(csv_paths)


def image_mask_resized_from_summary(df, image_id):
    im_mask = np.zeros((650, 650), dtype=np.uint8)

    if len(df[df.ImageId == image_id]) == 0:
        raise RuntimeError("ImageId not found on summaryData: {}".format(image_id))

    for idx, row in df[df.ImageId == image_id].iterrows():
        shape_obj = shapely.wkt.loads(row.PolygonWKT_Pix)
        if shape_obj.exterior is not None:
            coords = list(shape_obj.exterior.coords)
            if coords:
                x = [round(float(pp[0])) for pp in coords]
                y = [round(float(pp[1])) for pp in coords]
                yy, xx = skimage.draw.polygon(y, x, (650, 650))
                im_mask[yy, xx] = 1

                interiors = shape_obj.interiors
                for interior in interiors:
                    coords = list(interior.coords)
                    x = [round(float(pp[0])) for pp in coords]
                    y = [round(float(pp[1])) for pp in coords]
                    yy, xx = skimage.draw.polygon(y, x, (650, 650))
                    im_mask[yy, xx] = 0
    return im_mask


def prep_image_mask(area_id, csv_path):
    print("prep_image_mask for {}".format(area_id))

    df = pd.read_csv(csv_path)
    for image_id in tqdm.tqdm(df.ImageId.unique()):
        im_mask = image_mask_resized_from_summary(df, image_id)
        cv2.imwrite(f"data/spacenet2/labels/{image_id}.png", im_mask)


for id, csv_path in zip(ids, csv_paths):
    prep_image_mask(id, csv_path)
