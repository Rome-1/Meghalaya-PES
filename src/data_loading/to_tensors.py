import time
import os
from PIL import Image
from data_loading.process_functions import create_tnsors_pixels

Image.MAX_IMAGE_PIXELS = None

def to_tensors(years, region, sourcepath, wherepath, nightlight_metrics=["median", "maximum"]):
    if not os.path.exists(wherepath):
        os.makedirs(wherepath)

    for year in years:
        start = time.time()
        static, image, pixels, labels = create_tnsors_pixels(
            year=year,
            latest_yr=years[-1],
            tree_p=30,
            cutoffs=[2, 5, 8],
            sourcepath=sourcepath,
            rescale=True,
            wherepath=wherepath,
            nightlight_metrics=nightlight_metrics,
        )
        print("Total time (in seconds) needed to create tensors: ", time.time() - start)
