# george
# must set up rclone with google drive as gdrive.
import ee
import time
import os


ee.Initialize(project="ee-pes")

# get meghalaya boundary from admin boundary dataset
def download_boundary(dir, name):
    raise Exception("Config EE first")
    admin = ee.FeatureCollection("FAO/GAUL/2015/level1").select("ADM1_NAME")
    boundary = admin.filter(ee.Filter.eq("ADM1_NAME", name))

    # export meghalaya boundary
    try:
        os.system("rclone delete gdrive:/boundary")
    except:
        pass 

    task = ee.batch.Export.table.toDrive(
        collection=boundary,
        folder="boundary",
        description="boundary",
        fileFormat="shp"
    )

    task.start()

    time.sleep(60*4)
    os.system("rclone copy gdrive:/boundary " + dir + "/data/" + name.lower() + "/boundary")

    # download DSM
    dsm = ee.ImageCollection("JAXA/ALOS/AW3D30/V3_2").select("DSM").mosaic()

    # clip DSM to boundary with 120km buffer
    boundary_buffer = boundary.geometry().buffer(120_000)
    dsm = dsm.clip(boundary_buffer)

    # export DSM
    try: 
        os.system("rclone delete gdrive:/DSM.tif")
    except:
        pass

    task = ee.batch.Export.image.toDrive(
        image=dsm,
        description="DSM",
        fileFormat="GeoTIFF",
        region=boundary_buffer,
        maxPixels=1e13,
        scale=30,
    )

    task.start()

    time.sleep(60*20)
    os.system("rclone copyto gdrive:/DSM.tif " + dir + "/data/" + name.lower() + "/DSM.tif")

if __name__ == "__main__":
    download_boundary(name="Meghalaya", dir=os.environ.get('DeepForestcast_PATH'))

