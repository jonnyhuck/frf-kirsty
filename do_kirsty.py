import numpy as np
from os import listdir
from affine import Affine
from os.path import exists
from pandas import DataFrame
from geopandas import read_file
from rasterio import open as rio_open
from rasterio.features import rasterize
from rasterio.features import geometry_mask
from FuzzyRF import FuzzyRFTrainer, FuzzyRFGenerator


def prepare_training_data(vector_path, raster_path, field_name, concentration):
    """
    Prep the training data
    """
    #import contaminant data & reproject to projected coordinate system
    points = read_file(vector_path).to_crs(crs="EPSG:21096")

    #intitialise empty lists to store predictor variable names and predictor raster surfaces
    predictors = []
    rasters = [] 
    count = 1

    # initialize arrays to hold training data
    X = []
    y = []

    for file in sorted(listdir(raster_path)):

        #ignore MacOS file
        if file == '.DS_Store':
            continue

        # load raster dataset
        with rio_open(f'{raster_path}{file}') as src:

            #read data
            data = src.read(1)

            #append predictor variable name to list 
            predictors.append(file[:-4])

            if count == 1:

                res = src.res[0]
                bounds = src.bounds
                transform = src.transform
                orig = src

            #add opened raster to a list
            rasters.append(data)
            count += 1

    # faff the data in to the format expected by the classifier
    stacked_array = np.stack(rasters, axis=-1)

    #Predict only for valid pixels (not na)
    points = points.loc[(points[field_name] >= 0)]

    #set true or false depending on contaminant concentration
    points["conc"] = [1 if As[field_name] > concentration else 0 for id, As in points.iterrows()] 

    #loop through sample pointscd f
    for _, point in points.iterrows():

        # create a mask for the current geometry and extract the intersecting pixels
        mask = geometry_mask([point.geometry], transform=src.transform, invert=True, out_shape=(src.height, src.width))
        masked_pixels = stacked_array[mask]

        # append the pixel values and labels to the training arrays
        X.extend(masked_pixels)
        y.extend([point["conc"]] ) # * len(masked_pixels))

    # convert training arrays to numpy arrays
    X = np.array(X)
    y = np.array(y)

    mask = np.any(np.stack([band >= 0 for band in stacked_array]), axis=-1)

    # return 
    return stacked_array, X, y, transform, res, bounds, orig



if __name__ == "__main__":

    contaminants = {
        "Mn": ["Mn_ppm",    0.08,   './kirsty_data/mn/'],
        "As": ["As",        1.0,    './kirsty_data/as/'],
        "Fe": ["Fe_ppm",    0.3,    './kirsty_data/iron/'],
        "F":  ["F",         1.0,    './kirsty_data/F/'],
        "NO3":["NO3_ppm",   45.0,   './kirsty_data/nitrate/']
    }

    VECTOR_PATH = "./kirsty_data/water_quality/collated.shp"

    #load data 
    outline = read_file('./kirsty_data/outlines/uganda.shp').to_crs(crs="EPSG:21096")
    region_data = read_file('./kirsty_data/outlines/uga_admin1.shp').to_crs(crs="EPSG:21096")
    districts = read_file('./kirsty_data/outlines/districts.shp').to_crs(crs="EPSG:21096")
    water = read_file('./kirsty_data/outlines/water_area.shp').to_crs(crs="EPSG:21096")
   
    for key in contaminants.keys():

        FIELD_NAME = contaminants[key][0]
        CONCENTRATION = contaminants[key][1]
        RASTER_PATH = contaminants[key][2]
        ZARR_PATH   = f'./kirsty_data/kirsty_{key}.zarr'

        # prep input data
        print("preparing...")
        prepped = prepare_training_data(VECTOR_PATH, RASTER_PATH, FIELD_NAME, CONCENTRATION)

        #get image properties for setting up affine transformation
        res = prepped[4]
        bounds = prepped[5]
        src = prepped[6]

        affine2 = Affine(res, 0, bounds[0]-3000, 0, -res, bounds[3])

        # train a new model if needed
        if not exists(ZARR_PATH):
            print("training...")
            trainer = FuzzyRFTrainer(prepped[:3], trees=300, branches=7)

            print('saving training data...')
            gen = FuzzyRFGenerator.from_trainer(trainer, ZARR_PATH)

        # otherwise just load existing model params
        else:
            print("no training needed, loading...")
            gen = FuzzyRFGenerator(ZARR_PATH)

        # use it to draw n landscapes
        print("drawing landscapes...")

        # make 100 draws using generator
        landscapes = np.stack([landscape for landscape in gen.mc_draws(100)], axis=0)

        # summarise the draws
        medians = np.median(landscapes, axis=0)
        variances = np.var(landscapes, axis=0)
        means = np.mean(landscapes, axis=0)
        stdev = np.std(landscapes, axis=0)

        # rasterize the buffer (we don't want to mask as it will involve operations on the lcm dataset directly
        outer_mask = rasterize(outline.geometry, (gen.rows, gen.cols), fill=0, transform=affine2, default_value=1, all_touched=True)
        outer_mask2 = rasterize(outline.geometry, (gen.rows, gen.cols), fill=-999.0, transform=affine2, default_value=0, all_touched=True)
        water_mask = rasterize(water.geometry, (gen.rows, gen.cols), fill=1, transform=affine2, default_value=0, all_touched=True)
        water_mask2 = rasterize(water.geometry, (gen.rows, gen.cols), fill=0, transform=affine2, default_value=-999.0, all_touched=True)

        clip = np.zeros((gen.rows, gen.cols))
        clip2 = np.zeros((gen.rows, gen.cols))

        clip += (means[0] * (outer_mask * water_mask))
        clip += (outer_mask2 + water_mask2)

        clip2 += (means[1] * (outer_mask * water_mask))
        clip2 += (outer_mask2 + water_mask2)

        clip[clip < 0] = -999
        clip2[clip2 < 0] = -999

        # write means to file
        with rio_open(f"./kirsty_data/out/{key}_mean.tif", 'w', driver='GTiff', width=gen.cols, height=gen.rows, count=2, 
                    dtype=np.float32, nodata=-999, transform=affine2, crs={'init': "EPSG:21096"}) as out:
            out.write_band(1, clip.astype(np.float32))
            out.write_band(2, clip2.astype(np.float32))


        clip = np.zeros((gen.rows, gen.cols))
        clip2 = np.zeros((gen.rows, gen.cols))

        clip += (stdev[0] * (outer_mask * water_mask))
        clip += (outer_mask2 + water_mask2)

        clip2 += (stdev[1] * (outer_mask * water_mask))
        clip2 += (outer_mask2 + water_mask2)

        clip[clip < 0] = -999
        clip2[clip2 < 0] = -999

        # write means to file
        with rio_open(f"./kirsty_data/out/{key}_std.tif", 'w', driver='GTiff', width=gen.cols, height=gen.rows, count=2, 
                    dtype=np.float32, nodata=-999, transform=affine2, crs={'init': "EPSG:21096"}) as out:
            out.write_band(1, clip.astype(np.float32))
            out.write_band(2, clip2.astype(np.float32))

        #set. up dataframe
        cols = list(np.unique(region_data["adm1_name"]))
        cols.append("Total")
        results = DataFrame(columns=cols)    
        total = []

        #open population data
        with rio_open('./kirsty_data/population/pop_gen2.tif') as dst:
                
            affineP = Affine(dst.res[0], 0, dst.bounds[0], 0, -dst.res[0], dst.bounds[3])

            #read data
            population = dst.read(1)

            #create mask from region extent
            mask = rasterize(outline.geometry, (gen.rows, gen.cols), fill=0, transform=affine2, default_value=1, all_touched=True)

            #mask population
            total_pop = population * mask

            #get geometry of the study area - buffer to correct any topological problems
            minx, miny, maxx, maxy = outline.total_bounds

            #get bounds of study area
            tl_img = dst.index(minx, maxy)
            br_img = dst.index(maxx, miny)
            w, h = br_img[1]-tl_img[1], br_img[0]-tl_img[0]

            #loop through landscapes
            for landscape in landscapes:        

                    #mask landscape to region of interest
                    l = landscape[1] * mask

                    #initialise counters for total population
                    total_ug = 0
                    contaminated_ug = 0

                    #loop over cells
                    for r in range(gen.rows):
                        for c in range(gen.cols):

                            if r >= l.shape[0] or c >= l.shape[1]:
                                continue

                            if r < 0 or c < 0 or r >= total_pop.shape[0] or c >= total_pop.shape[1] or total_pop[(r,c)] < 0:
                                continue

                            total_ug += total_pop[(r,c)]

                            #if probability above 0.5
                            if l[(r,c)] > 0.5:

                                contaminated_ug += total_pop[(r,c)] #np_sum(pop_slice)


                    total.append(((contaminated_ug/total_ug))*100)


            results["Total"] = total

            #loop through uganda regions
            for id,region in region_data.iterrows():

                #get geometry of the study area - buffer to correct any topological problems
                bounds = region.geometry.bounds

                #get bounds of study area                
                tl_img = dst.index(bounds[0], bounds[3])
                br_img = dst.index(bounds[2], bounds[1])
                w, h = br_img[1]-tl_img[1], br_img[0]-tl_img[0]

                #initialise list to store results
                vals = []

                #create mask from region extent
                mask = rasterize([region.geometry], (gen.rows, gen.cols), fill=0, transform=affine2, default_value=1, all_touched=True)
                #mask2 = rasterize([region.geometry], (dst.shape[0], dst.shape[1]), fill=0, transform=affineP, default_value=1, all_touched=True)

                #mask population data to region of interest
                pop = population * mask

                #loop through landscapes 
                for landscape in landscapes:

                    #mask landscape to region of interest
                    l2 = landscape[1] * mask

                    #initialise counters at zero
                    region_pop = 0
                    contaminated_pop = 0

                    #loop through each cell of landscape
                    for r in range(tl_img[0], br_img[0]):
                        for c in range(tl_img[1], br_img[1]):


                            #if out of bounds, ignore
                            if r >= l2.shape[0] or c >= l2.shape[1]:
                                continue

                            #if less than zero or nan, ignore
                            if pop[(r,c)] < 0 or l[(r,c)] < 0 or np.isnan(pop[(r,c)]):
                                continue

                            #counter for total regional population
                            region_pop += pop[(r,c)]

                            #if exceeds cut off value, add to counter for pop living in contaminated areas
                            if l2[(r,c)] > 0.5:

                                contaminated_pop += pop[(r,c)]

                    vals.append(((contaminated_pop/region_pop))*100)

                #only use this when calculating regional stats
                results[str(region["adm1_name"])] = np.array(vals)

                #loop through uganda regions

            for id,region in districts.iterrows():

                #get geometry of the study area - buffer to correct any topological problems
                bounds = region.geometry.bounds

                #get bounds of study area                
                tl_img = dst.index(bounds[0], bounds[3])
                br_img = dst.index(bounds[2], bounds[1])
                w, h = br_img[1]-tl_img[1], br_img[0]-tl_img[0]

                #initialise list to store results
                vals = []

                #create mask from region extent
                mask = rasterize([region.geometry], (gen.rows, gen.cols), fill=0, transform=affine2, default_value=1, all_touched=True)

                #mask population data to region of interest
                pop = population * mask

                #loop through landscapes 
                for landscape in landscapes:

                    #mask landscape to region of interest
                    l2 = landscape[1] * mask

                    #initialise counters at zero
                    region_pop = 0
                    contaminated_pop = 0

                    #loop through each cell of landscape
                    for r in range(tl_img[0], br_img[0]):
                        for c in range(tl_img[1], br_img[1]):


                            #if out of bounds, ignore
                            if r >= l2.shape[0] or c >= l2.shape[1]:
                                continue

                            #if less than zero or nan, ignore
                            if pop[(r,c)] < 0 or l[(r,c)] < 0 or np.isnan(pop[(r,c)]):
                                continue

                            #counter for total regional population
                            region_pop += pop[(r,c)]

                            #if exceeds cut off value, add to counter for pop living in contaminated areas
                            if l2[(r,c)] > 0.5:

                                contaminated_pop += pop[(r,c)]

                    vals.append(((contaminated_pop/region_pop))*100)

                    #print(region["adm1_name"], region_pop)

                districts.loc[(districts["adm2_name"] == region["adm2_name"]), "pop_count"] = region_pop
                    
                #only use this when calculating district stats   
                districts.loc[(districts["adm2_name"] == region["adm2_name"]), f"mean_{key}"] = np.mean(vals)
                districts.loc[(districts["adm2_name"] == region["adm2_name"]), f"std_{key}"] = np.std(vals)                
                       
        #save regional results
        results.to_csv(f'./kirsty_data/out/{key}_regional.csv')

        #save district results
        districts.to_file(f'./kirsty_data/out/{key}_districts.shp')