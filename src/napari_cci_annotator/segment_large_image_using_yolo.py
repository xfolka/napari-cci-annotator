from ultralytics import YOLO
from ultralytics.engine.results import Results

from PIL import Image
import numpy as np
import dask as da
from dask.array import image
#import dask.bag as db
from dask import array
from pathlib import Path
import random
from functools import partial
from threading import Lock
#from shapely.affinity import translate
import threading
import skimage.color
import skimage.morphology
from timeit import default_timer as timer
from collections import defaultdict

import napari_cci_annotator._id_table as id_table

#import config

model_mutex = threading.Lock()

random.seed()

tableOfIds = id_table.EquivalenceList()

import threading
class IntGenerator:
    def __init__(self):
        self.lock = threading.Lock()
        self.cnt : np.uint32 = 100

    def __iter__(self): return self

    def getNext(self):
        self.lock.acquire()
        try:
            self.cnt += 1
            return self.cnt
        finally:
            self.lock.release()

intGen = IntGenerator()

def isotrophic_opening(radius,chunk):
    bin_chunk = (chunk > 0)
    interm = skimage.morphology.isotropic_opening(bin_chunk,radius)
    idxs = np.where(interm != 0)
    l_chunk = np.zeros(chunk.shape, dtype=chunk.dtype)  # Create array with correct dtype
    l_chunk[idxs] = chunk[idxs]
    return l_chunk

def area_opening(chunk):
    bin_chunk = (chunk > 0)
    interm = skimage.morphology.area_opening(bin_chunk)
    idxs = np.where(interm != 0)
    l_chunk = np.zeros(chunk.shape, dtype=chunk.dtype)  # Create array with correct dtype
    l_chunk[idxs] = chunk[idxs]
    return l_chunk

def calculate_dynamic_overlap(large_image_size, ann_size):
    n_s = large_image_size // ann_size
    rests = large_image_size % ann_size
    pixels_for_next_square = ann_size-rests
    overlap_to_distribute = pixels_for_next_square / (n_s+1)
    
    return overlap_to_distribute/2

def merge_border_segments(data, block_id, img_size, scan_vertical, scan_far_side = False):
    
    if data.shape[0] <= img_size and scan_vertical:
        return data
    if data.shape[1] <= img_size and not scan_vertical:
        return data

    print(f"computing chunk {block_id}")
    x = 1 if not scan_far_side else img_size
    y = 1 if not scan_far_side else img_size
    
    neighbour_mod = -1 if not scan_far_side else 1
    
    if scan_vertical: 
        neighbour_coords_mod =  (neighbour_mod,0)#(border_distance,0)
    else:
        neighbour_coords_mod =  (0,neighbour_mod)#(0,border_distance)

    connected_table = defaultdict(lambda: defaultdict(int))
    max_neighbour_local_table = defaultdict(lambda: defaultdict(int))
    for coord in range(img_size):
        if scan_vertical:
            y = coord 
        else:
            x = coord

        #construct table that holds which structures have most connecting pixels to the neighbour structure
        local_indices     = (x, y)
        neighbour_indices = (x + neighbour_coords_mod[0], y + neighbour_coords_mod[1])
        id_local = data[local_indices]
        id_neighbour =  data[neighbour_indices]
        if  id_local != 0 and id_neighbour != 0 and id_neighbour != id_local:
            connected_table[id_local][id_neighbour] += 1
            max_neighbour_local_table[id_neighbour][id_local] += 1 
            
    #only keep the maximum values per neighbour
    neighbour_max = {}
    for outer_key in max_neighbour_local_table.keys():
        max_cnt = 0
        id = max_neighbour_local_table[outer_key]
        id = {k: v for k, v in id.items() if v == max(id.values())}
        neighbour_max[outer_key] = list(id.keys())[0]
                
    for n in neighbour_max.keys():
        loc = neighbour_max[n]
        for l in connected_table.keys():
            if l != loc: #we know this is NOT the local id that should be set to neighbour
                connected_table[l][n] = 0

    filtered = []
    for idx, outer_key in enumerate(connected_table.keys()):
        max_cnt = 0
        filtered.append((outer_key,0))
        for inner_key in connected_table[outer_key].keys():
            value = connected_table[outer_key][inner_key]
            if value > max_cnt:
                max_cnt = value
                filtered[idx] = (outer_key,inner_key)
                
    #merge data in local and neighbour structures
    for (l,n) in filtered:
        id_l = l
        id_n = n
        tableOfIds.add_eqvivalence_pair(id_l,id_n)
        print(f"Adding equivalent ids: {id_l} {id_n} {block_id}")
        # print(f"merging with id: {id_l} {id_n} {block_id} {scan_vertical}")
        # idxs = np.where(data == id_l)
        # data[idxs] = id_n

    return data


def find_and_change_ids_along_border(data, block_info = None):    
    
    d1 = data.shape[0]
    d2 = data.shape[1]
    id_set = set()
    
    for y in [0,d2-1]:
        for x in range(d1):
            local_indices = (x, y)
            id_local = data[local_indices]
            if id_local != 0:
                id_set.add(id_local)
        
    for x in [0, d1-1]:
        for y in range(d2):
            local_indices = (x, y)
            id_local = data[local_indices]
            if id_local != 0:
                id_set.add(id_local)
        
    for id in id_set:
        eq_id = tableOfIds.get_equivalent_id(id)
        if id != eq_id:
            print(f"setting id: {id} to eq_id: {eq_id}")
            idxs = np.where(data == id)
            data[idxs] = eq_id
    
    return data
    


@da.delayed
def segment_with_yolo(model, data, dimension):
    input_data = np.ascontiguousarray(data)
    results = model(source=input_data, imgsz=dimension)
    return results

def segment_wrapper(model, dimension, data, block_id):
    with model_mutex:
        print(f"computing chunk {block_id}, {data.shape}")

        rgb_data = skimage.color.gray2rgb(data)
        
        result = segment_with_yolo(model,rgb_data,dimension)
        computed_result = result.compute()
             
    all_masks = np.zeros(shape=(dimension,dimension), dtype=np.uint32)
    if computed_result is None or computed_result[0].masks is None:
        return all_masks
    
    result_masks = computed_result[0].masks
    masks = result_masks.data.cpu().numpy()
    shape = computed_result[0].masks.shape
    
    sh1 = shape[1]
    sh2 = shape[2]
    #tmp_id = intGen.getNext()

    for n in range(shape[0]):
        
        #res = skimage.morphology.binary_closing(masks[n])
        #res = skimage.morphology.diameter_closing(masks[n],diameter_threshold=5)
        res =masks[n]
        mask = res * intGen.getNext()
        all_masks[:sh1, :sh2] = np.where(all_masks[:sh1, :sh2] == 0, mask, all_masks[:sh1, :sh2])

    

    return all_masks

#base = os.getcwd()
#base =  os.path.dirname(__file__)

#out_data_dir = config.RESULT_OUTPUT_DIR
#Path(out_data_dir).mkdir(parents=True, exist_ok=True)


def segment_large_image(model, imagePath, outPath = None):

    img_data = da.array.image.imread(imagePath)
    segment_large_image_data(model, img_data)

def segment_large_image_data(model, imageData, imgSize = 1024, over_lap = 100, iso_radius =4):

    large_image_tmp = da.array.from_array(imageData)
    s = large_image_tmp.shape
    img_size = imgSize#config.IMG_SIZE
    overlap = over_lap#config.OVERLAP
    chunk_size =int(img_size - (2 * overlap))

    large_image = large_image_tmp.reshape((s[0],s[1])).rechunk((chunk_size,chunk_size,1))
    #large_image = large_image_tmp.rechunk((chunk_size,chunk_size,1))

    bound_f = partial(segment_wrapper, model, img_size)
    segment_results = da.array.map_overlap(bound_f, large_image, dtype=np.uint32, chunks=(chunk_size,chunk_size) ,depth=overlap, boundary='reflect', trim=True, allow_rechunk=True)

    # isoRadius = iso_radius#config.ISO_OPENING_RADIUS
    # iso_opening_f = partial(isotrophic_opening, isoRadius)
    # iso_opening_results = segment_results.map_blocks(iso_opening_f, dtype=np.uint32,chunks=(img_size,img_size))
    
    #area_radius = config.AREA_OPENING_RADIUS
    #area_opening_f = partial(area_opening)
    #area_opening_results = segment_results.map_blocks(area_opening, dtype=np.uint32,chunks=(chunk_size,chunk_size))

    #dep ={0: (2,2),1: (2,2)}
    dep = 1
    merge_horizontal = partial(merge_border_segments,img_size = chunk_size, scan_vertical = False)
    #h1_result = area_opening_results.map_overlap(merge_horizontal,dtype=np.uint32,depth=dep, boundary="reflect", trim=True)
    h1_result = segment_results.map_overlap(merge_horizontal,dtype=np.uint32,depth=dep, boundary="reflect", trim=True)

    # merge_horizontal = partial(merge_border_segments,img_size = chunk_size, scan_vertical = False, scan_far_side = True)
    # h2_result = h1_result.map_overlap(merge_horizontal,dtype=np.uint32,depth=dep, boundary="reflect", trim=True)

    merge_vertical = partial(merge_border_segments,img_size = chunk_size, scan_vertical = True)
    v1_result = h1_result.map_overlap(merge_vertical,dtype=np.uint32,depth=dep, boundary="reflect", trim=True)

    # merge_vertical = partial(merge_border_segments,img_size = chunk_size, scan_vertical = True, scan_far_side = True)
    # v2_result = v1_result.map_overlap(merge_vertical,dtype=np.uint32,depth=dep, boundary="reflect", trim=True)
    res = v1_result.compute()

    tableOfIds.group_ids()

    new_dask_array = da.array.from_array(res)
    s = new_dask_array.shape
    img_size = imgSize#config.IMG_SIZE
    overlap = over_lap#config.OVERLAP
    chunk_size =int(img_size - (2 * overlap))
    final_dask = new_dask_array.reshape((s[0],s[1])).rechunk((chunk_size,chunk_size,1))


    end_result = final_dask.map_blocks(find_and_change_ids_along_border,dtype=np.uint32)

    print("starting...")
    
    start = timer()
    # result = v1_result.compute()
    result = end_result.compute()
#    result = combined_result.compute(scheduler='single-threaded')
    end = timer()
    
    print("stopping: ",end - start)
    
    return result
    # save_im = Image.fromarray(result)
    
    # if outPath:
    #     outdir = outPath
    # else:
    #     outdir = os.path.dirname(imagePath)
        
    # outfile = "segment_" + os.path.basename(imagePath)#config.RESULT_OUTPUT_MASK_IMAGE
    # save_im.save(outdir + "/" + outfile)
    # print(f"Image mask saved as {outfile}")
    
    # data = {
    #     "file_name" : os.path.basename(imagePath),
    #     "chunk_size" : chunk_size,
    #     "overlap" : overlap,
    #     "image_size" : img_size,
    #     "iso_opening_radius" : iso_radius
    # }

    # # Convert the dictionary to JSON
    # json_data = json.dumps(data, indent=4)

    # # Define the file path
    # file_path = "metadata_" + os.path.basename(imagePath) + ".json"

    # # Save the JSON to a file
    # with open(outdir + "/" + file_path, "w") as file:
    #     file.write(json_data)
    

# def segment_large_image_directory(path, outPath=None, modelPath=None):
#     # img_dl_path = config.DL_PATH + "images/"
#     # annot_dl_path = config.DL_PATH + "annotations/"

#     # Path(config.DL_PATH).mkdir(parents=True, exist_ok=True)
#     # Path(img_dl_path).mkdir(parents=True, exist_ok=True)
#     # Path(annot_dl_path).mkdir(parents=True, exist_ok=True)
#     # if config.CLEAR_OUTPUT_DIR:
    
#     if modelPath:
#         model_file_path = modelPath + "/" + config.MODEL_SAVE_FILE_NAME
#     else:
#         model_file_path = config.MODEL_SAVE_DIR + "/" + config.MODEL_SAVE_FILE_NAME
    
#     model = YOLO(model_file_path)
    
#     files = glob.glob(path + "/*.png")
#     for f in files:
#         segment_large_image(model, f, outPath)
        
        
# base =  os.path.dirname(__file__)
# segment_large_image_directory(base + "/../../images",base + "/../../images",base + "/../../models/myelin")