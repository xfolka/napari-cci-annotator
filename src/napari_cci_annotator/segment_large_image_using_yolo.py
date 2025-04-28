from ultralytics import YOLO
from ultralytics.engine.results import Results
#from csbdeep.utils import Path, normalize

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
import threading
import napari_cci_annotator._id_table as id_table

#import config



class IntGenerator:
    def __init__(self):
        self.lock = threading.Lock()
        self.cnt : np.uint32 = 100

    #def __iter__(self): return self

    def getNext(self):
        with self.lock:
            self.cnt += 1
            return self.cnt

class AbstractSegmenter:
    def __init__(self, img_size):
        self.model_mutex = threading.Lock()
        self.image_size = img_size
        
    def getImageSize(self):
        return self.image_size
    
    
    
# class StarDistSegmenter(AbstractSegmenter):
    
#     def __init__(self, model_name, base_dir, img_size):
#         super(StarDistSegmenter,self).__init__(img_size)
#         self.axis_norm = (0,1)   # normalize channels independently
#         self.model_name = model_name
#         self.model = StarDist2D(None, name=model_name, basedir=base_dir)

    
#     def segment_wrapper(self, data):
#         n_data = normalize(data,1,99.8,axis=self.axis_norm)
#         labels, details = self.model.predict_instances(n_data)
#         return labels
    

class YoloSegmenter(AbstractSegmenter):
    
    def __init__(self, model_path, image_dim):
        super(YoloSegmenter,self).__init__(image_dim)
        self.intGen = IntGenerator()
        self.model = YOLO(model_path, task='segment')
    
    def segment(self, data):
        input_data = np.ascontiguousarray(data)
        results = self.model(source=input_data, imgsz=self.image_size,verbose=False)
        return results
    

    def segment_wrapper(self, data, block_id):
        with self.model_mutex:
            #print(f"segment chunk {block_id}, {data.shape}, {dimension}")
            rgb_data = skimage.color.gray2rgb(data)
            input_data = np.ascontiguousarray(rgb_data)
            result = self.model(source=input_data, imgsz=self.getImageSize(),verbose=False)
 
        computed_result = result#[0].compute()
        all_masks = np.zeros(shape=data.shape, dtype=np.uint32)
        if computed_result is None or computed_result[0].masks is None:
            return all_masks
        
        result_masks = computed_result[0].masks
        masks = result_masks.data.cpu().numpy()
        segments = computed_result[0].masks.shape[0]
        
        sh1 = all_masks.shape[0]
        sh2 = all_masks.shape[1]

        for n in range(segments):
            res =masks[n]
            mask = res * self.intGen.getNext()
            all_masks[:sh1, :sh2] = np.where(all_masks[:sh1, :sh2] == 0, mask[:sh1, :sh2], all_masks[:sh1, :sh2])

        return all_masks


class LargeImageSegmenter:

    def __init__(self):

        self.model_mutex = threading.Lock()
        random.seed()
        self.tableOfIds = id_table.EquivalenceList()


    # def isotrophic_opening(radius,chunk):
    #     bin_chunk = (chunk > 0)
    #     interm = skimage.morphology.isotropic_opening(bin_chunk,radius)
    #     idxs = np.where(interm != 0)
    #     l_chunk = np.zeros(chunk.shape, dtype=chunk.dtype)  # Create array with correct dtype
    #     l_chunk[idxs] = chunk[idxs]
    #     return l_chunk

    # def area_opening(chunk):
    #     bin_chunk = (chunk > 0)
    #     interm = skimage.morphology.area_opening(bin_chunk)
    #     idxs = np.where(interm != 0)
    #     l_chunk = np.zeros(chunk.shape, dtype=chunk.dtype)  # Create array with correct dtype
    #     l_chunk[idxs] = chunk[idxs]
    #     return l_chunk

    # def calculate_dynamic_overlap(large_image_size, ann_size):
    #     n_s = large_image_size // ann_size
    #     rests = large_image_size % ann_size
    #     pixels_for_next_square = ann_size-rests
    #     overlap_to_distribute = pixels_for_next_square / (n_s+1)
        
    #     return overlap_to_distribute/2

    def caclulate_neighbour_equivalence_ids(self,data, block_id, img_size, scan_vertical, scan_far_side = False):

        print(f"calulating eq ids from {block_id}")
        x = 1 if not scan_far_side else data.shape[0] #img_size
        y = 1 if not scan_far_side else data.shape[1]
        
        neighbour_mod = -1 if not scan_far_side else 1
        
        if scan_vertical: 
            neighbour_coords_mod =  (neighbour_mod,0)#(border_distance,0)
            scan_size = data.shape[1]
        else:
            neighbour_coords_mod =  (0,neighbour_mod)#(0,border_distance)
            scan_size = data.shape[0]

        connected_table = defaultdict(lambda: defaultdict(int))
        max_neighbour_local_table = defaultdict(lambda: defaultdict(int))
        #for coord in range(img_size):
        for coord in range(scan_size):
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
            self.tableOfIds.add_eqvivalence_pair(id_l,id_n)
            #print(f"Adding equivalent ids: {id_l} {id_n} {block_id}")
            # print(f"merging with id: {id_l} {id_n} {block_id} {scan_vertical}")
            # idxs = np.where(data == id_l)
            # data[idxs] = id_n

        return data


    def find_and_change_ids_along_border(self,data, block_info = None):    
        
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
            eq_id = self.tableOfIds.get_equivalent_id(id)
            if id != eq_id:
                print(f"setting id: {id} to eq_id: {eq_id}")
                idxs = np.where(data == id)
                data[idxs] = eq_id
        
        return data
        


    # @da.delayed
    # def segment_with_yolo(self,model, data, dimension):
    #     input_data = np.ascontiguousarray(data)
    #     results = model(source=input_data, imgsz=dimension,verbose=False)
    #     return results

    # def segment_wrapper(self,model, dimension, data, block_id):
    #     #data.resize((dimension,dimension),refcheck=False)
    #     with self.model_mutex:
    #         print(f"segment chunk {block_id}, {data.shape}, {dimension}")
    #         rgb_data = skimage.color.gray2rgb(data)
    #         result = self.segment_with_yolo(model,rgb_data,dimension)
    #         computed_result = result.compute()
                
    #     #all_masks = np.zeros(shape=(dimension,dimension), dtype=np.uint32)
    #     all_masks = np.zeros(shape=data.shape, dtype=np.uint32)
    #     if computed_result is None or computed_result[0].masks is None:
    #         return all_masks
        
    #     result_masks = computed_result[0].masks
    #     masks = result_masks.data.cpu().numpy()
    #     segments = computed_result[0].masks.shape[0]
        
        
    #     sh1 = all_masks.shape[0]
    #     sh2 = all_masks.shape[1]
    #     #tmp_id = intGen.getNext()

    #     print(f"all_mask shape {all_masks.shape}, segmented shape {segments}")
    #     for n in range(segments):
    #         #res = skimage.morphology.binary_closing(masks[n])
    #         #res = skimage.morphology.diameter_closing(masks[n],diameter_threshold=5)
    #         res =masks[n]
    #         mask = res * self.intGen.getNext()
    #         all_masks[:sh1, :sh2] = np.where(all_masks[:sh1, :sh2] == 0, mask[:sh1, :sh2], all_masks[:sh1, :sh2])        

    #     return all_masks


    def segment_large_image(self, concrete_segmenter, imagePath, outPath = None):

        img_data = da.array.image.imread(imagePath)
        self.segment_large_image_data(concrete_segmenter, img_data, imgSize=concrete_segmenter.getImageSize())

    def segment_large_image_data(self, concrete_segmenter, imageData, imgSize = 1024, over_lap = 100, iso_radius =4):

        large_image_tmp = da.array.from_array(imageData)
        s = large_image_tmp.shape
        img_size = imgSize#config.IMG_SIZE
        overlap = over_lap#config.OVERLAP
        chunk_size =int(img_size - (2 * overlap))

        large_image = large_image_tmp.reshape((s[0],s[1])).rechunk((chunk_size,chunk_size,1))

        #bound_f = partial(self.segment_wrapper, model, img_size)
        meta = np.empty((chunk_size, chunk_size), dtype=np.uint32)
        segment_results = da.array.map_overlap(concrete_segmenter.segment_wrapper, large_image, meta=meta, chunks=(chunk_size,chunk_size) ,depth=overlap, boundary='reflect', trim=True, allow_rechunk=True)

        dep = 1
        merge_horizontal = partial(self.caclulate_neighbour_equivalence_ids,img_size = chunk_size, scan_vertical = False)
        h1_result = segment_results.map_overlap(merge_horizontal,dtype=np.uint32,depth=dep, boundary='reflect', trim=True, allow_rechunk=True)

        merge_vertical = partial(self.caclulate_neighbour_equivalence_ids,img_size = chunk_size, scan_vertical = True)
        v1_result = h1_result.map_overlap(merge_vertical,dtype=np.uint32,depth=dep, boundary='reflect', trim=True, allow_rechunk=True)

        res = v1_result.compute(scheduler="threads")

        self.tableOfIds.group_ids()

        new_dask_array = da.array.from_array(res)
        s = new_dask_array.shape
        img_size = imgSize#config.IMG_SIZE
        overlap = over_lap#config.OVERLAP
        chunk_size =int(img_size - (2 * overlap))
        final_dask = new_dask_array.reshape((s[0],s[1])).rechunk((chunk_size,chunk_size,1))
    
        end_result = final_dask.map_blocks(self.find_and_change_ids_along_border,dtype=np.uint32)

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