from skimage.morphology import remove_small_objects, isotropic_erosion
from time import sleep
from IPython.display import clear_output
from skimage.morphology import convex_hull_image
from skimage.measure import find_contours
from scipy import ndimage

import myelin_morphometrics as morpho
import warnings
from skimage.measure import label, regionprops_table
import pandas as pd
from skimage.color import label2rgb
import numpy as np
import xlsxwriter
import copy
from .segment_large_image_using_yolo import IntGenerator


from scipy.ndimage import center_of_mass

def bbox_from_BWimg(BW):
    # convex hull calculation including smoothing to avoid pixelation effects
    chull_img = convex_hull_image(BW)
    chull_contour = find_contours(chull_img)
    if len(chull_contour) != 1:
        raise ValueError("More than one contour")

    convex_hull = morpho.smoothBoundary(chull_contour[0])
    min_box_corners, min_box_edges, rot_mat = morpho.minBoundingBox(convex_hull, 'width')

    min_box_corners = min_box_corners - np.mean(min_box_corners, axis=0)
    #   move the box to the same reference as the inpurt convex hull
    c_mass = center_of_mass(chull_img, labels=None, index=None)
    min_box_corners = min_box_corners + c_mass


    return min_box_corners, min_box_edges, rot_mat

def get_box_with_rotation(bw_img, rot_mat):
    chull_img = convex_hull_image(bw_img)
    chull_contour = find_contours(chull_img)
    contours = len(chull_contour)
    if contours!=1:
        return contours, None, None 
    convex_hull = morpho.smoothBoundary(chull_contour[0])
    r_hull = morpho.rotate_points(convex_hull, rot_mat)
    box_dim = morpho.bbox_dimenssions(r_hull)

    min_bbox_points = np.array([
        [0, 0],
        [box_dim[0], 0],
        [box_dim[0], box_dim[1]],
        [0, box_dim[1]]
    ])

    #   rotate the box
    min_bbox_corners = morpho.rotate_points(min_bbox_points, rot_mat.T)
    #   move the box to 0,0
    min_bbox_corners = min_bbox_corners - np.mean(min_bbox_corners, axis=0)
    #   move the box to the same reference as the inpurt convex hull
    c_mass = center_of_mass(chull_img, labels=None, index=None)
    min_bbox_corners = min_bbox_corners + c_mass

    return contours, min_bbox_corners, box_dim

def get_label_for_coord(label_image, x, y):
    if label_image is None:
        return 0
    return label_image[x,y]

def extract_labeled_region(label_image, label):
    # Create a binary mask for the target
    # For labeled images: mask where pixel == target label
    mask = (label_image == label).astype(bool)
    
    # Find bounding box coordinates
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return None  # No target found
    
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    # Crop the image and mask to the bounding box
    cropped_image = label_image[y_min:y_max+1, x_min:x_max+1]
    cropped_mask = mask[y_min:y_max+1, x_min:x_max+1]
    
    # Apply mask to exclude non-target regions
    result = cropped_image * cropped_mask  # Retain only target label
    
    return result, cropped_mask
    

def morpho_data_generator(myelin_label_image, axon_label_image, data_image):
    
    properties = ['label', 'bbox','image','centroid']
    
    reg_table = regionprops_table(label_image=myelin_label_image,
                            properties=properties)
    
    num_labels = len(np.unique(myelin_label_image)) - 1
    print(f"Number of labels in layer: {len(reg_table['label'])} or {num_labels}")
    
    reg_table = pd.DataFrame(reg_table)
    totCnt = 0
    axon_max_label = np.max(axon_label_image)

    for index, row in reg_table.iterrows():

        min_r = row['bbox-0']
        min_c = row['bbox-1']
        max_r = row['bbox-2']
        max_c = row['bbox-3']

        myelin_lbl = copy.copy(myelin_label_image[min_r:max_r,min_c:max_c])
        myelin_bw = row['image']
        img_data = data_image[min_r:max_r,min_c:max_c]
        
        #get the axon label image (if there is one)
        x = row["centroid-0"]
        y = row["centroid-1"]
        axon_label = get_label_for_coord(axon_label_image,int(x),int(y))
        axon_bw_crop = None
        axon_lbl_crop = None
        if axon_label != 0:
            axon_lbl_crop = copy.copy(axon_label_image[min_r:max_r,min_c:max_c])
            axon_bw_crop = (axon_lbl_crop == axon_label).astype(bool)
            axon_lbl_crop = axon_lbl_crop * axon_bw_crop

        print(f"calculated {totCnt}:th morphometric!")
        totCnt += 1        
        yield get_morphos_for_label_data(myelin_lbl,myelin_bw, axon_lbl_crop, axon_bw_crop, img_data, axon_max_label, min_r, min_c)
        
        
def auto_calculate_axon_region(myelin_bw_fill, myelin_bw):
    auto_axon_region = np.logical_xor(myelin_bw_fill,myelin_bw)
    if np.sum(auto_axon_region)>= 100:
        return auto_axon_region
    
    warnings.warn("The axon was most probably not closed.")
    #recalculate the region differently
    myelin_bw_fill = convex_hull_image(myelin_bw_fill)
    auto_axon_region = np.logical_xor(myelin_bw_fill,myelin_bw)
    auto_axon_region = isotropic_erosion(auto_axon_region, 3)
    auto_axon_region = remove_small_objects(auto_axon_region, min_size=500)
    if np.sum(auto_axon_region)<500:
        #print("after axon sum...")
        return None

    return auto_axon_region
    
def get_morphos_for_label_data(myelin_lbl, myelin_bw, axon_lbl, axon_bw, img_data, axon_max_label, x_offset = 0, y_offset = 0):
    
    pad_size = 5
    myelin_lbl[np.logical_not(myelin_bw)] = 0

    properties = ['label','area','eccentricity','solidity','intensity_mean',
                    'axis_minor_length','axis_major_length','feret_diameter_max', 'centroid']
    props_table = regionprops_table(label_image=myelin_lbl,
                                    intensity_image=img_data,
                                    properties=properties)
    obj_table = pd.DataFrame(props_table)
    obj_table.rename(columns={'label' : 'myelin_label',
                        'area': 'myelin_area', 
                        'eccentricity': 'myelin_eccentricity',
                        'solidity': 'myelin_solidity',
                        'intensity_mean': 'myelin_intensity_mean',
                        'axis_minor_length': 'myelin_axis_minor_length',
                        'axis_major_length': 'myelin_axis_major_length',
                        'feret_diameter_max': 'myelin_feret_diameter_max'}, inplace=True)

    obj_table['nr_of_parts'] = 1
    obj_table['status_ok'] = True
    obj_table['error_msg'] = ""
    if len(obj_table.axes[0]) > 1:
        obj_table['nr_of_parts'] = obj_table.axes[0]
        obj_table['status_ok'] = False
        obj_table['error_msg'] = "More than one part"
        # yield obj_table
        # continue
        return obj_table
    
    #TODO: do this in rename above?
    obj_table['center_x'] = props_table['centroid-0'] + x_offset
    obj_table['center_y'] = props_table['centroid-1'] + y_offset

    # Create the padded myelin_bw image to avoid edge effects in the contours
    myelin_bw_fill = ndimage.binary_fill_holes(myelin_bw)

    debug = False
    if axon_bw is None:
        axon_region = auto_calculate_axon_region(myelin_bw_fill,myelin_bw)
        #debug = True
        obj_table['axon_calculation'] = "Automatic"
        #print("no axon here")
    else:
        obj_table['axon_calculation'] = "Annotation"
        axon_region = axon_bw

    if axon_region is None:
        obj_table['axon_calculation'] = "None/Error"
        obj_table['axon_area'] = 0
        obj_table['status_ok'] = False
        obj_table['error_msg'] = "0 Axon area"
        return obj_table
        
        
    else:
        axon_table = get_morphos_for_axon_label_data(axon_region,axon_lbl,obj_table, axon_max_label,pad_size)
        
    obj_table['myelin_hole_area'] = obj_table['axon_area']
        
    myelin_filled_area = np.sum(myelin_bw_fill)
    obj_table['myelin_filled_area'] =myelin_filled_area

    myelin_bw_pad = np.pad(myelin_bw_fill, ((pad_size, pad_size), (pad_size, pad_size),), mode='constant', constant_values=0)
    # convex hull calculation including smoothing to avoid pixelation effects
    try:
        myelin_min_box_corners, myelin_min_box_edges, rot_mat = bbox_from_BWimg(myelin_bw_pad)
        # using the bbox as feret calculator and AR
        obj_table['myelin_feret_max'] = np.max(myelin_min_box_edges)
        obj_table['myelin_feret_min'] = np.min(myelin_min_box_edges)
        obj_table['myelin_AR'] = np.max(myelin_min_box_edges)/np.min(myelin_min_box_edges)
        # calculating myelin median width
        obj_table['myelin_width'] = morpho.get_width(myelin_bw)
    except Exception as e:
        obj_table['status_ok'] = False
        obj_table['error_msg'] = str(e)
        return obj_table

    if axon_region is None:
        obj_table['myelin_width_min_feret_direction'] = "skipped"
        obj_table['myelin_width_max_feret_direction'] = "skipped"    
        return obj_table

    try:
        contours, max_axon_bbox, max_axon_size = get_box_with_rotation(axon_region, rot_mat.T)
    except Exception as e:
        obj_table['status_ok'] = False
        obj_table['error_msg'] = str(e)
        return obj_table
    
    if contours != 1:
        obj_table['myelin_width_min_feret_direction'] = "skipped"
        obj_table['myelin_width_max_feret_direction'] = "skipped"    
    else:
        myelin_width_min_feret = np.min(myelin_min_box_edges)-np.min(max_axon_size)#type: ignore
        myelin_width_max_feret = np.max(myelin_min_box_edges)-np.max(max_axon_size)#type: ignore
        obj_table['myelin_width_min_feret_direction'] = myelin_width_min_feret/2
        obj_table['myelin_width_max_feret_direction'] = myelin_width_max_feret/2
    
    return obj_table

axon_id_generator = IntGenerator()

def get_morphos_for_axon_label_data(axon_bw_img, axon_lbl_img, obj_table, axon_max_label, pad_size = 5, debug = False):

    # # I do not expect holes in the axon image
    if axon_lbl_img is None:
        axon_id_generator.setStartValue(axon_max_label)
        axon_lbl = axon_id_generator.getNext()
    else:
        properties = ['label']
        props_table = regionprops_table(label_image=axon_lbl_img,
                                    properties=properties)
        axon_lbl = props_table['label']
    
    obj_table['axon_label'] = axon_lbl
    axon_bw_img_tmp = ndimage.binary_fill_holes(axon_bw_img)
    axon_bw_pad = np.pad(axon_bw_img_tmp, ((pad_size, pad_size), (pad_size, pad_size),), mode='constant', constant_values=0)

    # # convex hull calculation including smoothing to avoid pixelation effects
    axon_min_box_corners, axon_min_box_edges, _ = bbox_from_BWimg(axon_bw_pad)

    # # using the bbox as feret calculator and AR
    obj_table['axon_area'] = np.sum(axon_bw_pad)
    obj_table['axon_feret_max'] = np.max(axon_min_box_edges)
    obj_table['axon_feret_min'] = np.min(axon_min_box_edges)
    obj_table['axon_AR'] = np.max(axon_min_box_edges)/np.min(axon_min_box_edges)

    return obj_table

            
    
def create_morpho_table_from_data(name, myelin_label_image, axon_label_image, data_image):


    total_table = pd.DataFrame()
    tot_cnt = 0
    for data in morpho_data_generator(myelin_label_image,axon_label_image, data_image):
        total_table = pd.concat([total_table, data], axis=0)
        tot_cnt += 1
        
    total_table.to_excel(name, index=False, engine='xlsxwriter')

