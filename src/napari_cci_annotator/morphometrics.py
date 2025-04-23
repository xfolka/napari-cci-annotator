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
    #assert len(chull_contour)==1, "we only expect one contour"
    contours = len(chull_contour)
    if contours!=1:
        return contours, None, None 
    convex_hull = morpho.smoothBoundary(chull_contour[0])
    r_hull = morpho.rotate_points(convex_hull, rot_mat)
    box_dim = morpho.bbox_dimenssions(r_hull)

    #print(box_dim)

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
    min_bbox_corners = min_bbox_corners + c_mass #((np.max(convex_hull, axis=0)-np.min(convex_hull, axis=0))/2)+np.min(convex_hull, axis=0)

    return contours, min_bbox_corners, box_dim


def get_morphos_for_label(label,label_image, data_image):
    lbl = np.where(label_image == label, label, 0)
    properties = ['bbox','image']
    reg_table = regionprops_table(label_image=label_image,
                            properties=properties)
    
    if len(reg_table) == 0:
        return None
    
    row = reg_table[0]
    min_r = row['bbox-0']
    min_c = row['bbox-1']
    max_r = row['bbox-2']
    max_c = row['bbox-3']

    myelin_lbl = label_image[min_r:max_r,min_c:max_c]
    myelin_bw = row['image']
    img_data = data_image[min_r:max_r,min_c:max_c]
    
    return get_morphos_for_label_data(myelin_lbl,myelin_bw,img_data,label)
    

def morpho_data_generator(label_image, data_image):
    pad_size = 5

    properties = ['label', 'bbox','image','centroid']
    
    reg_table = regionprops_table(label_image=label_image,
                            properties=properties)
    
    num_labels = len(np.unique(label_image)) - 1
    print(f"Number of labels in layer: {len(reg_table['label'])} or {num_labels}")
    
    reg_table = pd.DataFrame(reg_table)
    #reg_table.sample(5)

    openCnt = 0
    totCnt = 0


#with wk.webknossos_context(token=AUTH_TOKEN, timeout=WK_TIMEOUT):
    for index, row in reg_table.iterrows():
        #print(row)
        #obj_idx = row['label']
        totCnt += 1
    # bbox = skibbox2wkbbox(row.to_dict(), pSize)
    # img_data = img_layer.get_finest_mag().read(absolute_offset=bbox.topleft, size=bbox.size)

    # myelin_lbl = lbl_layers[myelin_idx].get_finest_mag().read(absolute_offset=bbox.topleft, size=bbox.size).squeeze()
    # axon_lbl   = lbl_layers[axon_idx].get_finest_mag().read(absolute_offset=bbox.topleft, size=bbox.size).squeeze()
    # mito_lbl   = lbl_layers[mito_idx].get_finest_mag().read(absolute_offset=bbox.topleft, size=bbox.size).squeeze()
    # dyst_lbl   = lbl_layers[dystrophic_idx].get_finest_mag().read(absolute_offset=bbox.topleft, size=bbox.size).squeeze()

        min_r = row['bbox-0']
        min_c = row['bbox-1']
        max_r = row['bbox-2']
        max_c = row['bbox-3']

        myelin_lbl = label_image[min_r:max_r,min_c:max_c]
        myelin_bw = row['image']
        img_data = data_image[min_r:max_r,min_c:max_c]
        yield get_morphos_for_label_data(myelin_lbl,myelin_bw,img_data,row['label'])
        
        
def get_morphos_for_label_data(myelin_lbl, myelin_bw, img_data, label):
        
    # get area of the imge identified as myelin, this is my main ROI which defines behaviour of all others.
#        myelin_bw = morpho.get_BW_from_lbl(myelin_lbl, obj_idx)
    # clean myelin label map, in case other neurons are close by
    myelin_lbl[np.logical_not(myelin_bw)] = 0

    properties = ['label','area','eccentricity','solidity','intensity_mean',
                    'axis_minor_length','axis_major_length','feret_diameter_max', 'centroid']
    props_table = regionprops_table(label_image=myelin_lbl,
                                    intensity_image=img_data,
                                    properties=properties)
    obj_table = pd.DataFrame(props_table)
    obj_table.rename(columns={'area': 'myelin_area', 
                        'eccentricity': 'myelin_eccentricity',
                        'solidity': 'myelin_solidity',
                        'intensity_mean': 'myelin_intensity_mean',
                        'axis_minor_length': 'myelin_axis_minor_length',
                        'axis_major_length': 'myelin_axis_major_length',
                        'feret_diameter_max': 'myelin_feret_diameter_max'}, inplace=True)

    obj_table['nr_of_parts'] = 1
    obj_table['status_ok'] = True
    obj_table['error_msg'] = ""
    ##assert len(obj_table.axes[0]) == 1, "we expected to have a single object"
    if len(obj_table.axes[0]) > 1:
        obj_table['nr_of_parts'] = obj_table.axes[0]
        obj_table['status_ok'] = False
        obj_table['error_msg'] = "More than one part"
        # yield obj_table
        # continue
        return obj_table
    
    #TODO: do this in rename above?
    obj_table['center_x'] = props_table['centroid-0']
    obj_table['center_y'] = props_table['centroid-1']

    # Create the padded myelin_bw image to avoid edge effects in the contours
    myelin_bw_fill = ndimage.binary_fill_holes(myelin_bw)
    # create the "axon area", this is the internal area of the myelin region
    auto_axon_region = np.logical_xor(myelin_bw_fill,myelin_bw)
    if np.sum(auto_axon_region)<100:
        openCnt += 1
    #    continue
        warnings.warn("The axon was most probably not closed.")
        myelin_bw_fill = convex_hull_image(myelin_bw_fill)
        auto_axon_region = np.logical_xor(myelin_bw_fill,myelin_bw)
        auto_axon_region = isotropic_erosion(auto_axon_region, 3)
        auto_axon_region = remove_small_objects(auto_axon_region, min_size=500)
        #assert np.sum(auto_axon_region)>500, f"unexpected problems with the auto axon region {obj_idx}"
        if np.sum(auto_axon_region)>500:
            obj_table['status_ok'] = False
            obj_table['error_msg'] = "Axon region > 500"
            # yield obj_table
            # continue
            return obj_table

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
        # yield obj_table
        # continue
        return obj_table

    auto_axon_area = np.sum(auto_axon_region)
    obj_table['myelin_hole_area'] =auto_axon_area

    try:
        contours, max_axon_bbox, max_axon_size = get_box_with_rotation(auto_axon_region, rot_mat.T)
    except Exception as e:
        obj_table['status_ok'] = False
        obj_table['error_msg'] = str(e)
        # yield obj_table
        # continue
        return obj_table
    
    if contours != 1:
        obj_table['myelin_width_min_feret_direction'] = "skipped"
        obj_table['myelin_width_max_feret_direction'] = "skipped"    
    else:
        myelin_width_min_feret = np.min(myelin_min_box_edges)-np.min(max_axon_size)
        myelin_width_max_feret = np.max(myelin_min_box_edges)-np.max(max_axon_size)
        obj_table['myelin_width_min_feret_direction'] = myelin_width_min_feret/2
        obj_table['myelin_width_max_feret_direction'] = myelin_width_max_feret/2
    
    #yield obj_table
    return obj_table

    # # cleaning axon label map
    # axon_bw = axon_lbl>0
    # axon_bw = np.logical_and(myelin_bw_fill, axon_bw)
    # user_def_axon = np.any(axon_bw)

    # if not(user_def_axon):
    #     print('no user defined axon, we will create one based on the myelin')
    #     axon_bw = auto_axon_region 
    # assert np.any(axon_bw), "The axon label map is empty: does not contain any True values"
    
    # # I do not expect holes in the axon image
    # axon_bw = ndimage.binary_fill_holes(axon_bw)
    # axon_bw_pad = np.pad(axon_bw, ((pad_size, pad_size), (pad_size, pad_size),), mode='constant', constant_values=0)

    # # convex hull calculation including smoothing to avoid pixelation effects
    # axon_min_box_corners, axon_min_box_edges, _ = bbox_from_BWimg(axon_bw_pad)

    # # using the bbox as feret calculator and AR
    # obj_table['axon_area'] = np.sum(axon_bw_pad)
    # obj_table['axon_feret_max'] = np.max(axon_min_box_edges)
    # obj_table['axon_feret_min'] = np.min(axon_min_box_edges)
    # obj_table['axon_AR'] = np.max(axon_min_box_edges)/np.min(axon_min_box_edges)

    # #check the mito labels
    # mito_bw = mito_lbl>0
    # mito_bw = np.logical_and(myelin_bw_fill, mito_bw)
    # user_def_mito = np.any(mito_bw)


    # if not(user_def_mito):
    #     print('no mito in the neuron')
    #     mito_number = 0
    #     mito_total_area = 0
    # else:
    #     mito_lbl = label(mito_bw)
    #     mito_number = np.max(mito_lbl)
    #     mito_total_area = np.sum(mito_bw)

    # obj_table['mito_total_area'] = mito_total_area
    # obj_table['mito_number'] = mito_number


    # # cleaning the distrophic label:
    # # cleaning axon label map
    # dyst_bw = dyst_lbl>0
    # dyst_bw = np.logical_and(myelin_bw_fill, dyst_bw)
    # is_dystrophic = np.any(dyst_bw)
    # obj_table['is_dystrophic'] = is_dystrophic
    # # for later plot
    # axon_lbl[np.logical_not(axon_bw)] = 0
    # axon_lbl[axon_bw] = obj_table['label'] + 1

    # mito_lbl[np.logical_not(mito_bw)] = 0
    # mito_lbl[mito_bw] = obj_table['label'] + 2

    # dyst_lbl[dyst_lbl>0] = obj_table['label'] + 3

    # color = label2rgb(myelin_lbl.T + axon_lbl.T + mito_lbl.T + dyst_lbl.T, image=img_data.squeeze().T, bg_label=0)

    # # Plot the image
    # clear_output(wait=True)
    # # Create a figure and axis for plotting
    # fig, ax = plt.subplots(figsize=(8, 8))

    # ax.imshow(color)
    # tmp = np.vstack((myelin_min_box_corners, myelin_min_box_corners[0,:]))
    # ax.plot(tmp[:, 0]-pad_size, tmp[:, 1]-pad_size, linewidth=2)

    # tmp = np.vstack((max_axon_bbox, max_axon_bbox[0,:]))
    # ax.plot(tmp[:, 0], tmp[:, 1], linewidth=2)

    # tmp = np.vstack((axon_min_box_corners, axon_min_box_corners[0,:]))
    # ax.plot(tmp[:, 0]-pad_size, tmp[:, 1]-pad_size, linewidth=2)



    # ax.set_title(f'Image {obj_idx}')

    # fig.canvas.draw()
    # # Display the plot (necessary for Jupyter Notebooks)
    # png_name = f'myelin_label_{obj_idx}.png'
    # plt.savefig(out_folder.joinpath(png_name))
    # plt.show()
            
    
def create_morpho_table_from_data(name, label_image, data_image):


#     total_table = []
#     pad_size = 5

#     properties = ['label', 'bbox','image']
#     reg_table = regionprops_table(label_image=label_image,
#                             properties=properties)
    
#     num_labels = len(np.unique(label_image)) - 1
#     print(f"Number of labels in layer: {len(reg_table['label'])} or {num_labels}")
    
#     reg_table = pd.DataFrame(reg_table)
#     #reg_table.sample(5)

#     openCnt = 0
#     totCnt = 0


# #with wk.webknossos_context(token=AUTH_TOKEN, timeout=WK_TIMEOUT):
#     for index, row in reg_table.iterrows():
#         #print(row)
#         obj_idx = row['label']
#         totCnt += 1
#     # bbox = skibbox2wkbbox(row.to_dict(), pSize)
#     # img_data = img_layer.get_finest_mag().read(absolute_offset=bbox.topleft, size=bbox.size)

#     # myelin_lbl = lbl_layers[myelin_idx].get_finest_mag().read(absolute_offset=bbox.topleft, size=bbox.size).squeeze()
#     # axon_lbl   = lbl_layers[axon_idx].get_finest_mag().read(absolute_offset=bbox.topleft, size=bbox.size).squeeze()
#     # mito_lbl   = lbl_layers[mito_idx].get_finest_mag().read(absolute_offset=bbox.topleft, size=bbox.size).squeeze()
#     # dyst_lbl   = lbl_layers[dystrophic_idx].get_finest_mag().read(absolute_offset=bbox.topleft, size=bbox.size).squeeze()

#         min_r = row['bbox-0']
#         min_c = row['bbox-1']
#         max_r = row['bbox-2']
#         max_c = row['bbox-3']

#         myelin_lbl = label_image[min_r:max_r,min_c:max_c]
#         myelin_bw = row['image']
#         img_data = data_image[min_r:max_r,min_c:max_c]
        
#         # get area of the imge identified as myelin, this is my main ROI which defines behaviour of all others.
# #        myelin_bw = morpho.get_BW_from_lbl(myelin_lbl, obj_idx)
#         # clean myelin label map, in case other neurons are close by
#         myelin_lbl[np.logical_not(myelin_bw)] = 0

#         properties = ['label', 'area','eccentricity','solidity','intensity_mean',
#                         'axis_minor_length','axis_major_length','feret_diameter_max']
#         props_table = regionprops_table(label_image=myelin_lbl,
#                                         intensity_image=img_data,
#                                         properties=properties)
#         obj_table = pd.DataFrame(props_table)
#         obj_table.rename(columns={'area': 'myelin_area', 
#                             'eccentricity': 'myelin_eccentricity',
#                             'solidity': 'myelin_solidity',
#                             'intensity_mean': 'myelin_intensity_mean',
#                             'axis_minor_length': 'myelin_axis_minor_length',
#                             'axis_major_length': 'myelin_axis_major_length',
#                             'feret_diameter_max': 'myelin_feret_diameter_max'}, inplace=True)

#         ##assert len(obj_table.axes[0]) == 1, "we expected to have a single object"
#         if len(obj_table.axes[0]) > 1:
#             continue
        
#         # Create the padded myelin_bw image to avoid edge effects in the contours
#         myelin_bw_fill = ndimage.binary_fill_holes(myelin_bw)
#         # create the "axon area", this is the internal area of the myelin region
#         auto_axon_region = np.logical_xor(myelin_bw_fill,myelin_bw)
#         if np.sum(auto_axon_region)<100:
#             openCnt += 1
#         #    continue
#             warnings.warn("The axon was most probably not closed.")
#             myelin_bw_fill = convex_hull_image(myelin_bw_fill)
#             auto_axon_region = np.logical_xor(myelin_bw_fill,myelin_bw)
#             auto_axon_region = isotropic_erosion(auto_axon_region, 3)
#             auto_axon_region = remove_small_objects(auto_axon_region, min_size=500)
#             #assert np.sum(auto_axon_region)>500, f"unexpected problems with the auto axon region {obj_idx}"
#             if np.sum(auto_axon_region)>500:
#                 continue

#         myelin_filled_area = np.sum(myelin_bw_fill)
#         obj_table['myelin_filled_area'] =myelin_filled_area

#         myelin_bw_pad = np.pad(myelin_bw_fill, ((pad_size, pad_size), (pad_size, pad_size),), mode='constant', constant_values=0)
#         # convex hull calculation including smoothing to avoid pixelation effects
#         try:
#             myelin_min_box_corners, myelin_min_box_edges, rot_mat = bbox_from_BWimg(myelin_bw_pad)
#             # using the bbox as feret calculator and AR
#             obj_table['myelin_feret_max'] = np.max(myelin_min_box_edges)
#             obj_table['myelin_feret_min'] = np.min(myelin_min_box_edges)
#             obj_table['myelin_AR'] = np.max(myelin_min_box_edges)/np.min(myelin_min_box_edges)
#             # calculating myelin median width
#             obj_table['myelin_width'] = morpho.get_width(myelin_bw)
#         except:
#             continue

#         auto_axon_area = np.sum(auto_axon_region)
#         obj_table['myelin_hole_area'] =auto_axon_area

#         contours, max_axon_bbox, max_axon_size = get_box_with_rotation(auto_axon_region, rot_mat.T)

#         if contours != 1:
#             obj_table['myelin_width_min_feret_direction'] = "skipped"
#             obj_table['myelin_width_max_feret_direction'] = "skipped"    
#         else:
#             myelin_width_min_feret = np.min(myelin_min_box_edges)-np.min(max_axon_size)
#             myelin_width_max_feret = np.max(myelin_min_box_edges)-np.max(max_axon_size)
#             obj_table['myelin_width_min_feret_direction'] = myelin_width_min_feret/2
#             obj_table['myelin_width_max_feret_direction'] = myelin_width_max_feret/2

#     # # cleaning axon label map
#     # axon_bw = axon_lbl>0
#     # axon_bw = np.logical_and(myelin_bw_fill, axon_bw)
#     # user_def_axon = np.any(axon_bw)

#     # if not(user_def_axon):
#     #     print('no user defined axon, we will create one based on the myelin')
#     #     axon_bw = auto_axon_region 
#     # assert np.any(axon_bw), "The axon label map is empty: does not contain any True values"
    
#     # # I do not expect holes in the axon image
#     # axon_bw = ndimage.binary_fill_holes(axon_bw)
#     # axon_bw_pad = np.pad(axon_bw, ((pad_size, pad_size), (pad_size, pad_size),), mode='constant', constant_values=0)

#     # # convex hull calculation including smoothing to avoid pixelation effects
#     # axon_min_box_corners, axon_min_box_edges, _ = bbox_from_BWimg(axon_bw_pad)

#     # # using the bbox as feret calculator and AR
#     # obj_table['axon_area'] = np.sum(axon_bw_pad)
#     # obj_table['axon_feret_max'] = np.max(axon_min_box_edges)
#     # obj_table['axon_feret_min'] = np.min(axon_min_box_edges)
#     # obj_table['axon_AR'] = np.max(axon_min_box_edges)/np.min(axon_min_box_edges)

#     # #check the mito labels
#     # mito_bw = mito_lbl>0
#     # mito_bw = np.logical_and(myelin_bw_fill, mito_bw)
#     # user_def_mito = np.any(mito_bw)


#     # if not(user_def_mito):
#     #     print('no mito in the neuron')
#     #     mito_number = 0
#     #     mito_total_area = 0
#     # else:
#     #     mito_lbl = label(mito_bw)
#     #     mito_number = np.max(mito_lbl)
#     #     mito_total_area = np.sum(mito_bw)

#     # obj_table['mito_total_area'] = mito_total_area
#     # obj_table['mito_number'] = mito_number


#     # # cleaning the distrophic label:
#     # # cleaning axon label map
#     # dyst_bw = dyst_lbl>0
#     # dyst_bw = np.logical_and(myelin_bw_fill, dyst_bw)
#     # is_dystrophic = np.any(dyst_bw)
#     # obj_table['is_dystrophic'] = is_dystrophic
#     # # for later plot
#     # axon_lbl[np.logical_not(axon_bw)] = 0
#     # axon_lbl[axon_bw] = obj_table['label'] + 1

#     # mito_lbl[np.logical_not(mito_bw)] = 0
#     # mito_lbl[mito_bw] = obj_table['label'] + 2

#     # dyst_lbl[dyst_lbl>0] = obj_table['label'] + 3

#     # color = label2rgb(myelin_lbl.T + axon_lbl.T + mito_lbl.T + dyst_lbl.T, image=img_data.squeeze().T, bg_label=0)

#     # # Plot the image
#     # clear_output(wait=True)
#     # # Create a figure and axis for plotting
#     # fig, ax = plt.subplots(figsize=(8, 8))

#     # ax.imshow(color)
#     # tmp = np.vstack((myelin_min_box_corners, myelin_min_box_corners[0,:]))
#     # ax.plot(tmp[:, 0]-pad_size, tmp[:, 1]-pad_size, linewidth=2)

#     # tmp = np.vstack((max_axon_bbox, max_axon_bbox[0,:]))
#     # ax.plot(tmp[:, 0], tmp[:, 1], linewidth=2)

#     # tmp = np.vstack((axon_min_box_corners, axon_min_box_corners[0,:]))
#     # ax.plot(tmp[:, 0]-pad_size, tmp[:, 1]-pad_size, linewidth=2)



#     # ax.set_title(f'Image {obj_idx}')

#     # fig.canvas.draw()
#     # # Display the plot (necessary for Jupyter Notebooks)
#     # png_name = f'myelin_label_{obj_idx}.png'
#     # plt.savefig(out_folder.joinpath(png_name))
#     # plt.show()

    total_table = pd.DataFrame()
    tot_cnt = 0
    for data in morpho_data_generator(label_image,data_image):
        total_table = pd.concat([total_table, data], axis=0)
        tot_cnt += 1
        

    #print(f"done for index: {obj_idx}")
    print(f"Morphonetrics done. Tot: {tot_cnt}")

    #if index>=1:
    #    break



    # Write the DataFrame to an Excel file
    #xlsx_name =  name + "_morphometrics.xlsx"
    total_table.to_excel(name, index=False, engine='xlsxwriter')

