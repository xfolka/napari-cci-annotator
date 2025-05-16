
from ast import Tuple
from pydantic import InstanceOf
from qtpy.QtWidgets import  QFileSystemModel
from qtpy.QtCore import QModelIndex, QDir
from skimage import io, color
import os
import numpy as np
from ultralytics import YOLO
import napari_cci_annotator._config as _config
from napari import layers
import concurrent.futures
#from .segment_large_image_using_yolo import segment_large_image_data
from .segment_large_image_using_yolo import YoloSegmenter, LargeImageSegmenter

from .morphometrics import create_morpho_table_from_data

class ImageHandler:
    
    def __init__(self):
        self.annFileModel = QFileSystemModel()
        self.imgFileModel = QFileSystemModel()

        self.imgFileModel.setNameFilters(["*.png"])
        self.annFileModel.setNameFilters(["*.png"])
        
        self.imgRootIndex = QModelIndex()
        self.annRootIndex = QModelIndex()
        
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4) 

    def getImgModel(self):
        return self.imgFileModel
    
    def getAnnModel(self):
        return self.annFileModel
    
    def setImgRootPath(self,path):
        self.imgFileModel.setRootPath(path)
        self.imgRootIndex = self.imgFileModel.index(path)
#        self.currentImgIndex = self.imgRootIndex
        return self.imgRootIndex
        
    def setAnnRootPath(self,path):
        self.annFileModel.setRootPath(path)
        self.annRootIndex = self.annFileModel.index(path)
        return self.annRootIndex
    
    def getImgData(self,index):
        return self.imgFileModel.data(index)
    
    def getAnnData(self,index):
        return self.annFileModel.data(index)
    
    def getAnnIndexFromImgIndex(self, imgIndex):
        aIdx = self.annFileModel.index(imgIndex.row(),imgIndex.col(), self.annRootIndex)
        return aIdx
    
    # def getCurrentImageData(self):
    #        return self.imgFileModel.data(self.currentImgIndex)
    
    def getImgIndexForFileName(self,fileName):
        p = self.imgFileModel.rootPath()
        return self.imgFileModel.index(fileName)
    
    def getAnnIndexForFileName(self,fileName):
        p = self.annFileModel.rootPath()
        return self.annFileModel.index(p + "/" + fileName)

    def getAnnotationLayerNameFromImageName(self,imgName):
        return "ann_" + imgName

    def getImageLayerNameFromAnnotationName(self,annName):
        return annName[len("ann_"):]

    def loadImageAndAnnotation(self,index,napariViewer):
        data = self.imgFileModel.data(index)
        img = io.imread(self.imgFileModel.rootPath() + "/" + data)
        napariViewer.add_image(img,name=data)
        
        idx = self.getAnnIndexForFileName(data)
        if idx.isValid():
            ann = io.imread(self.annFileModel.rootPath() + "/" + data)
        else:
            current_image = napariViewer.layers[0]  # Assuming the image is the first layer
            image_shape = current_image.data.shape
            ann = np.zeros(image_shape, dtype=current_image.dtype)
            
        napariViewer.add_labels(ann,name=self.getAnnotationLayerNameFromImageName(data))

    def saveAnnotationUsingName(self, annLayerName, saveName, napariViewer):
        annLayer = napariViewer.layers[annLayerName]
        io.imsave(self.annFileModel.rootPath() + "/" + saveName, annLayer.data)
        return True

    
    def saveAnnotationName(self, annName, napariViewer):
        annLayer = napariViewer.layers[annName]
        io.imsave(self.annFileModel.rootPath() + "/" + self.getImageLayerNameFromAnnotationName(annName), annLayer.data)
        return True
    
    def saveAnnotation(self,imgIndex,napariViewer):
        img_layer_name = self.imgFileModel.data(imgIndex)   
        ann_layer_name = self.getAnnotationLayerNameFromImageName(img_layer_name)
        if not ann_layer_name in napariViewer.layers:
            return False
        self.saveAnnotationName(ann_layer_name,napariViewer)
        
        # annLayer = napariViewer.layers[ann_layer_name]
        # io.imsave(self.annFileModel.rootPath() + "/" + img_layer_name, annLayer.data)
        # return True

    def nextImageIndex(self,imgIndex):
        nextIdx = None
        if not imgIndex or not imgIndex.isValid():
            nextIdx = self.imgFileModel.index(0,0)
        else:
            nextIdx = self.imgFileModel.index(imgIndex.row()+1,imgIndex.column(),imgIndex.parent())
        return nextIdx

    def checkForMatchingAnnotationsDir(self,imgDirPath):
        rDir = QDir(imgDirPath)
        rDir.cdUp()
        return rDir.exists("annotations")

    def getMatchingAnnotationsDir(self,imgDirPath):
        rDir = QDir(imgDirPath)
        rDir.cdUp()
        return rDir.absoluteFilePath("annotations")

    def setMatchingAnnotationsDir(self,imgDirPath):
        rDir = QDir(imgDirPath)
        rDir.cdUp()
        return self.setAnnRootPath(rDir.absoluteFilePath("annotations"))
    
    def getModelPath(self, cell_type, backend):
        cell_type_low = cell_type.lower()
        
        if cell_type_low != "axons" and cell_type_low != "myelin":
            return False, None
        
        rdir = os.path.dirname(os.path.realpath(__file__))
        
        model_file_path = rdir + '/' + _config.MODELS_DIR + '/' + cell_type_low + '/'
        
        if backend.startswith(_config.OPENVINO_BACKEND_PREFIX):
            if cell_type_low == "axons":
                model_file_path += _config.OPENVINO_AXON_MODEL_DIR_NAME
            else:
                model_file_path += _config.OPENVINO_MYELIN_MODEL_DIR_NAME
        else:
            if cell_type_low == "axons":
                model_file_path +=  _config.AXON_MODEL_FILENAME
            else:
                model_file_path +=  _config.MYELIN_MODEL_FILENAME
            
        return True, model_file_path
    
    def getDataToAnnotate(self, imgIndex):
        img = self.getImgData(imgIndex)
        imgData = io.imread(self.imgFileModel.rootPath() + "/" + img)
        rgb_data = color.gray2rgb(imgData)
        return rgb_data
    
    def getLayerToAnnotate(self, napariViewer):
        pass
        
    
    
    def autoAnnotateImage(self,imgIndx,napariViewer, cell_type, backend, labels_layer = None):
        _, model_file_path = self.getModelPath(cell_type,backend)
        if not model_file_path:
            return
        
        model = YOLO(model_file_path, task='segment')
        rgb_data = self.getDataToAnnotate(imgIndx)
        results = model.predict(source=rgb_data, imgsz=_config.IMG_SIZE,show_boxes=False,show_labels=False, verbose=False)
        result_masks = results[0].masks
        masks = result_masks.data.cpu().numpy()
        shape = results[0].masks.shape
        if not labels_layer:
            all_masks = np.zeros(shape=(_config.IMG_SIZE,_config.IMG_SIZE), dtype=np.uint32)
        else:
            all_masks = labels_layer.data
        for n in range(shape[0]):
            mask = masks[n,:,:] * (n+1) 
            mask = np.expand_dims(mask,axis=2)
            mask = np.squeeze(mask).astype(np.uint32)
            all_masks = np.where(all_masks == 0, mask, all_masks)

        if labels_layer is None:
            napariViewer.add_labels(all_masks,name="auto_"+self.getAnnotationLayerNameFromImageName(self.getImgData(imgIndx)))
        else:
            labels_layer.data = all_masks  # Update the existing layer's data
            labels_layer.refresh()  # Refresh the layer to reflect changes

    def annotate_selected_layer(self, overlap, napariViewer, be_type, cell_type, crop_image, clear_border):
        
        cell_type_low = cell_type.lower()
        
        if cell_type_low != "axons" and cell_type_low != "myelin":
            return False, None
        
        rdir = os.path.dirname(os.path.realpath(__file__))
        
        model_file_path = rdir + '/' + _config.MODELS_DIR + '/' + cell_type_low + '/'
        
        if be_type.startswith(_config.OPENVINO_BACKEND_PREFIX):
            if cell_type_low == "axons":
                model_file_path += _config.OPENVINO_AXON_MODEL_DIR_NAME
            else:
                model_file_path += _config.OPENVINO_MYELIN_MODEL_DIR_NAME
        else:
            if cell_type_low == "axons":
                model_file_path +=  _config.AXON_MODEL_FILENAME
            else:
                model_file_path +=  _config.MYELIN_MODEL_FILENAME
        
        yolo_seg = YoloSegmenter(model_file_path, 1024)
        #sd_seg = StarDistSegmenter(_config.AXON_MODEL_NAME,model_file_path,img_size=1024)
        #model = YOLO(model_file_path, task='segment')
        segmenter = LargeImageSegmenter()
        selected = napariViewer.layers.selection.active
        if not selected or not isinstance(selected, layers.Image):
            return False, None
        # Submit the task to the executor and get a Future object
        func = segmenter.segment_large_image_data
        if crop_image:
            func = segmenter.segment_large_image_crop_to_match_model_size
        
        future = self.executor.submit(func,yolo_seg, selected.data, 1024, overlap, clear_border)
        return True, future


    # def annotate_selected_layer(self, overlap, radius, napariViewer, be_type):
        
    #     rdir = os.path.dirname(os.path.realpath(__file__))
    #     if be_type.startswith(_config.OPENVINO_BACKEND_PREFIX):
    #         model_file_path = rdir + '/' + _config.OPENVINO_MODEL_PATH        
    #     else:
    #         model_file_path = rdir + '/' + _config.MYELIN_MODEL_PATH
        
    #     yoseg = YoloSegmenter(model_file_path,1024)
    #     #model = YOLO(model_file_path, task='segment')
    #     segmenter = LargeImageSegmenter()
    #     selected = napariViewer.layers.selection.active
    #     if not selected:
    #         return False, None
    #     # Submit the task to the executor and get a Future object
    #     future = self.executor.submit(segmenter.segment_large_image_data,yoseg, selected.data, 1024, over_lap=100)
    #     return True, future
        

    def delete_annotation(self, index, napariViewer):
        annName = self.getAnnData(index)
        annLayerName = self.getAnnotationLayerNameFromImageName(annName)
        if annLayerName in napariViewer.layers:
           annLayer = napariViewer.layers[annLayerName]
           napariViewer.layers.remove(annLayer)
        self.annFileModel.remove(index)

    def calulate_morphometrics(self, name, myelin_label_layer, axon_label_layer, image_layer):
        future = self.executor.submit(create_morpho_table_from_data, name, myelin_label_layer, axon_label_layer, image_layer)
        return future