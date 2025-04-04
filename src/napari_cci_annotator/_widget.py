"""
This module contains four napari widgets declared in
different ways:

- a pure Python function flagged with `autogenerate: true`
    in the plugin manifest. Type annotations are used by
    magicgui to generate widgets for each parameter. Best
    suited for simple processing tasks - usually taking
    in and/or returning a layer.
- a `magic_factory` decorated function. The `magic_factory`
    decorator allows us to customize aspects of the resulting
    GUI, including the widgets associated with each parameter.
    Best used when you have a very simple processing task,
    but want some control over the autogenerated widgets. If you
    find yourself needing to define lots of nested functions to achieve
    your functionality, maybe look at the `Container` widget!
- a `magicgui.widgets.Container` subclass. This provides lots
    of flexibility and customization options while still supporting
    `magicgui` widgets and convenience methods for creating widgets
    from type annotations. If you want to customize your widgets and
    connect callbacks, this is the best widget option for you.
- a `QWidget` subclass. This provides maximal flexibility but requires
    full specification of widget layouts, callbacks, events, etc.

References:
- Widget specification: https://napari.org/stable/plugins/guides.html?#widgets
- magicgui docs: https://pyapp-kit.github.io/magicgui/

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING

from magicgui import magic_factory
from magicgui.widgets import CheckBox, Container, create_widget
from qtpy.QtWidgets import QGridLayout, QMenu, QAction, QPushButton, QCheckBox, QMessageBox, QWidget, QFileDialog, QLabel, QListView, QAbstractItemView
from qtpy.QtCore import QModelIndex, QDir, Qt, QItemSelectionModel
from skimage.util import img_as_float
import napari_cci_annotator._image_handler as _image_handler
from napari.utils.notifications import show_info
from napari import layers
import numpy as np

if TYPE_CHECKING:
    import napari

class CciAnnotatorQWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, viewer: "napari.viewer.Viewer"): 
        super().__init__()
        self.viewer = viewer

        self.setLayout(QGridLayout())

        img_btn = QPushButton("Select image dir")
        img_btn.clicked.connect(self._on_img_btn_click)
        self.img_dir_label = QLabel("None selected")
        self.img_file_view = QListView()
        self.img_file_view.setEnabled(False)

        self.layout().addWidget(img_btn,0,0)
        self.layout().addWidget(self.img_dir_label,1,0)

        ann_btn = QPushButton("Select annotations dir")
        ann_btn.clicked.connect(self._on_ann_btn_click)
        self.ann_dir_label = QLabel("None selected")
        self.ann_file_view = QListView()
        self.ann_file_view.setEnabled(False)
        self.ann_file_view.setSelectionMode(QAbstractItemView.SingleSelection)

        self.layout().addWidget(ann_btn,0,1)
        self.layout().addWidget(self.ann_dir_label,1,1)        

        self.layout().addWidget(self.img_file_view,2,0)
        self.layout().addWidget(self.ann_file_view,2,1)
        
        self.img_file_view.doubleClicked.connect(self._dbl_click_image_file)
        self.img_file_view.clicked.connect(self._click_image_file)
        
        self.start_ann_btn = QPushButton("Start annotation")
        self.start_ann_btn.clicked.connect(self._start_annotation_clicked)
        self.start_ann_btn.setEnabled(False)
        
        self.next_ann_btn = QPushButton("Next (save)")
        self.next_ann_btn.clicked.connect(self._next_annotation_clicked)
        self.next_ann_btn.setEnabled(False)
        
        self.done_ann_btn = QPushButton("Done (save)")
        self.done_ann_btn.clicked.connect(self._done_annotation_clicked)
        self.done_ann_btn.setEnabled(False)
        
        self.auto_annotate_btn = QPushButton("Auto Annotate")
        self.auto_annotate_btn.clicked.connect(self._auto_annotate_clicked)
        self.auto_annotate_btn.setEnabled(False)
        
        self.new_labels_check = QCheckBox()
        
        self.save_annotation_btn = QPushButton("Save Annotation")
        self.save_annotation_btn.clicked.connect(self._save_annotation_clicked)
        self.save_annotation_btn.setEnabled(False)
        
        
        self.layout().addWidget(self.start_ann_btn,3,0)
        self.layout().addWidget(QLabel("Starts annotation from the selected image (or first)\nWhen you are done with that image press next\nto start annotating the next image."),3,1)

        self.layout().addWidget(self.next_ann_btn,4,0)
        self.layout().addWidget(QLabel("Saves the annotated image and loads the next image and annotation."),4,1)

        self.layout().addWidget(self.done_ann_btn,5,0)
        self.layout().addWidget(QLabel("Stops the annotation session (and saves the annotation)."),5,1)
        
        self.layout().addWidget(self.auto_annotate_btn,6,0)
        self.layout().addWidget(QLabel("Automatically annotates the current image."))

        # self.layout().addWidget(self.new_labels_check,7,0)
        # self.layout().addWidget(QLabel("Create new layer when auto annotating"))
        
        self.layout().addWidget(self.save_annotation_btn,7,0)
        self.layout().addWidget(QLabel("Saves the current annotation."))

        self.annotationInProgress = False
        self.imgHandler = _image_handler.ImageHandler()
        self.img_file_view.setModel(self.imgHandler.getImgModel())
        self.ann_file_view.setModel(self.imgHandler.getAnnModel())
               
        self.imgDirSet = False
        self.annDirSet = False

        #self._setup_ann_menu()
        self.ann_file_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.ann_file_view.customContextMenuRequested.connect(self._show_context_menu)
               
        # #DEBUG STUFF!!!!
        # idir = "/home/xfolka/Projects/gisela_workflow/dl/myelin/images"
        # self.set_image_directory(idir, False)
        # adir = "/home/xfolka/Projects/gisela_workflow/dl/myelin/annotations"
        # self.set_ann_directory(adir)
      
    def checkEnableLists(self):
        self.img_file_view.setEnabled(self.annDirSet and self.imgDirSet)
        self.ann_file_view.setEnabled(self.annDirSet and self.imgDirSet)
        
        self.start_ann_btn.setEnabled(self.annDirSet and self.imgDirSet)
            
    # def _setup_ann_menu(self):
    #     self.ann_menu = QMenu()
    #     delAct = QAction("Delete")
    #     delAct.triggered.connect(self._delete_annotation)
    #     self.ann_menu.addAction(delAct)
                
    def _show_context_menu(self, qpoint):
        globalPos = self.ann_file_view.mapToGlobal(qpoint)
        index = self.ann_file_view.indexAt(qpoint)
        nn_menu = QMenu()
        delAct = QAction("Delete")
        delAct.triggered.connect(lambda: self._delete_annotation(index))
        nn_menu.addAction(delAct)

        nn_menu.exec(globalPos)
        #nn_menu.show()
      
    def set_image_directory(self, img_dir, ask_for_ann_dir = True):
        self.img_dir_label.setText(img_dir)
        rIdx = self.imgHandler.setImgRootPath(img_dir)

        self.img_file_view.setRootIndex(rIdx)
        
        if ask_for_ann_dir and self.imgHandler.checkForMatchingAnnotationsDir(img_dir):
            btn = QMessageBox.question(None,"Found annotations directory","Annotations directory found next to images. Do you want to load the annotations from there?")
            if btn == QMessageBox.Yes:
                annPath = self.imgHandler.getMatchingAnnotationsDir(img_dir)
                self.set_ann_directory(annPath)
        self.imgDirSet = True        
        
        
#        self.img_file_view.setCurrentIndex()
        #self.img_file_view.setCurrentIndex(self.imgHandler.nextImageIndex(self.img_file_view.currentIndex()))
        self.checkEnableLists()
   
    def _delete_annotation(self,index):
        btn = QMessageBox.warning(None, 
                            "Delete annotation?",
                            "Are you sure you want to delete the annotation?",
                                buttons = QMessageBox.Yes | QMessageBox.No)
        if not btn == QMessageBox.Yes:
            return

        self.imgHandler.delete_annotation(index,self.viewer)

        return
                
    def set_ann_directory(self, ann_dir):
        self.ann_dir_label.setText(ann_dir)
        rIdx = self.imgHandler.setAnnRootPath(ann_dir)
        self.ann_file_view.setRootIndex(rIdx)
        self.annDirSet = True
        self.checkEnableLists()

    def _match_ann_view_by_filename(self,fileName):
        aindx = self.imgHandler.getAnnIndexForFileName(fileName)
        self.ann_file_view.setCurrentIndex(aindx)
        
    def _on_img_btn_click(self):
        img_dir = QFileDialog.getExistingDirectory(caption = "Get image directory")
        self.set_image_directory(img_dir)        
        show_info(f"Img dir: {img_dir} selected")

    def _on_ann_btn_click(self):
        ann_dir = QFileDialog.getExistingDirectory(caption = "Get annotation directory")
        if ann_dir == '':
            return
        self.set_ann_directory(ann_dir)
        show_info(f"Annotations dir: {ann_dir} selected")
        
    def _dbl_click_image_file(self, index):
        self._remove_all_layers()
        #data = self.imgHandler.getImgData(index)
        self.imgHandler.loadImageAndAnnotation(index,self.viewer)
        self.save_annotation_btn.setEnabled(True)
        self.auto_annotate_btn.setEnabled(True)
        
    def _select_image_and_corresponding_annotation(self,index):
        data = self.imgHandler.getImgData(index) 
        self._match_ann_view_by_filename(data)
        self.img_file_view.setCurrentIndex(index)
        
    def _get_first_labels_layer_if_any(self):
        for layer in self.viewer.layers:
            if isinstance(layer, layers.Labels):
                return layer
        return None
        
        
    def _auto_annotate_clicked(self):
        
        self.imgHandler.autoAnnotateImage(self.img_file_view.currentIndex(),self.viewer)
        
    def _count_label_layers(self):
        cnt = 0
        for layer in self.viewer.layers:
            if isinstance(layer, layers.Labels):
                cnt+=1
        return cnt
           
    def _save_only_one_layer(self):
        if self._count_label_layers() > 1:
            QMessageBox.information(None, 
                        "More than one labels layer",
                        "More than one labels layer exist.\nDelete all unwanted label layers and save again.",
                        buttons = QMessageBox.Ok)
            return False
        
        labels = self._get_first_labels_layer_if_any()
        imgName = self.imgHandler.getImgData(self.img_file_view.currentIndex())
        self.imgHandler.saveAnnotationUsingName(labels.name,imgName,self.viewer)
        return True
           
    def _save_annotation_clicked(self):
        self._save_only_one_layer()
                
    def _click_image_file(self, index):
        self._select_image_and_corresponding_annotation(index)
        data = self.imgHandler.getImgData(index) 
        self.move_layer_to_top_if_it_exists(data)
        
    def _remove_all_layers(self):
           self.viewer.layers.clear()
    
    def _remove_images_from_layers(self,img_layer_name):
        imgLayer = None
        annLayer = None
        if img_layer_name in self.viewer.layers:
           imgLayer = self.viewer.layers[img_layer_name]
           self.viewer.layers.remove(imgLayer)
        else:
            return (imgLayer, annLayer)
        
        ann_layer_name = self.imgHandler.getAnnotationLayerNameFromImageName(img_layer_name)
        if ann_layer_name in self.viewer.layers:
            annLayer = layer = self.viewer.layers[ann_layer_name]
            self.viewer.layers.remove(annLayer)
        
        return (imgLayer,annLayer)
        
    def move_layer_to_top_if_it_exists(self, img_layer_name):
 
        (imgLayer, annLayer) = self._remove_images_from_layers(img_layer_name)
 
        if imgLayer:
            self.viewer.layers.append(imgLayer)
        
        if annLayer:
            self.viewer.layers.append(annLayer)

    def _select_first_items(self):
        ridx = self.img_file_view.rootIndex()
        firstIndex = self.imgHandler.getImgModel().index(0, 0,ridx)
        if firstIndex.isValid():
            self.img_file_view.setCurrentIndex(firstIndex)
            self.img_file_view.selectionModel().select(firstIndex, QItemSelectionModel.Select)
        

    def _start_annotation_clicked(self):
        #show dialog warning for clearing layers before starting. User can cancel
        if len(self.viewer.layers) > 0: 
            btn = QMessageBox.warning(None, 
                                      "Erase progress",
                                      "Clicking OK and starting annotations will delete all current layers and destroy progress",
                                      buttons = QMessageBox.Ok | QMessageBox.Cancel)
            if not btn == QMessageBox.Ok:
                return
        
        self.start_ann_btn.setEnabled(False)
        self.next_ann_btn.setEnabled(True)
        self.done_ann_btn.setEnabled(True)
        self.img_file_view.setEnabled(False)
        self.ann_file_view.setEnabled(False)
        self.save_annotation_btn.setEnabled(True)
        self.auto_annotate_btn.setEnabled(True)
 
        if not self.img_file_view.currentIndex().isValid():
            self._select_first_items()
 
        self._remove_all_layers()       
        self._select_image_and_corresponding_annotation(self.img_file_view.currentIndex())
        self.imgHandler.loadImageAndAnnotation(self.img_file_view.currentIndex(),self.viewer)
        return
    
    def _next_annotation_clicked(self):
        
        #self.imgHandler.saveAnnotation(self.img_file_view.currentIndex(),self.viewer)
        if not self._save_only_one_layer():
            return 
        
        self._remove_all_layers()       
        
        nextIdx = self.imgHandler.nextImageIndex(self.img_file_view.currentIndex())
        self._select_image_and_corresponding_annotation(nextIdx)
        self.imgHandler.loadImageAndAnnotation(self.img_file_view.currentIndex(),self.viewer)
        
        return
    
    def _done_annotation_clicked(self):
        
        if not self._save_only_one_layer():
            return 
        
        self.start_ann_btn.setEnabled(True)
        self.next_ann_btn.setEnabled(False)
        self.done_ann_btn.setEnabled(False)
        self.img_file_view.setEnabled(True)
        self.ann_file_view.setEnabled(True)

        return
        

