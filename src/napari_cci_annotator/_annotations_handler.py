from qtpy.QtGui import  QStandardItem, QStandardItemModel
from qtpy.QtCore import QModelIndex, Qt
import numpy as np
from .morphometrics import create_morpho_table_from_data, morpho_data_generator
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

class AnnotationsHandler:
    
    def __init__(self):
        self._ann_model = QStandardItemModel()
        self._ann_model.setColumnCount(4)
        self._ann_model.setHeaderData(0,Qt.Horizontal,"Label id")
        self._ann_model.setHeaderData(1,Qt.Horizontal,"X coord")
        self._ann_model.setHeaderData(2,Qt.Horizontal,"Y coord")
        self._ann_model.setHeaderData(3,Qt.Horizontal,"Area (pixels)")
        self.executor = ThreadPoolExecutor()
        
    def get_annotations_model(self):
        return self._ann_model
        
    def clear_model(self):
        self._ann_model.clear()
        self._ann_model.setColumnCount(4)
        self._ann_model.setHeaderData(0,Qt.Horizontal,"Label id")
        self._ann_model.setHeaderData(1,Qt.Horizontal,"X coord")
        self._ann_model.setHeaderData(2,Qt.Horizontal,"Y coord")
        self._ann_model.setHeaderData(3,Qt.Horizontal,"Area (pixels)")
    
    def count(self):
        return self._ann_model.rowCount()
    
    def add_annotations_to_model(self, label_image, data_image, napariViewer):
        future = self.executor.submit(self._add_annotations, label_image, data_image, napariViewer)
        return future

    def _add_annotations(self, label_image, data_image, napariViewer):
        self.clear_model()

        for data in morpho_data_generator(label_image,data_image):
            if data.empty:
                continue
            items = []
            label = data.label[0]
            labelItem = QStandardItem(str(label))
            labelItem.setData(data)
            items.append(labelItem)

            xItem = QStandardItem(str(data['center_x'][0]))
            yItem = QStandardItem(str(data['center_y'][0]))
            items.append(xItem)
            items.append(yItem)

            areaItem = QStandardItem(str(data.myelin_area[0]))
            items.append(areaItem)
            self._ann_model.appendRow(items)
            print(f"Processing label {label}")

                
        #         # Process each label
        #         self._ann_model.appendRow(items)
        # unique_labels = np.unique(annLayerData)
        # for label in unique_labels:
        #     if label != 0:  # Assuming 0 is the background
        #         items = []
        #         labelItem = QStandardItem(str(label))
        #         items.append(labelItem)
                
        #         center_coordinate = np.mean(np.argwhere(annLayerData == label), axis=0)
        #         xItem = QStandardItem(str(center_coordinate[1]))
        #         yItem = QStandardItem(str(center_coordinate[0]))
        #         items.append(xItem)
        #         items.append(yItem)
                
        #         # Process each label
        
    def get_label_for_row(self,row):
        return float(self._ann_model.item(row,0).text())
        
    def get_coordinates_for_row(self,row):
        xc = float(self._ann_model.item(row,1).text())
        yc = float(self._ann_model.item(row,2).text())
        
        return (xc,yc)
    
    def remove_row(self,row):
        return self._ann_model.removeRow(row)
    
    def generate_xls_report(self, name):
        total_table = pd.DataFrame()
        for row in range(self._ann_model.rowCount()):
            item = self._ann_model.item(row,0)
            data = item.data()
            total_table = pd.concat([total_table, data], axis=0)

        total_table.to_excel(name, index=False, engine='xlsxwriter')
