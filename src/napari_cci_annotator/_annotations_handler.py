from qtpy.QtGui import  QStandardItem, QStandardItemModel
from qtpy.QtCore import QModelIndex, Qt, QSortFilterProxyModel
from qtpy.QtWidgets import QApplication, QStyle
import numpy as np
from .morphometrics import create_morpho_table_from_data, morpho_data_generator
from concurrent.futures import ThreadPoolExecutor
import pandas as pd


class NumericSortProxyModel(QSortFilterProxyModel):

    def lessThan(self, left, right):
        left_data = self.sourceModel().data(left, Qt.UserRole +1)
        right_data = self.sourceModel().data(right, Qt.UserRole +1)
        
        try:
            return float(left_data) < float(right_data)  # Numeric comparison
        except (ValueError, TypeError):
            return super().lessThan(left, right)  # Fallback to string


class AnnotationsHandler:
    
    def __init__(self):
        self._ann_model = QStandardItemModel()
        self._proxy_model = NumericSortProxyModel()
        self._proxy_model.setSourceModel(self._ann_model)
        self.executor = ThreadPoolExecutor()
        self.set_headers()
        
    def model(self):
        return self._proxy_model
        
    def set_headers(self):
        #self._ann_model.setColumnCount(6)
        self.model().setHeaderData(0,Qt.Horizontal,"Label id", Qt.DisplayRole)
        self._ann_model.setHeaderData(1,Qt.Horizontal,"X coord")
        self._ann_model.setHeaderData(2,Qt.Horizontal,"Y coord")
        self._ann_model.setHeaderData(3,Qt.Horizontal,"Area (pixels)")
        self._ann_model.setHeaderData(4,Qt.Horizontal,"# parts")
        self._ann_model.setHeaderData(5,Qt.Horizontal,"status")
        
        
    def clear_model(self):
        self._proxy_model.clear()
        self.set_headers()
        
    def count(self):
        return self._ann_model.rowCount()
    
    def add_annotations_to_model(self, label_image, data_image, napariViewer):
        future = self.executor.submit(self._add_annotations, label_image, data_image, napariViewer)
        return future

    def _add_annotations(self, label_image, data_image, napariViewer):
        self.clear_model()

        style = QApplication.style()

        gen = morpho_data_generator(label_image,data_image)
        for i in range(20):
            data = next(gen)

#        for data in morpho_data_generator(label_image,data_image):
            if data.empty:
                continue
            items = []
            label = data.label[0]
            labelItem = QStandardItem(str(label))
            labelItem.setData(label)
            labelItem.setData(data, Qt.UserRole+2)
            items.append(labelItem)

            x = data['center_x'][0]
            xItem = QStandardItem(str(x))
            xItem.setData(x)
            items.append(xItem)

            y = data['center_y'][0]
            yItem = QStandardItem(str(y))
            yItem.setData(y)
            items.append(yItem)

            area = data.myelin_area[0]
            areaItem = QStandardItem(str(area))
            areaItem.setData(area)
            items.append(areaItem)

            nit = data["nr_of_parts"][0]
            nrItems = QStandardItem(str(nit))
            nrItems.setData(nit)
            items.append(nrItems)
            
            
            statusItem = QStandardItem("")
            if not data['status_ok'][0]:
                statusItem.setIcon(style.standardIcon(QStyle.SP_MessageBoxWarning))
                statusItem.setToolTip(data['error_msg'][0])

            items.append(statusItem)
            
            self._ann_model.appendRow(items)
            
            #print(f"Processing label {label}")

                
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
        self.set_headers()
        
        
    def get_label_for_row(self,row):
        idx = self._proxy_model.index(row,0)
        return float(self._proxy_model.data(idx))
        
    def get_coordinates_for_row(self,row):
        x_idx = self._proxy_model.index(row,1)
        y_idx = self._proxy_model.index(row,2)
        xc = float(self._proxy_model.data(x_idx))
        yc = float(self._proxy_model.data(y_idx))
        
        return (xc,yc)
    
    def get_status_msg_for_row(self, row):
        idx = self._proxy_model.index(row,5)
        return str(self._proxy_model.data(idx))
    
    def remove_row(self,row):
        return self._proxy_model.removeRow(row)
    
    def generate_xls_report(self, name):
        total_table = pd.DataFrame()
        for row in range(self._ann_model.rowCount()):
            item = self._ann_model.item(row,0)
            data = item.data(Qt.UserRole+2)
            total_table = pd.concat([total_table, data], axis=0)

        total_table.to_excel(name, index=False, engine='xlsxwriter')
