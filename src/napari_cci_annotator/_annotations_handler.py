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
        # self.model().setColumnCount(8)
        self.set_headers()
        
        
    def model(self):
        return self._proxy_model
        
    def set_headers(self):
        #self._ann_model.setColumnCount(6)
        self.model().setHeaderData(0,Qt.Horizontal,"Myelin id", Qt.DisplayRole)
        self.model().setHeaderData(1,Qt.Horizontal,"Axon id", Qt.DisplayRole)
        self.model().setHeaderData(2,Qt.Horizontal,"X coord")
        self.model().setHeaderData(3,Qt.Horizontal,"Y coord")
        self.model().setHeaderData(4,Qt.Horizontal,"Area (px)")
        self.model().setHeaderData(5,Qt.Horizontal,"# parts")
        self.model().setHeaderData(6,Qt.Horizontal,"Axon area")
        self.model().setHeaderData(7,Qt.Horizontal,"Axon calcu.")
        self.model().setHeaderData(8,Qt.Horizontal,"status")
        
    def clear_model(self):
        self._proxy_model.clear()
        self.set_headers()
        
    def count(self):
        return self._ann_model.rowCount()
    
    def add_annotations_to_model(self, myelin_label_image, axon_label_image,data_image, napariViewer):
        future = self.executor.submit(self._add_annotations, myelin_label_image, axon_label_image, data_image, napariViewer)
        return future

    def _add_annotations(self, myelin_label_image, axon_label_image, data_image, napariViewer):
        self.clear_model()

        style = QApplication.style()

        cnt = 0
        # gen = morpho_data_generator(myelin_label_image,axon_label_image, data_image)
        # for i in range(20):
        #     data = next(gen)
        for data in morpho_data_generator(myelin_label_image,axon_label_image, data_image):
            if data.empty:
                continue
            items = []
            myelin_label = data.myelin_label[0]
            myelinLabelItem = QStandardItem(str(myelin_label))
            myelinLabelItem.setData(myelin_label)
            myelinLabelItem.setData(data, Qt.UserRole+2)
            items.append(myelinLabelItem)

            if data['status_ok'][0]:
                axon_label = data['axon_label'][0]
            else:
                axon_label = "-"
            axonLabelItem = QStandardItem(str(axon_label))
            axonLabelItem.setData(axon_label)
            items.append(axonLabelItem)

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
            
            aa = data["axon_area"][0]
            aaItems = QStandardItem(str(aa))
            aaItems.setData(aa)
            items.append(aaItems)
            
            ac = data["axon_calculation"][0]
            acItems = QStandardItem(str(ac))
            acItems.setData(ac)
            items.append(acItems)
            
            statusItem = QStandardItem("")
            
            if data['status_ok'][0] == False:
                statusItem.setIcon(style.standardIcon(QStyle.SP_MessageBoxWarning))
                statusItem.setToolTip(data['error_msg'][0])

            items.append(statusItem)
            
            self._ann_model.appendRow(items)
            print(f"adding {cnt} to model, status: {data['status_ok'][0]}")
            cnt += 1
            
        self.set_headers()
        
        
    def get_label_for_row(self,row):
        idx = self._proxy_model.index(row,0)
        return float(self._proxy_model.data(idx))
        
    def get_coordinates_for_row(self,row):
        x_idx = self._proxy_model.index(row,2)
        y_idx = self._proxy_model.index(row,3)
        xc = float(self._proxy_model.data(x_idx))
        yc = float(self._proxy_model.data(y_idx))
        
        return (xc,yc)
    
    def get_status_msg_for_row(self, row):
        idx = self._proxy_model.index(row,6)
        return str(self._proxy_model.data(idx))
    
    def remove_row(self,row):
        return self._proxy_model.removeRow(row)
    
    def generate_xls_report(self, name):
        future = self.executor.submit(self._generate_xls_report,name)
        return future

    
    def _generate_xls_report(self, name):
        total_table = pd.DataFrame()
        for row in range(self._ann_model.rowCount()):
            item = self._ann_model.item(row,0)
            data = item.data(Qt.UserRole+2)
            total_table = pd.concat([total_table, data], axis=0)

        total_table.to_excel(name, index=False, engine='xlsxwriter')
