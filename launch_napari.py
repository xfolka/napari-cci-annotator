# launch_napari.py
import napari
#from napari import Viewer, run

viewer = napari.Viewer()
dock_widget, plugin_widget = viewer.window.add_plugin_dock_widget(
    "napari-cci-annotator", "CCI Annotator Plugin"
)
# Optional steps to setup your plugin to a state of failure
# E.g. plugin_widget.parameter_name.value = "some value"
# E.g. plugin_widget.button.click()
napari.run()