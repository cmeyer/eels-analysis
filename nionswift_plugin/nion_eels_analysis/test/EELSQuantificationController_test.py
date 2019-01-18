# standard libraries
import contextlib
import unittest

# local libraries
from nionswift_plugin.nion_eels_analysis import EELSQuantificationController

# third party libraries
import numpy

# local libraries
from nion.data import Calibration
from nion.data import DataAndMetadata
from nion.swift.model import DataItem
from nion.swift.model import DocumentModel
from nion.swift.model import Graphics


class TestEELSQuantificationController(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def __create_spectrum(self) -> DataItem.DataItem:
        data = numpy.random.uniform(10, 1000, 1000).astype(numpy.float32)
        intensity_calibration = Calibration.Calibration(units="~")
        dimensional_calibrations = [Calibration.Calibration(scale=2.0, units="eV")]
        data_descriptor = DataAndMetadata.DataDescriptor(is_sequence=False, collection_dimension_count=0, datum_dimension_count=1)
        xdata = DataAndMetadata.new_data_and_metadata(data, intensity_calibration=intensity_calibration, dimensional_calibrations=dimensional_calibrations, data_descriptor=data_descriptor)
        return DataItem.new_data_item(xdata)

    def __compare_intervals(self, eels_data_item, eels_edge_interval, interval_graphic):
        eels_edge_interval_int = int(eels_edge_interval.start_ev), int(eels_edge_interval.end_ev)
        graphic_interval_int = int(eels_data_item.dimensional_calibrations[-1].convert_to_calibrated_value(
            interval_graphic.interval[0] * eels_data_item.data_shape[-1])), int(
            eels_data_item.dimensional_calibrations[-1].convert_to_calibrated_value(
                interval_graphic.interval[1] * eels_data_item.data_shape[-1]))
        self.assertEqual(eels_edge_interval_int, graphic_interval_int)

    def test_adding_removing_edge_tracks_corresponding_edge_displays(self):
        document_model = DocumentModel.DocumentModel()
        with contextlib.closing(document_model):
            q = EELSQuantificationController.EELSQuantification(document_model)
            eels_data_item = self.__create_spectrum()
            document_model.append_data_item(eels_data_item)
            eels_display_item = document_model.get_display_item_for_data_item(eels_data_item)
            qd = EELSQuantificationController.EELSQuantificationDisplay(q, eels_display_item, eels_data_item)
            self.assertEqual(0, len(qd.eels_edge_displays))
            q.append_edge(EELSQuantificationController.EELSEdge())
            self.assertEqual(1, len(qd.eels_edge_displays))
            q.remove_edge(0)
            self.assertEqual(0, len(qd.eels_edge_displays))

    def test_adding_edge_from_signal_interval_configures_display_layers(self):
        document_model = DocumentModel.DocumentModel()
        with contextlib.closing(document_model):
            q = EELSQuantificationController.EELSQuantification(document_model)
            eels_data_item = self.__create_spectrum()
            document_model.append_data_item(eels_data_item)
            eels_display_item = document_model.get_display_item_for_data_item(eels_data_item)
            signal_eels_interval = EELSQuantificationController.EELSInterval(start_ev=188, end_ev=208)
            signal_interval_graphic = Graphics.IntervalGraphic()
            signal_interval_graphic.interval = signal_eels_interval.to_fractional_interval(eels_data_item.data_shape[-1], eels_data_item.dimensional_calibrations[-1])
            eels_display_item.add_graphic(signal_interval_graphic)
            qd = EELSQuantificationController.EELSQuantificationDisplay(q, eels_display_item, eels_data_item)
            qc = EELSQuantificationController.EELSQuantificationController(document_model, qd)
            qc.add_eels_edge_from_interval_graphic(signal_interval_graphic)
            self.assertEqual(3, len(eels_display_item.display_data_channels))
            self.assertEqual(3, len(eels_display_item.display_layers))
            self.assertEqual(3, len(eels_display_item.graphics))
            self.assertEqual(2, eels_display_item.display_layers[0]["data_index"])
            self.assertEqual(1, eels_display_item.display_layers[1]["data_index"])
            self.assertEqual(0, eels_display_item.display_layers[2]["data_index"])  # original data should be at the back
            self.assertEqual(1, len(document_model.display_items))
            self.assertEqual(3, len(document_model.data_items))
            self.assertEqual(1, len(document_model.computations))

    def test_removing_edge_removes_all_items(self):
        document_model = DocumentModel.DocumentModel()
        with contextlib.closing(document_model):
            q = EELSQuantificationController.EELSQuantification(document_model)
            eels_data_item = self.__create_spectrum()
            document_model.append_data_item(eels_data_item)
            eels_display_item = document_model.get_display_item_for_data_item(eels_data_item)
            signal_eels_interval = EELSQuantificationController.EELSInterval(start_ev=188, end_ev=208)
            signal_interval_graphic = Graphics.IntervalGraphic()
            signal_interval_graphic.interval = signal_eels_interval.to_fractional_interval(eels_data_item.data_shape[-1], eels_data_item.dimensional_calibrations[-1])
            eels_display_item.add_graphic(signal_interval_graphic)
            qd = EELSQuantificationController.EELSQuantificationDisplay(q, eels_display_item, eels_data_item)
            qc = EELSQuantificationController.EELSQuantificationController(document_model, qd)
            eels_edge = qc.add_eels_edge_from_interval_graphic(signal_interval_graphic)
            qc.remove_eels_edge(eels_edge)
            self.assertEqual(1, len(eels_display_item.display_data_channels))
            self.assertEqual(1, len(eels_display_item.display_layers))
            self.assertEqual(0, len(eels_display_item.graphics))
            self.assertEqual(0, eels_display_item.display_layers[0]["data_index"])  # original data should be at the back
            self.assertEqual(1, len(document_model.display_items))
            self.assertEqual(1, len(document_model.data_items))
            self.assertEqual(0, len(document_model.computations))

    def test_signal_interval_graphic_and_eels_edge_signal_interval_are_connected(self):
        document_model = DocumentModel.DocumentModel()
        with contextlib.closing(document_model):
            q = EELSQuantificationController.EELSQuantification(document_model)
            eels_data_item = self.__create_spectrum()
            document_model.append_data_item(eels_data_item)
            eels_display_item = document_model.get_display_item_for_data_item(eels_data_item)
            signal_eels_interval = EELSQuantificationController.EELSInterval(start_ev=188, end_ev=208)
            signal_interval_graphic = Graphics.IntervalGraphic()
            signal_interval_graphic.interval = signal_eels_interval.to_fractional_interval(eels_data_item.data_shape[-1], eels_data_item.dimensional_calibrations[-1])
            eels_display_item.add_graphic(signal_interval_graphic)
            qd = EELSQuantificationController.EELSQuantificationDisplay(q, eels_display_item, eels_data_item)
            qc = EELSQuantificationController.EELSQuantificationController(document_model, qd)
            eels_edge = qc.add_eels_edge_from_interval_graphic(signal_interval_graphic)
            self.__compare_intervals(eels_data_item, eels_edge.signal_eels_interval, signal_interval_graphic)
            signal_interval_graphic.interval = (0.095, 0.105)
            self.__compare_intervals(eels_data_item, eels_edge.signal_eels_interval, signal_interval_graphic)
            eels_edge.signal_eels_interval = EELSQuantificationController.EELSInterval(start_ev=200, end_ev=220)
            self.__compare_intervals(eels_data_item, eels_edge.signal_eels_interval, signal_interval_graphic)

    def test_fit_interval_graphic_and_eels_edge_fit_interval_are_connected(self):
        document_model = DocumentModel.DocumentModel()
        with contextlib.closing(document_model):
            q = EELSQuantificationController.EELSQuantification(document_model)
            eels_data_item = self.__create_spectrum()
            document_model.append_data_item(eels_data_item)
            eels_display_item = document_model.get_display_item_for_data_item(eels_data_item)
            signal_eels_interval = EELSQuantificationController.EELSInterval(start_ev=188, end_ev=208)
            signal_interval_graphic = Graphics.IntervalGraphic()
            signal_interval_graphic.interval = signal_eels_interval.to_fractional_interval(eels_data_item.data_shape[-1], eels_data_item.dimensional_calibrations[-1])
            eels_display_item.add_graphic(signal_interval_graphic)
            qd = EELSQuantificationController.EELSQuantificationDisplay(q, eels_display_item, eels_data_item)
            qc = EELSQuantificationController.EELSQuantificationController(document_model, qd)
            eels_edge = qc.add_eels_edge_from_interval_graphic(signal_interval_graphic)
            self.__compare_intervals(eels_data_item, eels_edge.fit_eels_intervals[0], eels_display_item.graphics[1])
            eels_display_item.graphics[1].interval = (0.04, 0.05)
            self.__compare_intervals(eels_data_item, eels_edge.fit_eels_intervals[0], eels_display_item.graphics[1])
            eels_edge.fit_eels_intervals[0] = EELSQuantificationController.EELSInterval(start_ev=40, end_ev=80)
            self.__compare_intervals(eels_data_item, eels_edge.signal_eels_interval, signal_interval_graphic)

    def test_adding_or_removing_fit_interval_to_eels_edge_adds_or_removes_fit_interval_graphic(self):
        document_model = DocumentModel.DocumentModel()
        with contextlib.closing(document_model):
            q = EELSQuantificationController.EELSQuantification(document_model)
            eels_data_item = self.__create_spectrum()
            document_model.append_data_item(eels_data_item)
            eels_display_item = document_model.get_display_item_for_data_item(eels_data_item)
            signal_eels_interval = EELSQuantificationController.EELSInterval(start_ev=188, end_ev=208)
            signal_interval_graphic = Graphics.IntervalGraphic()
            signal_interval_graphic.interval = signal_eels_interval.to_fractional_interval(eels_data_item.data_shape[-1], eels_data_item.dimensional_calibrations[-1])
            eels_display_item.add_graphic(signal_interval_graphic)
            qd = EELSQuantificationController.EELSQuantificationDisplay(q, eels_display_item, eels_data_item)
            qc = EELSQuantificationController.EELSQuantificationController(document_model, qd)
            eels_edge = qc.add_eels_edge_from_interval_graphic(signal_interval_graphic)
            self.assertEqual(2, len(eels_edge.fit_eels_intervals))
            self.assertEqual(3, len(eels_display_item.graphics))
            self.__compare_intervals(eels_data_item, eels_edge.fit_eels_intervals[0], eels_display_item.graphics[1])
            self.__compare_intervals(eels_data_item, eels_edge.fit_eels_intervals[1], eels_display_item.graphics[2])
            # first remove the fit interval from the edge
            eels_edge.remove_fit_eels_interval(1)
            self.assertEqual(1, len(eels_edge.fit_eels_intervals))
            self.assertEqual(2, len(eels_display_item.graphics))
            self.__compare_intervals(eels_data_item, eels_edge.fit_eels_intervals[0], eels_display_item.graphics[1])
            # now add the fit interval to back to the edge
            eels_edge.append_fit_eels_interval(EELSQuantificationController.EELSInterval(start_ev=360, end_ev=400))
            self.assertEqual(2, len(eels_edge.fit_eels_intervals))
            self.assertEqual(3, len(eels_display_item.graphics))
            self.__compare_intervals(eels_data_item, eels_edge.fit_eels_intervals[0], eels_display_item.graphics[1])
            self.__compare_intervals(eels_data_item, eels_edge.fit_eels_intervals[1], eels_display_item.graphics[2])
            # finally remove the fit interval graphic. should remove the edge's fit interval too.
            eels_display_item.remove_graphic(eels_display_item.graphics[2])
            self.assertEqual(1, len(eels_edge.fit_eels_intervals))
            self.assertEqual(2, len(eels_display_item.graphics))
            self.__compare_intervals(eels_data_item, eels_edge.fit_eels_intervals[0], eels_display_item.graphics[1])

    def test_hiding_and_showing_eels_display_view(self):
        document_model = DocumentModel.DocumentModel()
        with contextlib.closing(document_model):
            q = EELSQuantificationController.EELSQuantification(document_model)
            eels_data_item = self.__create_spectrum()
            document_model.append_data_item(eels_data_item)
            eels_display_item = document_model.get_display_item_for_data_item(eels_data_item)
            signal_eels_interval = EELSQuantificationController.EELSInterval(start_ev=188, end_ev=208)
            signal_interval_graphic = Graphics.IntervalGraphic()
            signal_interval_graphic.interval = signal_eels_interval.to_fractional_interval(eels_data_item.data_shape[-1], eels_data_item.dimensional_calibrations[-1])
            eels_display_item.add_graphic(signal_interval_graphic)
            qd = EELSQuantificationController.EELSQuantificationDisplay(q, eels_display_item, eels_data_item)
            qc = EELSQuantificationController.EELSQuantificationController(document_model, qd)
            # remove the 1st fit interval, then the signal
            eels_edge = qc.add_eels_edge_from_interval_graphic(signal_interval_graphic)
            self.assertEqual(1, len(qd.eels_edge_displays))
            qc.hide_eels_edge(eels_edge)
            qc.show_eels_edge(eels_edge)

    def test_deleting_signal_hides_edge(self):
        document_model = DocumentModel.DocumentModel()
        with contextlib.closing(document_model):
            q = EELSQuantificationController.EELSQuantification(document_model)
            eels_data_item = self.__create_spectrum()
            document_model.append_data_item(eels_data_item)
            eels_display_item = document_model.get_display_item_for_data_item(eels_data_item)
            signal_eels_interval = EELSQuantificationController.EELSInterval(start_ev=188, end_ev=208)
            signal_interval_graphic = Graphics.IntervalGraphic()
            signal_interval_graphic.interval = signal_eels_interval.to_fractional_interval(eels_data_item.data_shape[-1], eels_data_item.dimensional_calibrations[-1])
            eels_display_item.add_graphic(signal_interval_graphic)
            qd = EELSQuantificationController.EELSQuantificationDisplay(q, eels_display_item, eels_data_item)
            qc = EELSQuantificationController.EELSQuantificationController(document_model, qd)
            eels_edge = qc.add_eels_edge_from_interval_graphic(signal_interval_graphic)
            self.assertEqual(1, len(qd.eels_edge_displays))
            self.assertTrue(qd.eels_edge_displays[0].is_visible)
            eels_display_item.remove_graphic(eels_display_item.graphics[0])
            self.assertEqual(1, len(q.eels_edges))
            self.assertFalse(qd.eels_edge_displays[0].is_visible)
            self.assertEqual(1, len(qd.eels_edge_displays))
            self.assertEqual(1, len(eels_display_item.display_data_channels))
            self.assertEqual(1, len(eels_display_item.display_layers))
            self.assertEqual(0, len(eels_display_item.graphics))
            self.assertEqual(0, eels_display_item.display_layers[0]["data_index"])  # original data should be at the back
            self.assertEqual(1, len(document_model.display_items))
            self.assertEqual(1, len(document_model.data_items))
            self.assertEqual(0, len(document_model.computations))

    def test_deleting_signal_hides_edge_if_one_background_deleted_first(self):
        document_model = DocumentModel.DocumentModel()
        with contextlib.closing(document_model):
            q = EELSQuantificationController.EELSQuantification(document_model)
            eels_data_item = self.__create_spectrum()
            document_model.append_data_item(eels_data_item)
            eels_display_item = document_model.get_display_item_for_data_item(eels_data_item)
            signal_eels_interval = EELSQuantificationController.EELSInterval(start_ev=188, end_ev=208)
            qd = EELSQuantificationController.EELSQuantificationDisplay(q, eels_display_item, eels_data_item)
            qc = EELSQuantificationController.EELSQuantificationController(document_model, qd)
            # remove the 2nd fit interval, then the signal
            signal_interval_graphic = Graphics.IntervalGraphic()
            signal_interval_graphic.interval = signal_eels_interval.to_fractional_interval(eels_data_item.data_shape[-1], eels_data_item.dimensional_calibrations[-1])
            eels_display_item.add_graphic(signal_interval_graphic)
            eels_edge = qc.add_eels_edge_from_interval_graphic(signal_interval_graphic)
            self.assertEqual(1, len(qd.eels_edge_displays))
            self.assertTrue(qd.eels_edge_displays[0].is_visible)
            eels_display_item.remove_graphic(eels_display_item.graphics[2])
            eels_display_item.remove_graphic(eels_display_item.graphics[0])
            self.assertEqual(1, len(q.eels_edges))
            self.assertFalse(qd.eels_edge_displays[0].is_visible)
            self.assertEqual(1, len(qd.eels_edge_displays))
            self.assertEqual(1, len(eels_display_item.display_data_channels))
            self.assertEqual(1, len(eels_display_item.display_layers))
            self.assertEqual(0, len(eels_display_item.graphics))
            self.assertEqual(0, eels_display_item.display_layers[0]["data_index"])  # original data should be at the back
            self.assertEqual(1, len(document_model.display_items))
            self.assertEqual(1, len(document_model.data_items))
            self.assertEqual(0, len(document_model.computations))

    def test_deleting_computation_hides_edge(self):
        document_model = DocumentModel.DocumentModel()
        with contextlib.closing(document_model):
            q = EELSQuantificationController.EELSQuantification(document_model)
            eels_data_item = self.__create_spectrum()
            document_model.append_data_item(eels_data_item)
            eels_display_item = document_model.get_display_item_for_data_item(eels_data_item)
            signal_eels_interval = EELSQuantificationController.EELSInterval(start_ev=188, end_ev=208)
            qd = EELSQuantificationController.EELSQuantificationDisplay(q, eels_display_item, eels_data_item)
            qc = EELSQuantificationController.EELSQuantificationController(document_model, qd)
            # remove the 2nd fit interval, then the signal
            signal_interval_graphic = Graphics.IntervalGraphic()
            signal_interval_graphic.interval = signal_eels_interval.to_fractional_interval(eels_data_item.data_shape[-1], eels_data_item.dimensional_calibrations[-1])
            eels_display_item.add_graphic(signal_interval_graphic)
            eels_edge = qc.add_eels_edge_from_interval_graphic(signal_interval_graphic)
            document_model.remove_computation(document_model.computations[0])
            self.assertEqual(1, len(q.eels_edges))
            self.assertFalse(qd.eels_edge_displays[0].is_visible)
            self.assertEqual(1, len(qd.eels_edge_displays))
            self.assertEqual(1, len(eels_display_item.display_data_channels))
            self.assertEqual(1, len(eels_display_item.display_layers))
            self.assertEqual(0, len(eels_display_item.graphics))
            self.assertEqual(0, eels_display_item.display_layers[0]["data_index"])  # original data should be at the back
            self.assertEqual(1, len(document_model.display_items))
            self.assertEqual(1, len(document_model.data_items))
            self.assertEqual(0, len(document_model.computations))

    def test_deleting_background_data_item_hides_edge(self):
        document_model = DocumentModel.DocumentModel()
        with contextlib.closing(document_model):
            q = EELSQuantificationController.EELSQuantification(document_model)
            eels_data_item = self.__create_spectrum()
            document_model.append_data_item(eels_data_item)
            eels_display_item = document_model.get_display_item_for_data_item(eels_data_item)
            signal_eels_interval = EELSQuantificationController.EELSInterval(start_ev=188, end_ev=208)
            qd = EELSQuantificationController.EELSQuantificationDisplay(q, eels_display_item, eels_data_item)
            qc = EELSQuantificationController.EELSQuantificationController(document_model, qd)
            # remove the 2nd fit interval, then the signal
            signal_interval_graphic = Graphics.IntervalGraphic()
            signal_interval_graphic.interval = signal_eels_interval.to_fractional_interval(eels_data_item.data_shape[-1], eels_data_item.dimensional_calibrations[-1])
            eels_display_item.add_graphic(signal_interval_graphic)
            eels_edge = qc.add_eels_edge_from_interval_graphic(signal_interval_graphic)
            document_model.remove_data_item(document_model.data_items[2])
            self.assertEqual(1, len(q.eels_edges))
            self.assertFalse(qd.eels_edge_displays[0].is_visible)
            self.assertEqual(1, len(qd.eels_edge_displays))
            self.assertEqual(1, len(eels_display_item.display_data_channels))
            self.assertEqual(1, len(eels_display_item.display_layers))
            self.assertEqual(0, len(eels_display_item.graphics))
            self.assertEqual(0, eels_display_item.display_layers[0]["data_index"])  # original data should be at the back
            self.assertEqual(1, len(document_model.display_items))
            self.assertEqual(1, len(document_model.data_items))
            self.assertEqual(0, len(document_model.computations))

    def test_deleting_signal_data_item_hides_edge(self):
        document_model = DocumentModel.DocumentModel()
        with contextlib.closing(document_model):
            q = EELSQuantificationController.EELSQuantification(document_model)
            eels_data_item = self.__create_spectrum()
            document_model.append_data_item(eels_data_item)
            eels_display_item = document_model.get_display_item_for_data_item(eels_data_item)
            signal_eels_interval = EELSQuantificationController.EELSInterval(start_ev=188, end_ev=208)
            qd = EELSQuantificationController.EELSQuantificationDisplay(q, eels_display_item, eels_data_item)
            qc = EELSQuantificationController.EELSQuantificationController(document_model, qd)
            # remove the 2nd fit interval, then the signal
            signal_interval_graphic = Graphics.IntervalGraphic()
            signal_interval_graphic.interval = signal_eels_interval.to_fractional_interval(eels_data_item.data_shape[-1], eels_data_item.dimensional_calibrations[-1])
            eels_display_item.add_graphic(signal_interval_graphic)
            eels_edge = qc.add_eels_edge_from_interval_graphic(signal_interval_graphic)
            document_model.remove_data_item(document_model.data_items[1])
            self.assertEqual(1, len(q.eels_edges))
            self.assertFalse(qd.eels_edge_displays[0].is_visible)
            self.assertEqual(1, len(qd.eels_edge_displays))
            self.assertEqual(1, len(eels_display_item.display_data_channels))
            self.assertEqual(1, len(eels_display_item.display_layers))
            self.assertEqual(0, len(eels_display_item.graphics))
            self.assertEqual(0, eels_display_item.display_layers[0]["data_index"])  # original data should be at the back
            self.assertEqual(1, len(document_model.display_items))
            self.assertEqual(1, len(document_model.data_items))
            self.assertEqual(0, len(document_model.computations))

    def test_adding_edge_configures_associated_data_structure(self):
        document_model = DocumentModel.DocumentModel()
        with contextlib.closing(document_model):
            q = EELSQuantificationController.EELSQuantification(document_model)
            self.assertEqual(1, len(document_model.data_structures))
            eels_data_item = self.__create_spectrum()
            document_model.append_data_item(eels_data_item)
            eels_display_item = document_model.get_display_item_for_data_item(eels_data_item)
            signal_eels_interval = EELSQuantificationController.EELSInterval(start_ev=188, end_ev=208)
            qd = EELSQuantificationController.EELSQuantificationDisplay(q, eels_display_item, eels_data_item)
            qc = EELSQuantificationController.EELSQuantificationController(document_model, qd)
            # remove the 2nd fit interval, then the signal
            signal_interval_graphic = Graphics.IntervalGraphic()
            signal_interval_graphic.interval = signal_eels_interval.to_fractional_interval(eels_data_item.data_shape[-1], eels_data_item.dimensional_calibrations[-1])
            eels_display_item.add_graphic(signal_interval_graphic)
            eels_edge = qc.add_eels_edge_from_interval_graphic(signal_interval_graphic)
            # check general properties
            self.assertEqual(2, len(document_model.data_structures))
            # check the eels quantification data structure
            self.assertEqual("nion.eels_quantification", document_model.data_structures[0].structure_type)
            self.assertEqual(1, len(document_model.data_structures[0].eels_edges))
            self.assertEqual(2, len(document_model.data_structures[0].eels_edges[0]["fit_eels_intervals"]))
            self.assertIn("signal_eels_interval", document_model.data_structures[0].eels_edges[0])
            # check the eels quantification display structure
            self.assertEqual("nion.eels_quantification_display", document_model.data_structures[1].structure_type)
            self.assertEqual(eels_data_item, document_model.data_structures[1].get_referenced_object("eels_data_item"))
            self.assertEqual(eels_display_item, document_model.data_structures[1].get_referenced_object("eels_display_item"))
            self.assertEqual(1, len(document_model.data_structures[1].eels_edge_displays))

    # test_eels_quantification_disconnects_if_data_structure_deleted
    # test_associated_data_structure_is_removed_when_display_item_removed
    # test_orphan_associated_data_structures_are_removed_on_reload
    # test_deleting_last_background_deletes_edge
