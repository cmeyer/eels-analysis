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
from nion.swift.model import Profile


def create_memory_profile_context():
    return Profile.MemoryProfileContext()


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
            qm = EELSQuantificationController.EELSQuantificationManager.get_instance(document_model)
            q = qm.create_eels_quantification()
            eels_data_item = self.__create_spectrum()
            document_model.append_data_item(eels_data_item)
            eels_display_item = document_model.get_display_item_for_data_item(eels_data_item)
            signal_eels_interval = EELSQuantificationController.EELSInterval(start_ev=188, end_ev=208)
            qd = qm.create_eels_quantification_display(q, eels_display_item, eels_data_item)
            self.assertEqual(0, len(qd.eels_edge_displays))
            eels_edge = EELSQuantificationController.EELSEdge(signal_eels_interval=signal_eels_interval)
            q.append_edge(eels_edge)
            qd.show_eels_edge(eels_edge)
            self.assertEqual(1, len(qd.eels_edge_displays))
            q.remove_edge(0)
            self.assertEqual(0, len(qd.eels_edge_displays))

    def test_adding_edge_from_signal_interval_configures_display_layers(self):
        document_model = DocumentModel.DocumentModel()
        with contextlib.closing(document_model):
            qm = EELSQuantificationController.EELSQuantificationManager.get_instance(document_model)
            q = qm.create_eels_quantification()
            self.assertEqual(1, len(document_model.data_structures))
            eels_data_item = self.__create_spectrum()
            document_model.append_data_item(eels_data_item)
            eels_display_item = document_model.get_display_item_for_data_item(eels_data_item)
            signal_eels_interval = EELSQuantificationController.EELSInterval(start_ev=188, end_ev=208)
            signal_interval_graphic = Graphics.IntervalGraphic()
            signal_interval_graphic.interval = signal_eels_interval.to_fractional_interval(eels_data_item.data_shape[-1], eels_data_item.dimensional_calibrations[-1])
            eels_display_item.add_graphic(signal_interval_graphic)
            qd = qm.create_eels_quantification_display(q, eels_display_item, eels_data_item)
            qd.add_eels_edge_from_interval_graphic(signal_interval_graphic)
            self.assertEqual(3, len(eels_display_item.display_data_channels))
            self.assertEqual(3, len(eels_display_item.display_layers))
            self.assertEqual(3, len(eels_display_item.graphics))
            self.assertEqual(2, eels_display_item.display_layers[0]["data_index"])
            self.assertEqual(1, eels_display_item.display_layers[1]["data_index"])
            self.assertEqual(0, eels_display_item.display_layers[2]["data_index"])  # original data should be at the back
            self.assertEqual(1, len(document_model.display_items))
            self.assertEqual(3, len(document_model.data_items))
            self.assertEqual(1, len(document_model.computations))
            self.assertEqual(2, len(document_model.data_structures))

    def test_removing_edge_removes_all_items(self):
        document_model = DocumentModel.DocumentModel()
        with contextlib.closing(document_model):
            qm = EELSQuantificationController.EELSQuantificationManager.get_instance(document_model)
            q = qm.create_eels_quantification()
            eels_data_item = self.__create_spectrum()
            document_model.append_data_item(eels_data_item)
            eels_display_item = document_model.get_display_item_for_data_item(eels_data_item)
            signal_eels_interval = EELSQuantificationController.EELSInterval(start_ev=188, end_ev=208)
            signal_interval_graphic = Graphics.IntervalGraphic()
            signal_interval_graphic.interval = signal_eels_interval.to_fractional_interval(eels_data_item.data_shape[-1], eels_data_item.dimensional_calibrations[-1])
            eels_display_item.add_graphic(signal_interval_graphic)
            qd = qm.create_eels_quantification_display(q, eels_display_item, eels_data_item)
            eels_edge = qd.add_eels_edge_from_interval_graphic(signal_interval_graphic)
            qd.remove_eels_edge(eels_edge)
            self.assertEqual(1, len(eels_display_item.display_data_channels))
            self.assertEqual(1, len(eels_display_item.display_layers))
            self.assertEqual(0, len(eels_display_item.graphics))
            self.assertEqual(0, eels_display_item.display_layers[0]["data_index"])  # original data should be at the back
            self.assertEqual(1, len(document_model.display_items))
            self.assertEqual(1, len(document_model.data_items))
            self.assertEqual(0, len(document_model.computations))
            self.assertEqual(2, len(document_model.data_structures))

    def test_signal_interval_graphic_and_eels_edge_signal_interval_are_connected(self):
        document_model = DocumentModel.DocumentModel()
        with contextlib.closing(document_model):
            qm = EELSQuantificationController.EELSQuantificationManager.get_instance(document_model)
            q = qm.create_eels_quantification()
            eels_data_item = self.__create_spectrum()
            document_model.append_data_item(eels_data_item)
            eels_display_item = document_model.get_display_item_for_data_item(eels_data_item)
            signal_eels_interval = EELSQuantificationController.EELSInterval(start_ev=188, end_ev=208)
            signal_interval_graphic = Graphics.IntervalGraphic()
            signal_interval_graphic.interval = signal_eels_interval.to_fractional_interval(eels_data_item.data_shape[-1], eels_data_item.dimensional_calibrations[-1])
            eels_display_item.add_graphic(signal_interval_graphic)
            qd = qm.create_eels_quantification_display(q, eels_display_item, eels_data_item)
            eels_edge = qd.add_eels_edge_from_interval_graphic(signal_interval_graphic)
            self.__compare_intervals(eels_data_item, eels_edge.signal_eels_interval, signal_interval_graphic)
            signal_interval_graphic.interval = (0.095, 0.105)
            self.__compare_intervals(eels_data_item, eels_edge.signal_eels_interval, signal_interval_graphic)
            eels_edge.signal_eels_interval = EELSQuantificationController.EELSInterval(start_ev=200, end_ev=220)
            self.__compare_intervals(eels_data_item, eels_edge.signal_eels_interval, signal_interval_graphic)

    def test_fit_interval_graphic_and_eels_edge_fit_interval_are_connected(self):
        document_model = DocumentModel.DocumentModel()
        with contextlib.closing(document_model):
            qm = EELSQuantificationController.EELSQuantificationManager.get_instance(document_model)
            q = qm.create_eels_quantification()
            eels_data_item = self.__create_spectrum()
            document_model.append_data_item(eels_data_item)
            eels_display_item = document_model.get_display_item_for_data_item(eels_data_item)
            signal_eels_interval = EELSQuantificationController.EELSInterval(start_ev=188, end_ev=208)
            signal_interval_graphic = Graphics.IntervalGraphic()
            signal_interval_graphic.interval = signal_eels_interval.to_fractional_interval(eels_data_item.data_shape[-1], eels_data_item.dimensional_calibrations[-1])
            eels_display_item.add_graphic(signal_interval_graphic)
            qd = qm.create_eels_quantification_display(q, eels_display_item, eels_data_item)
            eels_edge = qd.add_eels_edge_from_interval_graphic(signal_interval_graphic)
            self.__compare_intervals(eels_data_item, eels_edge.fit_eels_intervals[0], eels_display_item.graphics[1])
            eels_display_item.graphics[1].interval = (0.04, 0.05)
            self.__compare_intervals(eels_data_item, eels_edge.fit_eels_intervals[0], eels_display_item.graphics[1])
            eels_edge.fit_eels_intervals[0] = EELSQuantificationController.EELSInterval(start_ev=40, end_ev=80)
            self.__compare_intervals(eels_data_item, eels_edge.signal_eels_interval, signal_interval_graphic)

    def test_adding_or_removing_fit_interval_to_eels_edge_adds_or_removes_fit_interval_graphic(self):
        document_model = DocumentModel.DocumentModel()
        with contextlib.closing(document_model):
            qm = EELSQuantificationController.EELSQuantificationManager.get_instance(document_model)
            q = qm.create_eels_quantification()
            eels_data_item = self.__create_spectrum()
            document_model.append_data_item(eels_data_item)
            eels_display_item = document_model.get_display_item_for_data_item(eels_data_item)
            signal_eels_interval = EELSQuantificationController.EELSInterval(start_ev=188, end_ev=208)
            signal_interval_graphic = Graphics.IntervalGraphic()
            signal_interval_graphic.interval = signal_eels_interval.to_fractional_interval(eels_data_item.data_shape[-1], eels_data_item.dimensional_calibrations[-1])
            eels_display_item.add_graphic(signal_interval_graphic)
            qd = qm.create_eels_quantification_display(q, eels_display_item, eels_data_item)
            eels_edge = qd.add_eels_edge_from_interval_graphic(signal_interval_graphic)
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
            qm = EELSQuantificationController.EELSQuantificationManager.get_instance(document_model)
            q = qm.create_eels_quantification()
            eels_data_item = self.__create_spectrum()
            document_model.append_data_item(eels_data_item)
            eels_display_item = document_model.get_display_item_for_data_item(eels_data_item)
            signal_eels_interval = EELSQuantificationController.EELSInterval(start_ev=188, end_ev=208)
            signal_interval_graphic = Graphics.IntervalGraphic()
            signal_interval_graphic.interval = signal_eels_interval.to_fractional_interval(eels_data_item.data_shape[-1], eels_data_item.dimensional_calibrations[-1])
            eels_display_item.add_graphic(signal_interval_graphic)
            qd = qm.create_eels_quantification_display(q, eels_display_item, eels_data_item)
            # remove the 1st fit interval, then the signal
            eels_edge = qd.add_eels_edge_from_interval_graphic(signal_interval_graphic)
            self.assertEqual(1, len(qd.eels_edge_displays))
            qd.hide_eels_edge(eels_edge)
            qd.show_eels_edge(eels_edge)

    def test_deleting_signal_hides_edge(self):
        document_model = DocumentModel.DocumentModel()
        with contextlib.closing(document_model):
            qm = EELSQuantificationController.EELSQuantificationManager.get_instance(document_model)
            q = qm.create_eels_quantification()
            eels_data_item = self.__create_spectrum()
            document_model.append_data_item(eels_data_item)
            eels_display_item = document_model.get_display_item_for_data_item(eels_data_item)
            signal_eels_interval = EELSQuantificationController.EELSInterval(start_ev=188, end_ev=208)
            signal_interval_graphic = Graphics.IntervalGraphic()
            signal_interval_graphic.interval = signal_eels_interval.to_fractional_interval(eels_data_item.data_shape[-1], eels_data_item.dimensional_calibrations[-1])
            eels_display_item.add_graphic(signal_interval_graphic)
            qd = qm.create_eels_quantification_display(q, eels_display_item, eels_data_item)
            eels_edge = qd.add_eels_edge_from_interval_graphic(signal_interval_graphic)
            self.assertEqual(1, len(qd.eels_edge_displays))
            self.assertTrue(qd.is_eels_edge_visible(eels_edge))
            eels_display_item.remove_graphic(eels_display_item.graphics[0])
            self.assertEqual(1, len(q.eels_edges))
            self.assertEqual(0, len(qd.eels_edge_displays))
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
            qm = EELSQuantificationController.EELSQuantificationManager.get_instance(document_model)
            q = qm.create_eels_quantification()
            eels_data_item = self.__create_spectrum()
            document_model.append_data_item(eels_data_item)
            eels_display_item = document_model.get_display_item_for_data_item(eels_data_item)
            signal_eels_interval = EELSQuantificationController.EELSInterval(start_ev=188, end_ev=208)
            qd = qm.create_eels_quantification_display(q, eels_display_item, eels_data_item)
            # remove the 2nd fit interval, then the signal
            signal_interval_graphic = Graphics.IntervalGraphic()
            signal_interval_graphic.interval = signal_eels_interval.to_fractional_interval(eels_data_item.data_shape[-1], eels_data_item.dimensional_calibrations[-1])
            eels_display_item.add_graphic(signal_interval_graphic)
            eels_edge = qd.add_eels_edge_from_interval_graphic(signal_interval_graphic)
            self.assertEqual(1, len(qd.eels_edge_displays))
            self.assertTrue(qd.is_eels_edge_visible(eels_edge))
            eels_display_item.remove_graphic(eels_display_item.graphics[2])
            eels_display_item.remove_graphic(eels_display_item.graphics[0])
            self.assertEqual(1, len(q.eels_edges))
            self.assertEqual(0, len(qd.eels_edge_displays))
            self.assertEqual(1, len(eels_display_item.display_data_channels))
            self.assertEqual(1, len(eels_display_item.display_layers))
            self.assertEqual(0, len(eels_display_item.graphics))
            self.assertEqual(0, eels_display_item.display_layers[0]["data_index"])  # original data should be at the back
            self.assertEqual(1, len(document_model.display_items))
            self.assertEqual(1, len(document_model.data_items))
            self.assertEqual(0, len(document_model.computations))
            self.assertEqual(2, len(document_model.data_structures))

    def test_deleting_computation_hides_edge(self):
        document_model = DocumentModel.DocumentModel()
        with contextlib.closing(document_model):
            qm = EELSQuantificationController.EELSQuantificationManager.get_instance(document_model)
            q = qm.create_eels_quantification()
            eels_data_item = self.__create_spectrum()
            document_model.append_data_item(eels_data_item)
            eels_display_item = document_model.get_display_item_for_data_item(eels_data_item)
            signal_eels_interval = EELSQuantificationController.EELSInterval(start_ev=188, end_ev=208)
            qd = qm.create_eels_quantification_display(q, eels_display_item, eels_data_item)
            # remove the 2nd fit interval, then the signal
            signal_interval_graphic = Graphics.IntervalGraphic()
            signal_interval_graphic.interval = signal_eels_interval.to_fractional_interval(eels_data_item.data_shape[-1], eels_data_item.dimensional_calibrations[-1])
            eels_display_item.add_graphic(signal_interval_graphic)
            eels_edge = qd.add_eels_edge_from_interval_graphic(signal_interval_graphic)
            document_model.remove_computation(document_model.computations[0])
            self.assertEqual(1, len(q.eels_edges))
            self.assertEqual(0, len(qd.eels_edge_displays))
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
            qm = EELSQuantificationController.EELSQuantificationManager.get_instance(document_model)
            q = qm.create_eels_quantification()
            eels_data_item = self.__create_spectrum()
            document_model.append_data_item(eels_data_item)
            eels_display_item = document_model.get_display_item_for_data_item(eels_data_item)
            signal_eels_interval = EELSQuantificationController.EELSInterval(start_ev=188, end_ev=208)
            qd = qm.create_eels_quantification_display(q, eels_display_item, eels_data_item)
            # remove the 2nd fit interval, then the signal
            signal_interval_graphic = Graphics.IntervalGraphic()
            signal_interval_graphic.interval = signal_eels_interval.to_fractional_interval(eels_data_item.data_shape[-1], eels_data_item.dimensional_calibrations[-1])
            eels_display_item.add_graphic(signal_interval_graphic)
            eels_edge = qd.add_eels_edge_from_interval_graphic(signal_interval_graphic)
            document_model.remove_data_item(document_model.data_items[2])
            self.assertEqual(1, len(q.eels_edges))
            self.assertEqual(0, len(qd.eels_edge_displays))
            self.assertEqual(1, len(eels_display_item.display_data_channels))
            self.assertEqual(1, len(eels_display_item.display_layers))
            self.assertEqual(0, len(eels_display_item.graphics))
            self.assertEqual(0, eels_display_item.display_layers[0]["data_index"])  # original data should be at the back
            self.assertEqual(1, len(document_model.display_items))
            self.assertEqual(1, len(document_model.data_items))
            self.assertEqual(0, len(document_model.computations))
            self.assertEqual(2, len(document_model.data_structures))

    def test_deleting_signal_data_item_hides_edge(self):
        document_model = DocumentModel.DocumentModel()
        with contextlib.closing(document_model):
            qm = EELSQuantificationController.EELSQuantificationManager.get_instance(document_model)
            q = qm.create_eels_quantification()
            eels_data_item = self.__create_spectrum()
            document_model.append_data_item(eels_data_item)
            eels_display_item = document_model.get_display_item_for_data_item(eels_data_item)
            signal_eels_interval = EELSQuantificationController.EELSInterval(start_ev=188, end_ev=208)
            qd = qm.create_eels_quantification_display(q, eels_display_item, eels_data_item)
            # remove the 2nd fit interval, then the signal
            signal_interval_graphic = Graphics.IntervalGraphic()
            signal_interval_graphic.interval = signal_eels_interval.to_fractional_interval(eels_data_item.data_shape[-1], eels_data_item.dimensional_calibrations[-1])
            eels_display_item.add_graphic(signal_interval_graphic)
            eels_edge = qd.add_eels_edge_from_interval_graphic(signal_interval_graphic)
            document_model.remove_data_item(document_model.data_items[1])
            self.assertEqual(1, len(q.eels_edges))
            self.assertEqual(0, len(qd.eels_edge_displays))
            self.assertEqual(1, len(eels_display_item.display_data_channels))
            self.assertEqual(1, len(eels_display_item.display_layers))
            self.assertEqual(0, len(eels_display_item.graphics))
            self.assertEqual(0, eels_display_item.display_layers[0]["data_index"])  # original data should be at the back
            self.assertEqual(1, len(document_model.display_items))
            self.assertEqual(1, len(document_model.data_items))
            self.assertEqual(0, len(document_model.computations))
            self.assertEqual(2, len(document_model.data_structures))

    def test_adding_edge_configures_associated_data_structure(self):
        document_model = DocumentModel.DocumentModel()
        with contextlib.closing(document_model):
            qm = EELSQuantificationController.EELSQuantificationManager.get_instance(document_model)
            q = qm.create_eels_quantification()
            self.assertEqual(1, len(document_model.data_structures))
            eels_data_item = self.__create_spectrum()
            document_model.append_data_item(eels_data_item)
            eels_display_item = document_model.get_display_item_for_data_item(eels_data_item)
            signal_eels_interval = EELSQuantificationController.EELSInterval(start_ev=188, end_ev=208)
            qd = qm.create_eels_quantification_display(q, eels_display_item, eels_data_item)
            # remove the 2nd fit interval, then the signal
            signal_interval_graphic = Graphics.IntervalGraphic()
            signal_interval_graphic.interval = signal_eels_interval.to_fractional_interval(eels_data_item.data_shape[-1], eels_data_item.dimensional_calibrations[-1])
            eels_display_item.add_graphic(signal_interval_graphic)
            eels_edge = qd.add_eels_edge_from_interval_graphic(signal_interval_graphic)
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

    def test_reloading_quantification_with_edges(self):
        with create_memory_profile_context() as profile_context:
            document_model = DocumentModel.DocumentModel(profile=profile_context.create_profile())
            with contextlib.closing(document_model):
                qm = EELSQuantificationController.EELSQuantificationManager.get_instance(document_model)
                q = qm.create_eels_quantification()
                signal_eels_interval = EELSQuantificationController.EELSInterval(start_ev=188, end_ev=208)
                fit_eels_interval = EELSQuantificationController.EELSInterval(start_ev=160, end_ev=180)
                eels_edge = EELSQuantificationController.EELSEdge(signal_eels_interval=signal_eels_interval, fit_eels_intervals=[fit_eels_interval])
                q.append_edge(eels_edge)
            document_model = DocumentModel.DocumentModel(profile=profile_context.create_profile())
            with contextlib.closing(document_model):
                qm = EELSQuantificationController.EELSQuantificationManager.get_instance(document_model)
                self.assertEqual(1, len(qm.eels_quantifications))
                q = qm.eels_quantifications[0]
                self.assertEqual(1, len(q.eels_edges))
                eels_edge = q.eels_edges[0]
                self.assertEqual(188.0, eels_edge.signal_eels_interval.start_ev)
                self.assertEqual(208.0, eels_edge.signal_eels_interval.end_ev)
                self.assertEqual(1, len(eels_edge.fit_eels_intervals))
                self.assertEqual(160.0, eels_edge.fit_eels_intervals[0].start_ev)
                self.assertEqual(180.0, eels_edge.fit_eels_intervals[0].end_ev)

    def test_reloading_quantification_display_with_edges(self):
        with create_memory_profile_context() as profile_context:
            document_model = DocumentModel.DocumentModel(profile=profile_context.create_profile())
            with contextlib.closing(document_model):
                qm = EELSQuantificationController.EELSQuantificationManager.get_instance(document_model)
                q = qm.create_eels_quantification()
                signal_eels_interval = EELSQuantificationController.EELSInterval(start_ev=188, end_ev=208)
                fit_eels_interval = EELSQuantificationController.EELSInterval(start_ev=160, end_ev=180)
                eels_edge = EELSQuantificationController.EELSEdge(signal_eels_interval=signal_eels_interval, fit_eels_intervals=[fit_eels_interval])
                q.append_edge(eels_edge)
                eels_data_item = self.__create_spectrum()
                document_model.append_data_item(eels_data_item)
                eels_display_item = document_model.get_display_item_for_data_item(eels_data_item)
                qd = qm.create_eels_quantification_display(q, eels_display_item, eels_data_item)
                self.assertEqual(1, len(qm.get_eels_quantification_displays(q)))
                qd.show_eels_edge(eels_edge)
                self.assertTrue(qd.is_eels_edge_visible(eels_edge))
            document_model = DocumentModel.DocumentModel(profile=profile_context.create_profile())
            with contextlib.closing(document_model):
                qm = EELSQuantificationController.EELSQuantificationManager.get_instance(document_model)
                q = qm.eels_quantifications[0]
                eels_edge = q.eels_edges[0]
                self.assertEqual(1, len(qm.get_eels_quantification_displays(q)))
                qd = qm.get_eels_quantification_displays(q)[0]
                self.assertEqual(document_model.data_items[0], qd.eels_data_item)
                self.assertEqual(document_model.display_items[0], qd.eels_display_item)
                self.assertTrue(qd.is_eels_edge_visible(eels_edge))
                self.assertEqual(1, len(qd.eels_edge_displays))
                self.assertEqual(3, len(eels_display_item.display_data_channels))
                self.assertEqual(3, len(eels_display_item.display_layers))
                self.assertEqual(2, len(eels_display_item.graphics))
                self.assertEqual(2, eels_display_item.display_layers[0]["data_index"])
                self.assertEqual(1, eels_display_item.display_layers[1]["data_index"])
                self.assertEqual(0, eels_display_item.display_layers[2]["data_index"])  # original data should be at the back
                self.assertEqual(1, len(document_model.display_items))
                self.assertEqual(3, len(document_model.data_items))
                self.assertEqual(1, len(document_model.computations))
                self.assertEqual(2, len(document_model.data_structures))

    def test_reloading_quantification_display_with_edges_and_hiding_edge_succeeds(self):
        # this tests whetehr reloading quantification display properly populates the edge displays with their
        # data items, layers, computations, etc.
        with create_memory_profile_context() as profile_context:
            document_model = DocumentModel.DocumentModel(profile=profile_context.create_profile())
            with contextlib.closing(document_model):
                qm = EELSQuantificationController.EELSQuantificationManager.get_instance(document_model)
                q = qm.create_eels_quantification()
                signal_eels_interval = EELSQuantificationController.EELSInterval(start_ev=188, end_ev=208)
                fit_eels_interval = EELSQuantificationController.EELSInterval(start_ev=160, end_ev=180)
                eels_edge = EELSQuantificationController.EELSEdge(signal_eels_interval=signal_eels_interval, fit_eels_intervals=[fit_eels_interval])
                q.append_edge(eels_edge)
                eels_data_item = self.__create_spectrum()
                document_model.append_data_item(eels_data_item)
                eels_display_item = document_model.get_display_item_for_data_item(eels_data_item)
                qd = qm.create_eels_quantification_display(q, eels_display_item, eels_data_item)
                self.assertEqual(1, len(qm.get_eels_quantification_displays(q)))
                qd.show_eels_edge(eels_edge)
                self.assertTrue(qd.is_eels_edge_visible(eels_edge))
            document_model = DocumentModel.DocumentModel(profile=profile_context.create_profile())
            with contextlib.closing(document_model):
                qm = EELSQuantificationController.EELSQuantificationManager.get_instance(document_model)
                q = qm.eels_quantifications[0]
                eels_edge = q.eels_edges[0]
                self.assertEqual(1, len(qm.get_eels_quantification_displays(q)))
                qd = qm.get_eels_quantification_displays(q)[0]
                eels_display_item = qd.eels_display_item
                self.assertTrue(qd.is_eels_edge_visible(eels_edge))
                qd.hide_eels_edge(qd.eels_edge_displays[0].eels_edge)
                self.assertEqual(0, len(qd.eels_edge_displays))
                self.assertEqual(1, len(eels_display_item.display_data_channels))
                self.assertEqual(1, len(eels_display_item.display_layers))
                self.assertEqual(0, len(eels_display_item.graphics))
                self.assertEqual(0, eels_display_item.display_layers[0]["data_index"])  # original data should be at the back
                self.assertEqual(1, len(document_model.display_items))
                self.assertEqual(1, len(document_model.data_items))
                self.assertEqual(0, len(document_model.computations))
                self.assertEqual(2, len(document_model.data_structures))

    def test_removing_quantification_display_removes_associated_data_structure(self):
        with create_memory_profile_context() as profile_context:
            document_model = DocumentModel.DocumentModel(profile=profile_context.create_profile())
            with contextlib.closing(document_model):
                qm = EELSQuantificationController.EELSQuantificationManager.get_instance(document_model)
                q = qm.create_eels_quantification()
                signal_eels_interval = EELSQuantificationController.EELSInterval(start_ev=188, end_ev=208)
                fit_eels_interval = EELSQuantificationController.EELSInterval(start_ev=160, end_ev=180)
                eels_edge = EELSQuantificationController.EELSEdge(signal_eels_interval=signal_eels_interval, fit_eels_intervals=[fit_eels_interval])
                q.append_edge(eels_edge)
                eels_data_item = self.__create_spectrum()
                document_model.append_data_item(eels_data_item)
                eels_display_item = document_model.get_display_item_for_data_item(eels_data_item)
                qd = qm.create_eels_quantification_display(q, eels_display_item, eels_data_item)
                qd.show_eels_edge(eels_edge)
                qm.destroy_eels_quantification_display(qd)
                self.assertEqual(0, len(qm.get_eels_quantification_displays(q)))
                self.assertEqual(1, len(document_model.data_structures))
                self.assertEqual(1, len(document_model.display_items))
                self.assertEqual(1, len(document_model.data_items))
                self.assertEqual(1, len(eels_display_item.display_data_channels))
                self.assertEqual(1, len(eels_display_item.display_layers))
                self.assertEqual(0, len(eels_display_item.graphics))

    # test_removing_edge_removes_layers_computation_etc
    # test_eels_quantification_display_loads_out_of_order_from_quantification
    # test_eels_quantification_disconnects_if_data_structure_deleted
    # test_associated_data_structure_is_removed_when_display_item_removed
    # test_orphan_associated_data_structures_are_removed_on_reload
    # test_deleting_last_background_deletes_edge
    # test_removing_eels_quantification_removes_associated_data_structure
