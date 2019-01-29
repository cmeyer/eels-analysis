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

    def test_adding_edge_configures_associated_data_structure(self):
        document_model = DocumentModel.DocumentModel()
        with contextlib.closing(document_model):
            qm = EELSQuantificationController.EELSQuantificationManager.get_instance(document_model)
            eels_data_item = self.__create_spectrum()
            document_model.append_data_item(eels_data_item)
            eels_display_item = document_model.get_display_item_for_data_item(eels_data_item)
            signal_eels_interval = EELSQuantificationController.EELSInterval(start_ev=188, end_ev=208)
            signal_interval_graphic = Graphics.IntervalGraphic()
            signal_interval_graphic.interval = signal_eels_interval.to_fractional_interval(eels_data_item.data_shape[-1], eels_data_item.dimensional_calibrations[-1])
            eels_display_item.add_graphic(signal_interval_graphic)
            eels_edge = qm.add_eels_edge_from_interval_graphic(eels_display_item, eels_data_item, signal_interval_graphic)
            # check general properties
            self.assertEqual(1, len(document_model.data_structures))
            # check the eels quantification display structure
            self.assertEqual("nion.eels_quantification_display", document_model.data_structures[0].structure_type)
            self.assertEqual(eels_data_item, document_model.data_structures[0].get_referenced_object("eels_data_item"))
            self.assertEqual(eels_display_item, document_model.data_structures[0].get_referenced_object("eels_display_item"))
            self.assertEqual(signal_eels_interval._write_to_dict(), document_model.data_structures[0].get_property_value("signal_eels_interval"))
            self.assertEqual(2, len(document_model.data_structures[0].get_property_value("fit_eels_intervals")))

    def test_adding_edge_from_signal_interval_configures_display_layers(self):
        document_model = DocumentModel.DocumentModel()
        with contextlib.closing(document_model):
            qm = EELSQuantificationController.EELSQuantificationManager.get_instance(document_model)
            eels_data_item = self.__create_spectrum()
            document_model.append_data_item(eels_data_item)
            eels_display_item = document_model.get_display_item_for_data_item(eels_data_item)
            signal_eels_interval = EELSQuantificationController.EELSInterval(start_ev=188, end_ev=208)
            signal_interval_graphic = Graphics.IntervalGraphic()
            signal_interval_graphic.interval = signal_eels_interval.to_fractional_interval(eels_data_item.data_shape[-1], eels_data_item.dimensional_calibrations[-1])
            eels_display_item.add_graphic(signal_interval_graphic)
            eels_edge = qm.add_eels_edge_from_interval_graphic(eels_display_item, eels_data_item, signal_interval_graphic)
            self.assertEqual(eels_display_item, eels_edge.eels_display_item)
            self.assertEqual(eels_data_item, eels_edge.eels_data_item)
            self.assertEqual(3, len(eels_display_item.display_data_channels))
            self.assertEqual(3, len(eels_display_item.display_layers))
            self.assertEqual(3, len(eels_display_item.graphics))
            self.assertEqual(2, eels_display_item.display_layers[0]["data_index"])
            self.assertEqual(1, eels_display_item.display_layers[1]["data_index"])
            self.assertEqual(0, eels_display_item.display_layers[2]["data_index"])  # original data should be at the back
            self.assertEqual(1, len(document_model.display_items))
            self.assertEqual(3, len(document_model.data_items))
            self.assertEqual(1, len(document_model.computations))
            self.assertEqual(1, len(document_model.data_structures))

    def test_removing_edge_removes_all_items(self):
        document_model = DocumentModel.DocumentModel()
        with contextlib.closing(document_model):
            qm = EELSQuantificationController.EELSQuantificationManager.get_instance(document_model)
            eels_data_item = self.__create_spectrum()
            document_model.append_data_item(eels_data_item)
            eels_display_item = document_model.get_display_item_for_data_item(eels_data_item)
            signal_eels_interval = EELSQuantificationController.EELSInterval(start_ev=188, end_ev=208)
            signal_interval_graphic = Graphics.IntervalGraphic()
            signal_interval_graphic.interval = signal_eels_interval.to_fractional_interval(eels_data_item.data_shape[-1], eels_data_item.dimensional_calibrations[-1])
            eels_display_item.add_graphic(signal_interval_graphic)
            eels_edge = qm.add_eels_edge_from_interval_graphic(eels_display_item, eels_data_item, signal_interval_graphic)
            qm.remove_eels_edge(eels_edge)
            self.assertEqual(1, len(eels_display_item.display_data_channels))
            self.assertEqual(1, len(eels_display_item.display_layers))
            self.assertEqual(0, len(eels_display_item.graphics))
            self.assertEqual(0, eels_display_item.display_layers[0]["data_index"])  # original data should be at the back
            self.assertEqual(1, len(document_model.display_items))
            self.assertEqual(1, len(document_model.data_items))
            self.assertEqual(0, len(document_model.computations))
            self.assertEqual(0, len(document_model.data_structures))

    def test_hiding_edge_cleans_up_dependent_items(self):
        document_model = DocumentModel.DocumentModel()
        with contextlib.closing(document_model):
            qm = EELSQuantificationController.EELSQuantificationManager.get_instance(document_model)
            eels_data_item = self.__create_spectrum()
            document_model.append_data_item(eels_data_item)
            eels_display_item = document_model.get_display_item_for_data_item(eels_data_item)
            signal_eels_interval = EELSQuantificationController.EELSInterval(start_ev=188, end_ev=208)
            signal_interval_graphic = Graphics.IntervalGraphic()
            signal_interval_graphic.interval = signal_eels_interval.to_fractional_interval(eels_data_item.data_shape[-1], eels_data_item.dimensional_calibrations[-1])
            eels_display_item.add_graphic(signal_interval_graphic)
            eels_edge = qm.add_eels_edge_from_interval_graphic(eels_display_item, eels_data_item, signal_interval_graphic)
            eels_edge.hide()
            self.assertEqual(1, len(eels_display_item.display_data_channels))
            self.assertEqual(1, len(eels_display_item.display_layers))
            self.assertEqual(0, len(eels_display_item.graphics))
            self.assertEqual(0, eels_display_item.display_layers[0]["data_index"])  # original data should be at the back
            self.assertEqual(1, len(document_model.display_items))
            self.assertEqual(1, len(document_model.data_items))
            self.assertEqual(0, len(document_model.computations))
            self.assertEqual(1, len(document_model.data_structures))

    def test_showing_hidden_edge_configures_items(self):
        document_model = DocumentModel.DocumentModel()
        with contextlib.closing(document_model):
            qm = EELSQuantificationController.EELSQuantificationManager.get_instance(document_model)
            eels_data_item = self.__create_spectrum()
            document_model.append_data_item(eels_data_item)
            eels_display_item = document_model.get_display_item_for_data_item(eels_data_item)
            signal_eels_interval = EELSQuantificationController.EELSInterval(start_ev=188, end_ev=208)
            signal_interval_graphic = Graphics.IntervalGraphic()
            signal_interval_graphic.interval = signal_eels_interval.to_fractional_interval(eels_data_item.data_shape[-1], eels_data_item.dimensional_calibrations[-1])
            eels_display_item.add_graphic(signal_interval_graphic)
            eels_edge = qm.add_eels_edge_from_interval_graphic(eels_display_item, eels_data_item, signal_interval_graphic)
            eels_edge.hide()
            eels_edge.show()
            self.assertEqual(eels_display_item, eels_edge.eels_display_item)
            self.assertEqual(eels_data_item, eels_edge.eels_data_item)
            self.assertEqual(3, len(eels_display_item.display_data_channels))
            self.assertEqual(3, len(eels_display_item.display_layers))
            self.assertEqual(3, len(eels_display_item.graphics))
            self.assertEqual(2, eels_display_item.display_layers[0]["data_index"])
            self.assertEqual(1, eels_display_item.display_layers[1]["data_index"])
            self.assertEqual(0, eels_display_item.display_layers[2]["data_index"])  # original data should be at the back
            self.assertEqual(1, len(document_model.display_items))
            self.assertEqual(3, len(document_model.data_items))
            self.assertEqual(1, len(document_model.computations))
            self.assertEqual(1, len(document_model.data_structures))

    def test_signal_interval_graphic_and_eels_edge_signal_interval_are_connected(self):
        document_model = DocumentModel.DocumentModel()
        with contextlib.closing(document_model):
            qm = EELSQuantificationController.EELSQuantificationManager.get_instance(document_model)
            eels_data_item = self.__create_spectrum()
            document_model.append_data_item(eels_data_item)
            eels_display_item = document_model.get_display_item_for_data_item(eels_data_item)
            signal_eels_interval = EELSQuantificationController.EELSInterval(start_ev=188, end_ev=208)
            signal_interval_graphic = Graphics.IntervalGraphic()
            signal_interval_graphic.interval = signal_eels_interval.to_fractional_interval(eels_data_item.data_shape[-1], eels_data_item.dimensional_calibrations[-1])
            eels_display_item.add_graphic(signal_interval_graphic)
            eels_edge = qm.add_eels_edge_from_interval_graphic(eels_display_item, eels_data_item, signal_interval_graphic)
            self.__compare_intervals(eels_data_item, eels_edge.signal_eels_interval, signal_interval_graphic)
            signal_interval_graphic.interval = (0.095, 0.105)
            self.__compare_intervals(eels_data_item, eels_edge.signal_eels_interval, signal_interval_graphic)
            eels_edge.signal_eels_interval = EELSQuantificationController.EELSInterval(start_ev=200, end_ev=220)
            self.__compare_intervals(eels_data_item, eels_edge.signal_eels_interval, signal_interval_graphic)

            did_change = False
            def changed(name: str) -> None:
                nonlocal did_change
                did_change = True

            with contextlib.closing(eels_edge.property_changed_event.listen(changed)):
                signal_interval_graphic.interval = (0.096, 0.106)
                self.assertTrue(did_change)

    def test_signal_interval_graphic_and_eels_edge_signal_interval_are_connected_after_reload(self):
        with create_memory_profile_context() as profile_context:
            document_model = DocumentModel.DocumentModel(profile=profile_context.create_profile())
            with contextlib.closing(document_model):
                qm = EELSQuantificationController.EELSQuantificationManager.get_instance(document_model)
                eels_data_item = self.__create_spectrum()
                document_model.append_data_item(eels_data_item)
                eels_display_item = document_model.get_display_item_for_data_item(eels_data_item)
                signal_eels_interval = EELSQuantificationController.EELSInterval(start_ev=188, end_ev=208)
                signal_interval_graphic = Graphics.IntervalGraphic()
                signal_interval_graphic.interval = signal_eels_interval.to_fractional_interval(eels_data_item.data_shape[-1], eels_data_item.dimensional_calibrations[-1])
                eels_display_item.add_graphic(signal_interval_graphic)
                eels_edge = qm.add_eels_edge_from_interval_graphic(eels_display_item, eels_data_item, signal_interval_graphic)
            document_model = DocumentModel.DocumentModel(profile=profile_context.create_profile())
            with contextlib.closing(document_model):
                qm = EELSQuantificationController.EELSQuantificationManager.get_instance(document_model)
                eels_data_item = document_model.data_items[0]
                eels_edge = qm.eels_edges[0]
                signal_interval_graphic = eels_edge.signal_interval_graphic
                self.__compare_intervals(eels_data_item, eels_edge.signal_eels_interval, signal_interval_graphic)
                signal_interval_graphic.interval = (0.095, 0.105)
                self.__compare_intervals(eels_data_item, eels_edge.signal_eels_interval, signal_interval_graphic)
                eels_edge.signal_eels_interval = EELSQuantificationController.EELSInterval(start_ev=190, end_ev=210)
                self.__compare_intervals(eels_data_item, eels_edge.signal_eels_interval, signal_interval_graphic)

                did_change = False
                def changed(name: str) -> None:
                    nonlocal did_change
                    did_change = True

                with contextlib.closing(eels_edge.property_changed_event.listen(changed)):
                    signal_interval_graphic.interval = (0.096, 0.106)
                    self.assertTrue(did_change)

    def test_fit_interval_graphic_and_eels_edge_fit_interval_are_connected(self):
        document_model = DocumentModel.DocumentModel()
        with contextlib.closing(document_model):
            qm = EELSQuantificationController.EELSQuantificationManager.get_instance(document_model)
            eels_data_item = self.__create_spectrum()
            document_model.append_data_item(eels_data_item)
            eels_display_item = document_model.get_display_item_for_data_item(eels_data_item)
            signal_eels_interval = EELSQuantificationController.EELSInterval(start_ev=188, end_ev=208)
            signal_interval_graphic = Graphics.IntervalGraphic()
            signal_interval_graphic.interval = signal_eels_interval.to_fractional_interval(eels_data_item.data_shape[-1], eels_data_item.dimensional_calibrations[-1])
            eels_display_item.add_graphic(signal_interval_graphic)
            eels_edge = qm.add_eels_edge_from_interval_graphic(eels_display_item, eels_data_item, signal_interval_graphic)
            self.__compare_intervals(eels_data_item, eels_edge.fit_eels_intervals[0], eels_display_item.graphics[1])
            eels_display_item.graphics[1].interval = (0.04, 0.05)
            self.__compare_intervals(eels_data_item, eels_edge.fit_eels_intervals[0], eels_display_item.graphics[1])
            eels_edge.fit_eels_intervals[0] = EELSQuantificationController.EELSInterval(start_ev=40, end_ev=80)
            self.__compare_intervals(eels_data_item, eels_edge.signal_eels_interval, signal_interval_graphic)

    def test_adding_or_removing_fit_interval_to_eels_edge_adds_or_removes_fit_interval_graphic(self):
        document_model = DocumentModel.DocumentModel()
        with contextlib.closing(document_model):
            qm = EELSQuantificationController.EELSQuantificationManager.get_instance(document_model)
            eels_data_item = self.__create_spectrum()
            document_model.append_data_item(eels_data_item)
            eels_display_item = document_model.get_display_item_for_data_item(eels_data_item)
            signal_eels_interval = EELSQuantificationController.EELSInterval(start_ev=188, end_ev=208)
            signal_interval_graphic = Graphics.IntervalGraphic()
            signal_interval_graphic.interval = signal_eels_interval.to_fractional_interval(eels_data_item.data_shape[-1], eels_data_item.dimensional_calibrations[-1])
            eels_display_item.add_graphic(signal_interval_graphic)
            eels_edge = qm.add_eels_edge_from_interval_graphic(eels_display_item, eels_data_item, signal_interval_graphic)
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

    def test_deleting_signal_hides_edge(self):
        document_model = DocumentModel.DocumentModel()
        with contextlib.closing(document_model):
            qm = EELSQuantificationController.EELSQuantificationManager.get_instance(document_model)
            eels_data_item = self.__create_spectrum()
            document_model.append_data_item(eels_data_item)
            eels_display_item = document_model.get_display_item_for_data_item(eels_data_item)
            signal_eels_interval = EELSQuantificationController.EELSInterval(start_ev=188, end_ev=208)
            signal_interval_graphic = Graphics.IntervalGraphic()
            signal_interval_graphic.interval = signal_eels_interval.to_fractional_interval(eels_data_item.data_shape[-1], eels_data_item.dimensional_calibrations[-1])
            eels_display_item.add_graphic(signal_interval_graphic)
            eels_edge = qm.add_eels_edge_from_interval_graphic(eels_display_item, eels_data_item, signal_interval_graphic)
            eels_display_item.remove_graphic(eels_display_item.graphics[0])
            self.assertEqual(1, len(eels_display_item.display_data_channels))
            self.assertEqual(1, len(eels_display_item.display_layers))
            self.assertEqual(0, len(eels_display_item.graphics))
            self.assertEqual(0, eels_display_item.display_layers[0]["data_index"])  # original data should be at the back
            self.assertEqual(1, len(document_model.display_items))
            self.assertEqual(1, len(document_model.data_items))
            self.assertEqual(0, len(document_model.computations))
            self.assertEqual(1, len(document_model.data_structures))

    def test_deleting_signal_hides_edge_if_one_background_deleted_first(self):
        document_model = DocumentModel.DocumentModel()
        with contextlib.closing(document_model):
            qm = EELSQuantificationController.EELSQuantificationManager.get_instance(document_model)
            eels_data_item = self.__create_spectrum()
            document_model.append_data_item(eels_data_item)
            eels_display_item = document_model.get_display_item_for_data_item(eels_data_item)
            signal_eels_interval = EELSQuantificationController.EELSInterval(start_ev=188, end_ev=208)
            signal_interval_graphic = Graphics.IntervalGraphic()
            signal_interval_graphic.interval = signal_eels_interval.to_fractional_interval(eels_data_item.data_shape[-1], eels_data_item.dimensional_calibrations[-1])
            eels_display_item.add_graphic(signal_interval_graphic)
            eels_edge = qm.add_eels_edge_from_interval_graphic(eels_display_item, eels_data_item, signal_interval_graphic)
            eels_display_item.remove_graphic(eels_display_item.graphics[2])
            eels_display_item.remove_graphic(eels_display_item.graphics[0])
            self.assertEqual(1, len(eels_display_item.display_data_channels))
            self.assertEqual(1, len(eels_display_item.display_layers))
            self.assertEqual(0, len(eels_display_item.graphics))
            self.assertEqual(0, eels_display_item.display_layers[0]["data_index"])  # original data should be at the back
            self.assertEqual(1, len(document_model.display_items))
            self.assertEqual(1, len(document_model.data_items))
            self.assertEqual(0, len(document_model.computations))
            self.assertEqual(1, len(document_model.data_structures))

    def test_deleting_computation_hides_edge(self):
        document_model = DocumentModel.DocumentModel()
        with contextlib.closing(document_model):
            qm = EELSQuantificationController.EELSQuantificationManager.get_instance(document_model)
            eels_data_item = self.__create_spectrum()
            document_model.append_data_item(eels_data_item)
            eels_display_item = document_model.get_display_item_for_data_item(eels_data_item)
            signal_eels_interval = EELSQuantificationController.EELSInterval(start_ev=188, end_ev=208)
            signal_interval_graphic = Graphics.IntervalGraphic()
            signal_interval_graphic.interval = signal_eels_interval.to_fractional_interval(eels_data_item.data_shape[-1], eels_data_item.dimensional_calibrations[-1])
            eels_display_item.add_graphic(signal_interval_graphic)
            eels_edge = qm.add_eels_edge_from_interval_graphic(eels_display_item, eels_data_item, signal_interval_graphic)
            document_model.remove_computation(document_model.computations[0])
            self.assertEqual(1, len(eels_display_item.display_data_channels))
            self.assertEqual(1, len(eels_display_item.display_layers))
            self.assertEqual(0, len(eels_display_item.graphics))
            self.assertEqual(0, eels_display_item.display_layers[0]["data_index"])  # original data should be at the back
            self.assertEqual(1, len(document_model.display_items))
            self.assertEqual(1, len(document_model.data_items))
            self.assertEqual(0, len(document_model.computations))
            self.assertEqual(1, len(document_model.data_structures))

    def test_deleting_background_data_item_hides_edge(self):
        document_model = DocumentModel.DocumentModel()
        with contextlib.closing(document_model):
            qm = EELSQuantificationController.EELSQuantificationManager.get_instance(document_model)
            eels_data_item = self.__create_spectrum()
            document_model.append_data_item(eels_data_item)
            eels_display_item = document_model.get_display_item_for_data_item(eels_data_item)
            signal_eels_interval = EELSQuantificationController.EELSInterval(start_ev=188, end_ev=208)
            signal_interval_graphic = Graphics.IntervalGraphic()
            signal_interval_graphic.interval = signal_eels_interval.to_fractional_interval(eels_data_item.data_shape[-1], eels_data_item.dimensional_calibrations[-1])
            eels_display_item.add_graphic(signal_interval_graphic)
            eels_edge = qm.add_eels_edge_from_interval_graphic(eels_display_item, eels_data_item, signal_interval_graphic)
            document_model.remove_data_item(document_model.data_items[2])
            self.assertEqual(1, len(eels_display_item.display_data_channels))
            self.assertEqual(1, len(eels_display_item.display_layers))
            self.assertEqual(0, len(eels_display_item.graphics))
            self.assertEqual(0, eels_display_item.display_layers[0]["data_index"])  # original data should be at the back
            self.assertEqual(1, len(document_model.display_items))
            self.assertEqual(1, len(document_model.data_items))
            self.assertEqual(0, len(document_model.computations))
            self.assertEqual(1, len(document_model.data_structures))

    def test_deleting_signal_data_item_hides_edge(self):
        document_model = DocumentModel.DocumentModel()
        with contextlib.closing(document_model):
            qm = EELSQuantificationController.EELSQuantificationManager.get_instance(document_model)
            eels_data_item = self.__create_spectrum()
            document_model.append_data_item(eels_data_item)
            eels_display_item = document_model.get_display_item_for_data_item(eels_data_item)
            signal_eels_interval = EELSQuantificationController.EELSInterval(start_ev=188, end_ev=208)
            signal_interval_graphic = Graphics.IntervalGraphic()
            signal_interval_graphic.interval = signal_eels_interval.to_fractional_interval(eels_data_item.data_shape[-1], eels_data_item.dimensional_calibrations[-1])
            eels_display_item.add_graphic(signal_interval_graphic)
            eels_edge = qm.add_eels_edge_from_interval_graphic(eels_display_item, eels_data_item, signal_interval_graphic)
            document_model.remove_data_item(document_model.data_items[1])
            self.assertEqual(1, len(eels_display_item.display_data_channels))
            self.assertEqual(1, len(eels_display_item.display_layers))
            self.assertEqual(0, len(eels_display_item.graphics))
            self.assertEqual(0, eels_display_item.display_layers[0]["data_index"])  # original data should be at the back
            self.assertEqual(1, len(document_model.display_items))
            self.assertEqual(1, len(document_model.data_items))
            self.assertEqual(0, len(document_model.computations))
            self.assertEqual(1, len(document_model.data_structures))

    def test_reloading_quantification_with_edges(self):
        with create_memory_profile_context() as profile_context:
            document_model = DocumentModel.DocumentModel(profile=profile_context.create_profile())
            with contextlib.closing(document_model):
                qm = EELSQuantificationController.EELSQuantificationManager.get_instance(document_model)
                eels_data_item = self.__create_spectrum()
                document_model.append_data_item(eels_data_item)
                eels_display_item = document_model.get_display_item_for_data_item(eels_data_item)
                signal_eels_interval = EELSQuantificationController.EELSInterval(start_ev=200, end_ev=220)
                signal_interval_graphic = Graphics.IntervalGraphic()
                signal_interval_graphic.interval = signal_eels_interval.to_fractional_interval(eels_data_item.data_shape[-1], eels_data_item.dimensional_calibrations[-1])
                eels_display_item.add_graphic(signal_interval_graphic)
                eels_edge = qm.add_eels_edge_from_interval_graphic(eels_display_item, eels_data_item, signal_interval_graphic)
                eels_edge.hide()
            document_model = DocumentModel.DocumentModel(profile=profile_context.create_profile())
            with contextlib.closing(document_model):
                qm = EELSQuantificationController.EELSQuantificationManager.get_instance(document_model)
                self.assertEqual(1, len(qm.eels_edges))
                eels_edge = qm.eels_edges[0]
                self.assertAlmostEqual(200.0, eels_edge.signal_eels_interval.start_ev)
                self.assertAlmostEqual(220.0, eels_edge.signal_eels_interval.end_ev)
                self.assertEqual(2, len(eels_edge.fit_eels_intervals))
                self.assertAlmostEqual(160.0, eels_edge.fit_eels_intervals[0].start_ev)
                self.assertAlmostEqual(180.0, eels_edge.fit_eels_intervals[0].end_ev)

    def test_reloading_quantification_display_with_edges(self):
        with create_memory_profile_context() as profile_context:
            document_model = DocumentModel.DocumentModel(profile=profile_context.create_profile())
            with contextlib.closing(document_model):
                qm = EELSQuantificationController.EELSQuantificationManager.get_instance(document_model)
                eels_data_item = self.__create_spectrum()
                document_model.append_data_item(eels_data_item)
                eels_display_item = document_model.get_display_item_for_data_item(eels_data_item)
                signal_eels_interval = EELSQuantificationController.EELSInterval(start_ev=200, end_ev=220)
                signal_interval_graphic = Graphics.IntervalGraphic()
                signal_interval_graphic.interval = signal_eels_interval.to_fractional_interval(eels_data_item.data_shape[-1], eels_data_item.dimensional_calibrations[-1])
                eels_display_item.add_graphic(signal_interval_graphic)
                eels_edge = qm.add_eels_edge_from_interval_graphic(eels_display_item, eels_data_item, signal_interval_graphic)
            document_model = DocumentModel.DocumentModel(profile=profile_context.create_profile())
            with contextlib.closing(document_model):
                qm = EELSQuantificationController.EELSQuantificationManager.get_instance(document_model)
                eels_edge = qm.eels_edges[0]
                self.assertEqual(document_model.data_items[0], eels_edge.eels_data_item)
                self.assertEqual(document_model.display_items[0], eels_edge.eels_display_item)
                self.assertEqual(3, len(eels_display_item.display_data_channels))
                self.assertEqual(3, len(eels_display_item.display_layers))
                self.assertEqual(3, len(eels_display_item.graphics))
                self.assertEqual(2, eels_display_item.display_layers[0]["data_index"])
                self.assertEqual(1, eels_display_item.display_layers[1]["data_index"])
                self.assertEqual(0, eels_display_item.display_layers[2]["data_index"])  # original data should be at the back
                self.assertEqual(1, len(document_model.display_items))
                self.assertEqual(3, len(document_model.data_items))
                self.assertEqual(document_model.data_items[2], eels_edge.background_data_item)
                self.assertEqual(document_model.data_items[1], eels_edge.signal_data_item)
                self.assertEqual(1, len(document_model.computations))
                self.assertEqual(1, len(document_model.data_structures))

    def test_reloading_quantification_display_with_edges_and_hiding_edge_succeeds(self):
        # this tests whether reloading quantification display properly populates the edge displays with their
        # data items, layers, computations, etc.
        with create_memory_profile_context() as profile_context:
            document_model = DocumentModel.DocumentModel(profile=profile_context.create_profile())
            with contextlib.closing(document_model):
                qm = EELSQuantificationController.EELSQuantificationManager.get_instance(document_model)
                eels_data_item = self.__create_spectrum()
                document_model.append_data_item(eels_data_item)
                eels_display_item = document_model.get_display_item_for_data_item(eels_data_item)
                signal_eels_interval = EELSQuantificationController.EELSInterval(start_ev=200, end_ev=220)
                signal_interval_graphic = Graphics.IntervalGraphic()
                signal_interval_graphic.interval = signal_eels_interval.to_fractional_interval(eels_data_item.data_shape[-1], eels_data_item.dimensional_calibrations[-1])
                eels_display_item.add_graphic(signal_interval_graphic)
                eels_edge = qm.add_eels_edge_from_interval_graphic(eels_display_item, eels_data_item, signal_interval_graphic)
            document_model = DocumentModel.DocumentModel(profile=profile_context.create_profile())
            with contextlib.closing(document_model):
                qm = EELSQuantificationController.EELSQuantificationManager.get_instance(document_model)
                eels_edge = qm.eels_edges[0]
                eels_display_item = eels_edge.eels_display_item
                eels_edge.hide()
                self.assertEqual(1, len(qm.eels_edges))
                self.assertEqual(1, len(eels_display_item.display_data_channels))
                self.assertEqual(1, len(eels_display_item.display_layers))
                self.assertEqual(0, len(eels_display_item.graphics))
                self.assertEqual(0, eels_display_item.display_layers[0]["data_index"])  # original data should be at the back
                self.assertEqual(1, len(document_model.display_items))
                self.assertEqual(1, len(document_model.data_items))
                self.assertEqual(0, len(document_model.computations))
                self.assertEqual(1, len(document_model.data_structures))

    def test_removing_quantification_display_removes_associated_data_structure(self):
        document_model = DocumentModel.DocumentModel()
        with contextlib.closing(document_model):
            qm = EELSQuantificationController.EELSQuantificationManager.get_instance(document_model)
            eels_data_item = self.__create_spectrum()
            document_model.append_data_item(eels_data_item)
            eels_display_item = document_model.get_display_item_for_data_item(eels_data_item)
            signal_eels_interval = EELSQuantificationController.EELSInterval(start_ev=200, end_ev=220)
            signal_interval_graphic = Graphics.IntervalGraphic()
            signal_interval_graphic.interval = signal_eels_interval.to_fractional_interval(eels_data_item.data_shape[-1], eels_data_item.dimensional_calibrations[-1])
            eels_display_item.add_graphic(signal_interval_graphic)
            eels_edge = qm.add_eels_edge_from_interval_graphic(eels_display_item, eels_data_item, signal_interval_graphic)
            qm.remove_eels_edge(eels_edge)
            self.assertEqual(0, len(document_model.data_structures))
            self.assertEqual(1, len(document_model.display_items))
            self.assertEqual(1, len(document_model.data_items))
            self.assertEqual(1, len(eels_display_item.display_data_channels))
            self.assertEqual(1, len(eels_display_item.display_layers))
            self.assertEqual(0, len(eels_display_item.graphics))

    def test_removing_eels_display_item_removes_associated_data_structure(self):
        document_model = DocumentModel.DocumentModel()
        with contextlib.closing(document_model):
            qm = EELSQuantificationController.EELSQuantificationManager.get_instance(document_model)
            eels_data_item = self.__create_spectrum()
            document_model.append_data_item(eels_data_item)
            eels_display_item = document_model.get_display_item_for_data_item(eels_data_item)
            signal_eels_interval = EELSQuantificationController.EELSInterval(start_ev=200, end_ev=220)
            signal_interval_graphic = Graphics.IntervalGraphic()
            signal_interval_graphic.interval = signal_eels_interval.to_fractional_interval(eels_data_item.data_shape[-1], eels_data_item.dimensional_calibrations[-1])
            eels_display_item.add_graphic(signal_interval_graphic)
            eels_edge = qm.add_eels_edge_from_interval_graphic(eels_display_item, eels_data_item, signal_interval_graphic)
            document_model.remove_display_item(eels_display_item)
            self.assertEqual(0, len(document_model.data_structures))
            self.assertEqual(0, len(document_model.display_items))
            self.assertEqual(0, len(document_model.data_items))

    def test_removing_eels_data_item_removes_associated_data_structure(self):
        document_model = DocumentModel.DocumentModel()
        with contextlib.closing(document_model):
            qm = EELSQuantificationController.EELSQuantificationManager.get_instance(document_model)
            eels_data_item = self.__create_spectrum()
            document_model.append_data_item(eels_data_item)
            eels_display_item = document_model.get_display_item_for_data_item(eels_data_item)
            signal_eels_interval = EELSQuantificationController.EELSInterval(start_ev=200, end_ev=220)
            signal_interval_graphic = Graphics.IntervalGraphic()
            signal_interval_graphic.interval = signal_eels_interval.to_fractional_interval(eels_data_item.data_shape[-1], eels_data_item.dimensional_calibrations[-1])
            eels_display_item.add_graphic(signal_interval_graphic)
            eels_edge = qm.add_eels_edge_from_interval_graphic(eels_display_item, eels_data_item, signal_interval_graphic)
            document_model.remove_data_item(eels_data_item)
            self.assertEqual(0, len(document_model.data_structures))
            self.assertEqual(0, len(document_model.display_items))
            self.assertEqual(0, len(document_model.data_items))

    def test_eels_edges_model_tracks_edges_added_to_and_removed_from_display_item(self):
        document_model = DocumentModel.DocumentModel()
        with contextlib.closing(document_model):
            qm = EELSQuantificationController.EELSQuantificationManager.get_instance(document_model)
            eels_data_item = self.__create_spectrum()
            document_model.append_data_item(eels_data_item)
            eels_display_item = document_model.get_display_item_for_data_item(eels_data_item)
            signal_eels_interval = EELSQuantificationController.EELSInterval(start_ev=200, end_ev=220)
            signal_interval_graphic = Graphics.IntervalGraphic()
            signal_interval_graphic.interval = signal_eels_interval.to_fractional_interval(eels_data_item.data_shape[-1], eels_data_item.dimensional_calibrations[-1])
            eels_display_item.add_graphic(signal_interval_graphic)
            eels_edges_model = qm.get_eels_edges_model_for_display_item()
            self.assertEqual(0, len(eels_edges_model.items))
            eels_edges_model.set_display_item(eels_display_item)
            eels_edge1 = qm.add_eels_edge_from_interval_graphic(eels_display_item, eels_data_item, signal_interval_graphic)
            self.assertEqual(1, len(eels_edges_model.items))
            self.assertEqual(eels_edge1, eels_edges_model.items[0])
            eels_edge2 = qm.add_eels_edge_from_interval_graphic(eels_display_item, eels_data_item, signal_interval_graphic)
            self.assertEqual(2, len(eels_edges_model.items))
            self.assertEqual(eels_edge1, eels_edges_model.items[0])
            self.assertEqual(eels_edge2, eels_edges_model.items[1])
            qm.remove_eels_edge(eels_edge1)
            self.assertEqual(1, len(eels_edges_model.items))
            self.assertEqual(eels_edge2, eels_edges_model.items[0])
            eels_edges_model.set_display_item(None)
            self.assertEqual(0, len(eels_edges_model.items))

    # test_add_edge_using_electron_shell
    # test_eels_edge_loads_properly_if_loaded_before_data_item_or_display_item
    # test_orphan_associated_data_structures_are_removed_on_reload
    # test_deleting_last_background_deletes_edge
    # test_version_1_is_enforced_on_data_structures (to allow v2 to not load in this version in the future)
    # test_change_eels_data_item_calibration_keeps_eels_intervals_constant
