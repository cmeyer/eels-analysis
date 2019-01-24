# standard libraries
import contextlib
import unittest

# local libraries
from nionswift_plugin.nion_eels_analysis import EELSQuantificationController
from nionswift_plugin.nion_eels_analysis import EELSQuantificationPanel

# third party libraries
import numpy

# local libraries
from nion.data import Calibration
from nion.data import DataAndMetadata
from nion.swift.model import DataItem
from nion.swift.model import DocumentModel
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

    def test_creating_handler_ensures_one_quantification(self):
        document_model = DocumentModel.DocumentModel()
        with contextlib.closing(document_model):
            qm = EELSQuantificationController.EELSQuantificationManager.get_instance(document_model)
            h = EELSQuantificationPanel.Handler(qm)
            self.assertEqual(1, len(qm.eels_quantifications))

    def test_creating_handler_uses_existing_quantification(self):
        document_model = DocumentModel.DocumentModel()
        with contextlib.closing(document_model):
            qm = EELSQuantificationController.EELSQuantificationManager.get_instance(document_model)
            q = qm.create_eels_quantification()
            h = EELSQuantificationPanel.Handler(qm)
            self.assertEqual(1, len(qm.eels_quantifications))

    def test_creating_handler_populates_quantification_choices_menu(self):
        document_model = DocumentModel.DocumentModel()
        with contextlib.closing(document_model):
            qm = EELSQuantificationController.EELSQuantificationManager.get_instance(document_model)
            q1 = qm.create_eels_quantification(title="Q1")
            q2 = qm.create_eels_quantification(title="Q2")
            h = EELSQuantificationPanel.Handler(qm)
            self.assertEqual(2, len(h.eels_quantification_choices.value))
            self.assertEqual("Q1", h.eels_quantification_choices.value[0])
            self.assertEqual("Q2", h.eels_quantification_choices.value[1])
            self.assertEqual(0, h.eels_quantification_index.value)

    def test_changing_title_on_quantification_updates_quantification_choices_menu(self):
        document_model = DocumentModel.DocumentModel()
        with contextlib.closing(document_model):
            qm = EELSQuantificationController.EELSQuantificationManager.get_instance(document_model)
            q1 = qm.create_eels_quantification(title="Q1")
            q2 = qm.create_eels_quantification(title="Q2")
            h = EELSQuantificationPanel.Handler(qm)
            q1.title = "Q11"
            self.assertEqual(2, len(h.eels_quantification_choices.value))
            self.assertEqual("Q11", h.eels_quantification_choices.value[0])
            self.assertEqual("Q2", h.eels_quantification_choices.value[1])
            self.assertEqual(0, h.eels_quantification_index.value)

    def test_changing_title_in_handler_updates_quantification_choices_menu(self):
        document_model = DocumentModel.DocumentModel()
        with contextlib.closing(document_model):
            qm = EELSQuantificationController.EELSQuantificationManager.get_instance(document_model)
            q1 = qm.create_eels_quantification(title="Q1")
            q2 = qm.create_eels_quantification(title="Q2")
            h = EELSQuantificationPanel.Handler(qm)
            h.eels_quantification_title = "Q11"
            self.assertEqual(2, len(h.eels_quantification_choices.value))
            self.assertEqual("Q11", h.eels_quantification_choices.value[0])
            self.assertEqual("Q2", h.eels_quantification_choices.value[1])
            self.assertEqual(0, h.eels_quantification_index.value)

    def test_changing_title_in_handler_should_keep_selection(self):
        document_model = DocumentModel.DocumentModel()
        with contextlib.closing(document_model):
            qm = EELSQuantificationController.EELSQuantificationManager.get_instance(document_model)
            q1 = qm.create_eels_quantification(title="Q1")
            q2 = qm.create_eels_quantification(title="Q2")
            h = EELSQuantificationPanel.Handler(qm)
            h.eels_quantification_index.value = 1
            h.eels_quantification_title = "Q22"
            self.assertEqual(2, len(h.eels_quantification_choices.value))
            self.assertEqual("Q1", h.eels_quantification_choices.value[0])
            self.assertEqual("Q22", h.eels_quantification_choices.value[1])
            self.assertEqual(1, h.eels_quantification_index.value)

    def test_adding_quantification_updates_quantification_choices_menu(self):
        document_model = DocumentModel.DocumentModel()
        with contextlib.closing(document_model):
            qm = EELSQuantificationController.EELSQuantificationManager.get_instance(document_model)
            q1 = qm.create_eels_quantification(title="Q1")
            q2 = qm.create_eels_quantification(title="Q2")
            h = EELSQuantificationPanel.Handler(qm)
            self.assertEqual(2, len(h.eels_quantification_choices.value))
            q3 = qm.create_eels_quantification(title="Q3")
            self.assertEqual(3, len(h.eels_quantification_choices.value))

    def test_removing_quantification_updates_quantification_choices_menu(self):
        document_model = DocumentModel.DocumentModel()
        with contextlib.closing(document_model):
            qm = EELSQuantificationController.EELSQuantificationManager.get_instance(document_model)
            q1 = qm.create_eels_quantification(title="Q1")
            q2 = qm.create_eels_quantification(title="Q2")
            h = EELSQuantificationPanel.Handler(qm)
            self.assertEqual(2, len(h.eels_quantification_choices.value))
            qm.destroy_eels_quantification(q1)
            self.assertEqual(1, len(h.eels_quantification_choices.value))

    def test_removing_quantification_updates_selection(self):
        document_model = DocumentModel.DocumentModel()
        with contextlib.closing(document_model):
            qm = EELSQuantificationController.EELSQuantificationManager.get_instance(document_model)
            q1 = qm.create_eels_quantification(title="Q1")
            q2 = qm.create_eels_quantification(title="Q2")
            q3 = qm.create_eels_quantification(title="Q3")
            q4 = qm.create_eels_quantification(title="Q4")
            h = EELSQuantificationPanel.Handler(qm)
            h.eels_quantification_index.value = 1
            self.assertEqual(q2, h.eels_quantification)
            qm.destroy_eels_quantification(q3)
            self.assertEqual(q2, h.eels_quantification)
            qm.destroy_eels_quantification(q1)
            self.assertEqual(q2, h.eels_quantification)

    def test_removing_quantification_using_handler_deletes_quantification(self):
        document_model = DocumentModel.DocumentModel()
        with contextlib.closing(document_model):
            qm = EELSQuantificationController.EELSQuantificationManager.get_instance(document_model)
            q1 = qm.create_eels_quantification(title="Q1")
            q2 = qm.create_eels_quantification(title="Q2")
            h = EELSQuantificationPanel.Handler(qm)
            h.eels_quantification_index.value = 1
            h.delete_eels_quantification()
            self.assertEqual(1, len(qm.eels_quantifications))
            self.assertEqual("Q1", h.eels_quantification_choices.value[0])

    def test_removing_last_quantification_closes_window(self):
        document_model = DocumentModel.DocumentModel()
        with contextlib.closing(document_model):
            qm = EELSQuantificationController.EELSQuantificationManager.get_instance(document_model)
            q1 = qm.create_eels_quantification(title="Q1")
            h = EELSQuantificationPanel.Handler(qm)
            request_close_ref = [False]
            def request_close():
                request_close_ref[0] = True
            h.init_window(request_close)
            h.delete_eels_quantification()
            self.assertEqual(0, len(qm.eels_quantifications))
            self.assertTrue(request_close_ref[0])

    def test_changing_selected_quantification_notifies_eels_quantification_title_change(self):
        document_model = DocumentModel.DocumentModel()
        with contextlib.closing(document_model):
            qm = EELSQuantificationController.EELSQuantificationManager.get_instance(document_model)
            q1 = qm.create_eels_quantification(title="Q1")
            q2 = qm.create_eels_quantification(title="Q2")
            h = EELSQuantificationPanel.Handler(qm)
            self.assertEqual("Q1", h.eels_quantification_title)
            notified_ref = [False]
            def property_changed(property_name):
                if property_name == "eels_quantification_title":
                    notified_ref[0] = True
            with contextlib.closing(h.property_changed_event.listen(property_changed)):
                h.eels_quantification_index.value = 1
            self.assertEqual("Q2", h.eels_quantification_title)
            self.assertTrue(notified_ref[0])
