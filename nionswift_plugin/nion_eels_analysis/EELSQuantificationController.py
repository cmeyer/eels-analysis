"""EELS Quantification objects.
"""
import functools
import gettext
import typing

from nion.data import Calibration
from nion.eels_analysis import PeriodicTable
from nion.swift.model import DataItem
from nion.swift.model import DataStructure
from nion.swift.model import DisplayItem
from nion.swift.model import DocumentModel
from nion.swift.model import Graphics
from nion.swift.model import Symbolic
from nion.utils import Binding
from nion.utils import ListModel
from nion.utils import Observable


_ = gettext.gettext


class EELSInterval:
    """An interval value object."""

    def __init__(self, start_ev: float=None, end_ev: float=None):
        super().__init__()
        self.__start_ev = start_ev
        self.__end_ev = end_ev

    @staticmethod
    def from_fractional_interval(data_len: int, calibration: Calibration.Calibration, interval: typing.Tuple[float, float]) -> "EELSInterval":
        assert data_len > 0
        start_pixel, end_pixel = interval[0] * data_len, interval[1] * data_len
        return EELSInterval(start_ev=calibration.convert_to_calibrated_value(start_pixel), end_ev=calibration.convert_to_calibrated_value(end_pixel))

    @staticmethod
    def from_d(d: typing.Dict) -> "EELSInterval":
        start_ev = (d or dict()).get("start_ev", None)
        end_ev = (d or dict()).get("end_ev", None)
        return EELSInterval(start_ev, end_ev)

    @property
    def start_ev(self) -> typing.Optional[float]:
        return self.__start_ev

    @start_ev.setter
    def start_ev(self, value: typing.Optional[float]) -> None:
        self.__start_ev = value

    @property
    def end_ev(self) -> typing.Optional[float]:
        return self.__end_ev

    @end_ev.setter
    def end_ev(self, value: typing.Optional[float]) -> None:
        self.__end_ev = value

    @property
    def width_ev(self) -> typing.Optional[float]:
        if self.start_ev is not None and self.end_ev is not None:
            return self.end_ev - self.start_ev
        return None

    def to_fractional_interval(self, data_len: int, calibration: Calibration.Calibration) -> typing.Tuple[float, float]:
        assert data_len > 0
        start_pixel = calibration.convert_from_calibrated_value(self.start_ev)
        end_pixel = calibration.convert_from_calibrated_value(self.end_ev)
        return start_pixel / data_len, end_pixel / data_len

    def _write_to_dict(self) -> typing.Dict:
        d = dict()
        if self.__start_ev is not None:
            d["start_ev"] = self.start_ev
        if self.__end_ev is not None:
            d["end_ev"] = self.end_ev
        return d


class EELSInterfaceToFractionalIntervalConverter:
    def __init__(self, eels_data_len: int, eels_calibration: Calibration.Calibration):
        self.__eels_data_len = eels_data_len
        self.__eels_calibration = eels_calibration

    def convert(self, eels_interval: EELSInterval) -> typing.Tuple[float, float]:
        return eels_interval.to_fractional_interval(self.__eels_data_len, self.__eels_calibration)

    def convert_back(self, interval: typing.Tuple[float, float]) -> EELSInterval:
        return EELSInterval.from_fractional_interval(self.__eels_data_len, self.__eels_calibration, interval)


class EELSEdge(Observable.Observable):
    """An edge is a signal interval, a list of fit intervals, and other edge identifying information."""

    def __init__(self, document_model: DocumentModel.DocumentModel, data_structure: DataStructure.DataStructure):
        super().__init__()
        self.__document_model = document_model
        self.__data_structure = data_structure
        self.__signal_eels_interval = None
        self.__fit_eels_intervals = list()
        self.__electron_shell = None
        self.background_data_item = None
        self.signal_data_item = None
        self.signal_interval_graphic = None
        self.fit_interval_graphics = list()
        self.computation = None
        self.__signal_interval_connection = None
        self.__interval_list_connection = None
        self.__signal_interval_about_to_close_connection = None
        self.__computation_about_to_close_connection = None
        self.__read()

    def close(self):
        if self.__signal_interval_connection:
            self.__signal_interval_connection.close()
            self.__signal_interval_connection = None
        if self.__interval_list_connection:
            self.__interval_list_connection.close()
            self.__interval_list_connection = None
        if self.__signal_interval_about_to_close_connection:
            self.__signal_interval_about_to_close_connection.close()
            self.__signal_interval_about_to_close_connection = None
        if self.__computation_about_to_close_connection:
            self.__computation_about_to_close_connection.close()
            self.__computation_about_to_close_connection = None

    @property
    def data_structure(self) -> DataStructure.DataStructure:
        return self.__data_structure

    @property
    def signal_eels_interval(self) -> typing.Optional[EELSInterval]:
        return self.__signal_eels_interval

    @signal_eels_interval.setter
    def signal_eels_interval(self, value: typing.Optional[EELSInterval]) -> None:
        self.__signal_eels_interval = value
        self.notify_property_changed("signal_eels_interval")

    @property
    def electron_shell(self) -> typing.Optional[PeriodicTable.ElectronShell]:
        return self.__electron_shell

    @electron_shell.setter
    def electron_shell(self, value: typing.Optional[PeriodicTable.ElectronShell]) -> None:
        self.__signal_eels_interval = value
        self.notify_property_changed("electron_shell")

    def insert_fit_eels_interval(self, index: int, interval: EELSInterval) -> None:
        self.__fit_eels_intervals.insert(index, interval)
        self.notify_insert_item("fit_eels_intervals", interval, index)

    def append_fit_eels_interval(self, interval: EELSInterval) -> None:
        self.insert_fit_eels_interval(len(self.__fit_eels_intervals), interval)

    def remove_fit_eels_interval(self, index: int) -> None:
        fit_interval = self.__fit_eels_intervals[index]
        self.__fit_eels_intervals.remove(fit_interval)
        self.notify_remove_item("fit_eels_intervals", fit_interval, index)

    def set_fit_eels_interval(self, index: int, interval: EELSInterval) -> None:
        self.__fit_eels_intervals[index] = interval
        self.notify_item_content_changed("fit_eels_intervals", interval, index)

    @property
    def fit_eels_intervals(self) -> typing.List[EELSInterval]:
        return self.__fit_eels_intervals

    def destroy(self) -> None:
        self.__document_model.remove_data_structure(self.__data_structure)
        self.__data_structure = None
        self.close()

    def __write(self) -> None:
        if self.__signal_eels_interval:
            self.__data_structure.set_property_value("signal_eels_interval", self.__signal_eels_interval._write_to_dict())
        if len(self.__fit_eels_intervals) > 0:
            self.__data_structure.set_property_value("fit_eels_intervals", [fit_eels_interval._write_to_dict() for fit_eels_interval in self.__fit_eels_intervals])
        if self.__electron_shell:
            self.__data_structure.set_property_value("electron_shell", self.__electron_shell._write_to_dict())
        self.__data_structure.set_referenced_object("eels_display_item", self.eels_display_item)
        self.__data_structure.set_referenced_object("eels_data_item", self.eels_data_item)

    def __read(self) -> None:
        signal_eels_interval_d = self.__data_structure.get_property_value("signal_eels_interval", None)
        if signal_eels_interval_d:
            self.__signal_eels_interval = EELSInterval.from_d(signal_eels_interval_d)
        self.__fit_eels_intervals = [EELSInterval.from_d(d) for d in self.__data_structure.get_property_value("fit_eels_intervals", list())]
        electron_shell_d = self.__data_structure.get_property_value("electron_shell", None)
        if electron_shell_d:
            self.__electron_shell = PeriodicTable.ElectronShell.from_d(electron_shell_d)

        self.eels_data_item = self.__data_structure.get_referenced_object("eels_data_item")
        self.eels_display_item = self.__data_structure.get_referenced_object("eels_display_item")

        def notify_remove(cascade_items: typing.List) -> None:
            # ensure nothing is deleted twice
            self.computation = None
            self.signal_interval_graphic = None
            self.fit_interval_graphics = list()
            self.signal_data_item = None
            self.background_data_item = None

        self.__eels_data_item_about_to_be_removed_event_listener = self.eels_data_item.about_to_cascade_delete_event.listen(notify_remove) if self.eels_data_item else None
        self.__eels_display_item_about_to_be_removed_event_listener = self.eels_display_item.about_to_cascade_delete_event.listen(notify_remove) if self.eels_display_item else None

        for computation in self.__document_model.computations:
            if computation.processing_id == "eels.background_subtraction2" and computation.source == self.eels_display_item:
                self.computation = computation
                self.fit_interval_graphics = computation._get_variable("fit_interval_graphics").bound_item.value
                self.signal_interval_graphic = computation._get_variable("signal_interval_graphic").bound_item.value
                self.signal_data_item = computation.get_referenced_object("subtracted")
                self.background_data_item = computation.get_referenced_object("background")
                break

        if self.computation:
            self.show()

    @property
    def is_visible(self) -> bool:
        return self.computation is not None

    def show(self) -> None:
        document_model = self.__document_model
        eels_data_item = self.eels_data_item
        eels_display_item = self.eels_display_item

        # create new data items for signal and background, add them to the document model
        if self.signal_data_item:
            signal_data_item = self.signal_data_item
        else:
            signal_data_item = DataItem.DataItem()
            document_model.append_data_item(signal_data_item, auto_display=False)
            signal_data_item.title = f"{eels_data_item.title} Signal"
        if self.background_data_item:
            background_data_item = self.background_data_item
        else:
            background_data_item = DataItem.DataItem()
            document_model.append_data_item(background_data_item, auto_display=False)
            background_data_item.title = f"{eels_data_item.title} Background"

        # create display data channels and display layers for signal and background
        background_display_data_channel = eels_display_item.get_display_data_channel_for_data_item(background_data_item)
        if not background_display_data_channel:
            eels_display_item.append_display_data_channel(DisplayItem.DisplayDataChannel(background_data_item))
            background_display_data_channel = eels_display_item.get_display_data_channel_for_data_item(background_data_item)
            background_data_item_index = eels_display_item.display_data_channels.index(background_display_data_channel)
            eels_display_item.insert_display_layer(0, data_index=background_data_item_index)
            eels_display_item._set_display_layer_property(0, "label", _("Background"))
            eels_display_item._set_display_layer_property(0, "fill_color", "rgba(255, 0, 0, 0.3)")

        signal_display_data_channel = eels_display_item.get_display_data_channel_for_data_item(signal_data_item)
        if not signal_display_data_channel:
            eels_display_item.append_display_data_channel(DisplayItem.DisplayDataChannel(signal_data_item))
            signal_display_data_channel = eels_display_item.get_display_data_channel_for_data_item(signal_data_item)
            signal_data_item_index = eels_display_item.display_data_channels.index(signal_display_data_channel)
            eels_display_item.insert_display_layer(0, data_index=signal_data_item_index)
            eels_display_item._set_display_layer_property(0, "label", _("Signal"))
            eels_display_item._set_display_layer_property(0, "fill_color", "lime")

        # useful values
        eels_data_len = eels_data_item.data_shape[-1]
        eels_calibration = eels_data_item.dimensional_calibrations[-1]

        # create the signal interval graphic
        if self.signal_interval_graphic:
            signal_interval_graphic = self.signal_interval_graphic
        else:
            signal_interval_graphic = Graphics.IntervalGraphic()
            signal_interval_graphic.interval = self.signal_eels_interval.to_fractional_interval(eels_data_len, eels_calibration)
            eels_display_item.add_graphic(signal_interval_graphic)

        # watch for signal graphic being deleted and treat it like hiding
        if self.__signal_interval_about_to_close_connection:
            self.__signal_interval_about_to_close_connection.close()
            self.__signal_interval_about_to_close_connection = None

        def signal_interval_graphic_removed(cascade_items: typing.List) -> None:
            self.signal_interval_graphic = None
            self.hide()

        self.__signal_interval_about_to_close_connection = signal_interval_graphic.about_to_cascade_delete_event.listen(signal_interval_graphic_removed)

        # bind signal interval graphic to signal interval
        if self.__signal_interval_connection:
            self.__signal_interval_connection.close()
            self.__signal_interval_connection = None
        self.__signal_interval_connection = IntervalConnection(eels_data_item, self, "signal_eels_interval", signal_interval_graphic)

        # create the fit interval graphics
        fit_interval_graphics = self.fit_interval_graphics
        for index, fit_eels_interval in enumerate(self.fit_eels_intervals):
            if len(fit_interval_graphics) <= index:
                fit_interval_graphic = Graphics.IntervalGraphic()
                eels_display_item.add_graphic(fit_interval_graphic)
                fit_interval_graphics.append(fit_interval_graphic)
            else:
                fit_interval_graphic = fit_interval_graphics[index]
            fit_interval_graphic.interval = fit_eels_interval.to_fractional_interval(eels_data_len, eels_calibration)

        # create the computation to compute background and signal
        if self.computation:
            computation = self.computation
        else:
            computation = document_model.create_computation()
            computation.processing_id = "eels.background_subtraction2"
            computation.source = eels_display_item
            computation.create_object("eels_spectrum_data_item", document_model.get_object_specifier(eels_data_item))
            computation.create_objects("fit_interval_graphics", [document_model.get_object_specifier(fit_interval_graphic) for fit_interval_graphic in fit_interval_graphics])
            computation.create_object("signal_interval_graphic", document_model.get_object_specifier(signal_interval_graphic))
            computation.create_result("subtracted", document_model.get_object_specifier(signal_data_item))
            computation.create_result("background", document_model.get_object_specifier(background_data_item))
            document_model.append_computation(computation)

        # bind fit interval graphics to fit intervals
        if self.__interval_list_connection:
            self.__interval_list_connection.close()
            self.__interval_list_connection = None

        self.__interval_list_connection = IntervalListConnection(document_model, eels_display_item, eels_data_item, self, fit_interval_graphics, computation)

        # watch for computation being removed
        if self.__computation_about_to_close_connection:
            self.__computation_about_to_close_connection.close()
            self.__computation_about_to_close_connection = None

        def computation_removed(cascade_items: typing.List) -> None:
            self.computation = None
            # computation will delete the two data items
            self.background_data_item = None
            self.signal_data_item = None
            self.hide()

        self.__computation_about_to_close_connection = computation.about_to_cascade_delete_event.listen(computation_removed)

        # enable the legend display
        eels_display_item.set_display_property("legend_position", "top-right")

        # store values
        self.background_data_item = background_data_item
        self.signal_data_item = signal_data_item
        self.signal_interval_graphic = signal_interval_graphic
        self.fit_interval_graphics = fit_interval_graphics
        self.computation = computation

    def hide(self) -> None:
        if self.__signal_interval_connection:
            self.__signal_interval_connection.close()
            self.__signal_interval_connection = None
        if self.__interval_list_connection:
            self.__interval_list_connection.close()
            self.__interval_list_connection = None
        if self.__signal_interval_about_to_close_connection:
            self.__signal_interval_about_to_close_connection.close()
            self.__signal_interval_about_to_close_connection = None
        if self.__computation_about_to_close_connection:
            self.__computation_about_to_close_connection.close()
            self.__computation_about_to_close_connection = None
        if self.computation:
            self.__document_model.remove_computation(self.computation)
            self.computation = None
        if self.signal_interval_graphic:
            self.eels_display_item.remove_graphic(self.signal_interval_graphic)
            self.signal_interval_graphic = None
        for graphic in self.fit_interval_graphics:
            self.eels_display_item.remove_graphic(graphic)
        self.fit_interval_graphics = list()
        # these items should auto remove with the computation
        if self.background_data_item in self.__document_model.data_items:
            self.__document_model.remove_data_item(self.background_data_item)
        self.background_data_item = None
        if self.signal_data_item in self.__document_model.data_items:
            self.__document_model.remove_data_item(self.signal_data_item)
        self.signal_data_item = None


class Singleton(type):
    def __init__(cls, name, bases, dict):
        super().__init__(name, bases, dict)
        cls.instance = None

    def __call__(cls, *args, **kw):
        if cls.instance is None:
            cls.instance = super().__call__(*args, **kw)
        return cls.instance


class EELSQuantificationManager:
    instances = dict()
    listeners = dict()

    @classmethod
    def get_instance(cls, document_model: DocumentModel.DocumentModel) -> "EELSQuantificationManager":
        if not document_model in cls.instances:
            cls.instances[document_model] = EELSQuantificationManager(document_model)

            def document_about_to_close():
                cls.instances.pop(document_model)
                cls.listeners.pop(document_model)

            cls.listeners[document_model] = document_model.about_to_close_event.listen(document_about_to_close)

        return cls.instances[document_model]

    def __init__(self, document_model: DocumentModel.DocumentModel):
        self.__document_model = document_model

        self.__eels_edges = list()
        self.__eels_quantification_display_list_model = ListModel.FilteredListModel(container=self.__document_model, master_items_key="data_structures", items_key="eels_quantification_display_data_structures")
        self.__eels_quantification_display_list_model.filter = ListModel.EqFilter("structure_type", "nion.eels_quantification_display")

        def eels_quantification_display_list_item_inserted(key: str, value: DataStructure.DataStructure, before_index: int) -> None:
            data_structure = value
            self.__eels_edges.insert(before_index, EELSEdge(document_model, data_structure))

        def eels_quantification_display_list_item_removed(key: str, value: DataStructure.DataStructure, index: int) -> None:
            data_structure = value
            for eels_edge in self.__eels_edges:
                if eels_edge.data_structure == data_structure:
                    eels_edge.hide()
            self.__eels_edges.pop(index)

        self.__eels_quantification_display_list_item_inserted_event_listener = self.__eels_quantification_display_list_model.item_inserted_event.listen(eels_quantification_display_list_item_inserted)
        self.__eels_quantification_display_list_item_removed_event_listener = self.__eels_quantification_display_list_model.item_removed_event.listen(eels_quantification_display_list_item_removed)

        for index, eels_quantification_display in enumerate(self.__eels_quantification_display_list_model.items):
            eels_quantification_display_list_item_inserted("eels_quantification_display_data_structures", eels_quantification_display, index)

    def close(self) -> None:
        self.__eels_quantification_display_list_item_inserted_event_listener.close()
        self.__eels_quantification_display_list_item_inserted_event_listener = None
        self.__eels_quantification_display_list_item_removed_event_listener.close()
        self.__eels_quantification_display_list_item_removed_event_listener = None
        self.__eels_quantification_display_list_model.close()
        self.__eels_quantification_display_list_model = None

    @property
    def eels_edges(self) -> typing.List[EELSEdge]:
        return self.__eels_edges

    def add_eels_edge_from_interval_graphic(self, eels_display_item: DisplayItem.DisplayItem, eels_data_item: DataItem.DataItem, signal_interval_graphic: Graphics.IntervalGraphic) -> typing.Optional[EELSEdge]:
        # get the fractional signal interval from the graphic
        signal_interval = signal_interval_graphic.interval

        # calculate fit intervals ahead and behind the signal
        fit_ahead_interval = signal_interval[0] * 0.8, signal_interval[0] * 0.9
        fit_behind_interval = signal_interval[1] * 1.1, signal_interval[1] * 1.2

        # get length and calibration values from the EELS data item
        eels_data_len = eels_data_item.data_shape[-1]
        eels_data_calibration = eels_data_item.dimensional_calibrations[-1]

        # create the signal and two fit EELS intervals
        signal_eels_interval = EELSInterval.from_fractional_interval(eels_data_len, eels_data_calibration, signal_interval)
        fit_ahead_eels_interval = EELSInterval.from_fractional_interval(eels_data_len, eels_data_calibration, fit_ahead_interval)
        fit_behind_eels_interval = EELSInterval.from_fractional_interval(eels_data_len, eels_data_calibration, fit_behind_interval)

        data_structure = DataStructure.DataStructure(structure_type="nion.eels_quantification_display", source=eels_display_item)
        data_structure.set_referenced_object("eels_display_item", eels_display_item)
        data_structure.set_referenced_object("eels_data_item", eels_data_item)
        data_structure.set_property_value("signal_eels_interval", signal_eels_interval._write_to_dict())
        data_structure.set_property_value("fit_eels_intervals", [fit_ahead_eels_interval._write_to_dict(), fit_behind_eels_interval._write_to_dict()])
        self.__document_model.append_data_structure(data_structure)

        for eels_edge in self.__eels_edges:
            if eels_edge.data_structure == data_structure:
                eels_edge.signal_interval_graphic = signal_interval_graphic
                eels_edge.show()
                return eels_edge

        # return the edge
        return None

    def remove_eels_edge(self, eels_edge: EELSEdge) -> None:
        self.__document_model.remove_data_structure(eels_edge.data_structure)

    def get_eels_edges_model_for_display_item(self, display_item: DisplayItem.DisplayItem = None) -> ListModel.MappedListModel:

        class EELSEdgeListModel(ListModel.MappedListModel):

            def __init__(self, document_model: DocumentModel.DocumentModel, qm, display_item: DisplayItem.DisplayItem):
                self.__eels_edge_data_structures_model = ListModel.FilteredListModel(container=document_model, items_key="data_structures")

                self.set_display_item(display_item)

                def map_data_structure_to_eels_edge(data_structure: DataStructure.DataStructure) -> EELSEdge:
                    for eels_edge in qm.eels_edges:
                        if eels_edge.data_structure == data_structure:
                            return eels_edge
                    assert False

                super().__init__(container=self.__eels_edge_data_structures_model, master_items_key="data_structures", items_key="eels_edges", map_fn=map_data_structure_to_eels_edge)

            def close(self):
                self.__eels_edge_data_structures_model.close()
                self.__eels_edge_data_structures_model = None
                super().close()

            def set_display_item(self, display_item: typing.Optional[DisplayItem.DisplayItem]) -> None:

                def match_data_structure(data_structure: DataStructure.DataStructure) -> bool:
                    return data_structure.structure_type == "nion.eels_quantification_display" and display_item and data_structure.get_referenced_object("eels_display_item") == display_item

                self.__eels_edge_data_structures_model.filter = ListModel.PredicateFilter(match_data_structure)

        return EELSEdgeListModel(self.__document_model, self, display_item)


class IntervalConnection:

    def __init__(self, eels_data_item: DataItem.DataItem, eels_edge: EELSEdge, interval_property_name: str, interval_graphic: Graphics.IntervalGraphic):
        self.__interval_graphic_listener = None
        self.__interval_binding = None

        eels_data_len = eels_data_item.data_shape[-1]
        eels_calibration = eels_data_item.dimensional_calibrations[-1]

        converter = EELSInterfaceToFractionalIntervalConverter(eels_data_len, eels_calibration)
        interval_binding = Binding.PropertyBinding(eels_edge, interval_property_name, converter=converter)

        def update_interval(interval):
            interval_graphic.interval = interval

        interval_binding.target_setter = update_interval

        blocked = False
        def update_eels_interval(property_name):
            nonlocal blocked
            if property_name == "interval" and not blocked:
                blocked = True
                interval_binding.update_source(interval_graphic.interval)
                blocked = False

        self.__interval_graphic_listener = interval_graphic.property_changed_event.listen(update_eels_interval)
        self.__interval_binding = interval_binding

    def close(self):
        if self.__interval_graphic_listener:
            self.__interval_graphic_listener.close()
            self.__interval_graphic_listener = None
        if self.__interval_binding:
            self.__interval_binding.close()
            self.__interval_binding = None


class IntervalListConnection:

    def __init__(self, document_model: DocumentModel.DocumentModel, eels_display_item: DisplayItem.DisplayItem, eels_data_item: DataItem.DataItem, eels_edge: EELSEdge, fit_interval_graphics: typing.List[Graphics.IntervalGraphic], computation: Symbolic.Computation):
        self.__fit_interval_graphic_property_changed_listeners = list()
        self.__fit_interval_graphic_about_to_be_removed_listeners = list()
        self.__fit_interval_graphics = fit_interval_graphics

        eels_data_len = eels_data_item.data_shape[-1]
        eels_calibration = eels_data_item.dimensional_calibrations[-1]

        converter = EELSInterfaceToFractionalIntervalConverter(eels_data_len, eels_calibration)

        blocked = False
        def update_fit_eels_interval(index: int, property_name: str) -> None:
            nonlocal blocked
            if property_name == "interval" and not blocked:
                blocked = True
                eels_edge.set_fit_eels_interval(index, converter.convert_back(self.__fit_interval_graphics[index].interval))
                blocked = False

        remove_blocked = False  # argh.
        def remove_fit_eels_interval(index: int, cascade_items: typing.List) -> None:
            nonlocal remove_blocked

            # this message comes from the library.
            remove_blocked = True

            # remove edge; but block notifications are blocked. ugly.
            eels_edge.remove_fit_eels_interval(index)

            # unbind interval graphic from fit eels interval
            self.__fit_interval_graphic_property_changed_listeners[index].close()
            del self.__fit_interval_graphic_property_changed_listeners[index]
            self.__fit_interval_graphic_about_to_be_removed_listeners[index].close()
            del self.__fit_interval_graphic_about_to_be_removed_listeners[index]

            # keep the fit interval graphics list up to date
            del self.__fit_interval_graphics[index]

            remove_blocked = False

        def fit_eels_interval_inserted(key: str, value, before_index: int) -> None:
            # this message comes from the EELS edge
            if key == "fit_eels_intervals":
                fit_eels_interval = value

                # create interval graphic on the display item
                fit_interval_graphic = Graphics.IntervalGraphic()
                eels_display_item.add_graphic(fit_interval_graphic)
                self.__fit_interval_graphics.insert(before_index, fit_interval_graphic)

                # update the interval graphic value
                fit_interval_graphic.interval = fit_eels_interval.to_fractional_interval(eels_data_len, eels_calibration)

                # bind interval graphic to the fit eels interval
                self.__fit_interval_graphic_property_changed_listeners.insert(before_index, fit_interval_graphic.property_changed_event.listen(functools.partial(update_fit_eels_interval, before_index)))
                self.__fit_interval_graphic_about_to_be_removed_listeners.insert(before_index, fit_interval_graphic.about_to_cascade_delete_event.listen(functools.partial(remove_fit_eels_interval, before_index)))

                # add interval graphic to computation
                computation.insert_item_into_objects("fit_interval_graphics", before_index, document_model.get_object_specifier(fit_interval_graphic))

        def fit_eels_interval_removed(key: str, value, index: int) -> None:
            nonlocal remove_blocked

            # this message comes from the EELS edge
            if key == "fit_eels_intervals" and not remove_blocked:
                # unbind interval graphic from fit eels interval
                self.__fit_interval_graphic_property_changed_listeners[index].close()
                del self.__fit_interval_graphic_property_changed_listeners[index]
                self.__fit_interval_graphic_about_to_be_removed_listeners[index].close()
                del self.__fit_interval_graphic_about_to_be_removed_listeners[index]

                # remove interval graphic on the display item. this will also remove the graphic from the computation.
                eels_display_item.remove_graphic(self.__fit_interval_graphics[index])

                # keep the fit interval graphics list up to date
                del self.__fit_interval_graphics[index]

        def fit_eels_interval_value_changed(key: str, value, index: int) -> None:
            if key == "fit_eels_intervals":
                fit_eels_interval = value

                # update the associated interval graphic
                self.__fit_interval_graphics[index].interval = converter.convert(fit_eels_interval)

        self.__item_inserted_event_listener = eels_edge.item_inserted_event.listen(fit_eels_interval_inserted)
        self.__item_removed_event_listener = eels_edge.item_removed_event.listen(fit_eels_interval_removed)
        self.__item_content_changed_event_listener = eels_edge.item_content_changed_event.listen(fit_eels_interval_value_changed)

        # initial binding for fit interval graphics

        for index, fit_interval_graphic in enumerate(fit_interval_graphics):
            self.__fit_interval_graphic_property_changed_listeners.insert(index, fit_interval_graphic.property_changed_event.listen(functools.partial(update_fit_eels_interval, index)))
            self.__fit_interval_graphic_about_to_be_removed_listeners.insert(index, fit_interval_graphic.about_to_cascade_delete_event.listen(functools.partial(remove_fit_eels_interval, index)))

    def close(self):
        self.__item_inserted_event_listener.close()
        self.__item_inserted_event_listener = None
        self.__item_removed_event_listener.close()
        self.__item_removed_event_listener = None
        self.__item_content_changed_event_listener.close()
        self.__item_content_changed_event_listener = None

        for interval_graphic_listener in self.__fit_interval_graphic_property_changed_listeners:
            interval_graphic_listener.close()
        self.__fit_interval_graphic_property_changed_listeners = None

        for interval_graphic_listener in self.__fit_interval_graphic_about_to_be_removed_listeners:
            interval_graphic_listener.close()
        self.__fit_interval_graphic_about_to_be_removed_listeners = None
