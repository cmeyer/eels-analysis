"""EELS Quantification objects.
"""
import copy
import gettext
import typing

from nion.data import Calibration
from nion.eels_analysis import PeriodicTable
from nion.swift.model import DataItem
from nion.swift.model import DisplayItem
from nion.swift.model import DocumentModel
from nion.swift.model import Graphics
from nion.utils import Observable


_ = gettext.gettext


class EELSInterval(Observable.Observable):
    """An interval."""

    def __init__(self, start_ev: float=None, end_ev: float=None):
        super().__init__()
        self.__start_ev = start_ev
        self.__end_ev = end_ev

    @staticmethod
    def from_fractional_interval(data_len: int, calibration: Calibration.Calibration, interval: typing.Tuple[float, float]) -> "EELSInterval":
        assert data_len > 0
        start_pixel, end_pixel = interval[0] * data_len, interval[1] * data_len
        return EELSInterval(start_ev=calibration.convert_to_calibrated_value(start_pixel), end_ev=calibration.convert_to_calibrated_value(end_pixel))

    @property
    def start_ev(self) -> typing.Optional[float]:
        return self.__start_ev

    @start_ev.setter
    def start_ev(self, value: typing.Optional[float]) -> None:
        self.__start_ev = value
        self.notify_property_changed("start_ev")
        self.notify_property_changed("width_ev")

    @property
    def end_ev(self) -> typing.Optional[float]:
        return self.__end_ev

    @end_ev.setter
    def end_ev(self, value: typing.Optional[float]) -> None:
        self.__end_ev = value
        self.notify_property_changed("end_ev")
        self.notify_property_changed("width_ev")

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


class EELSEdge(Observable.Observable):
    """An edge is a signal interval, a list of fit intervals, and other edge identifying information."""

    def __init__(self, *, signal_eels_interval: EELSInterval=None, fit_eels_intervals: typing.List[EELSInterval]=None, electron_shell: PeriodicTable.ElectronShell=None):
        super().__init__()
        self.__signal_eels_interval = signal_eels_interval
        self.__fit_eels_intervals = fit_eels_intervals or list()
        self.__electron_shell = electron_shell

    @property
    def signal_eels_interval(self) -> typing.Optional[EELSInterval]:
        return self.__signal_eels_interval

    @signal_eels_interval.setter
    def signal_eels_interval(self, value: typing.Optional[EELSInterval]) -> None:
        self.__signal_eels_interval = value
        self.notify_property_changed("signal_eels_interval")

    @property
    def electron_shell(self) -> typing.Optional[PeriodicTable.ElectronShell]:
        return self.__signal_eels_interval

    @electron_shell.setter
    def electron_shell(self, value: typing.Optional[PeriodicTable.ElectronShell]) -> None:
        self.__signal_eels_interval = value
        self.notify_property_changed("electron_shell")

    def insert_fit_eels_interval(self, index: int, interval: EELSInterval) -> None:
        self.__fit_eels_intervals.insert(index, interval)
        self.notify_insert_item("fit_intervals", interval, index)

    def append_fit_eels_interval(self, interval: EELSInterval) -> None:
        self.insert_fit_eels_interval(len(self.__fit_eels_intervals), interval)

    def remove_fit_eels_interval(self, index: int) -> None:
        fit_interval = self.__fit_eels_intervals[index]
        self.__fit_eels_intervals.remove(fit_interval)
        self.notify_remove_item("fit_intervals", fit_interval, index)

    @property
    def fit_eels_intervals(self) -> typing.List[EELSInterval]:
        return self.__fit_eels_intervals


class EELSQuantification(Observable.Observable):
    """Quantification settings include a list of edges."""

    def __init__(self, *, eels_edges: typing.List[EELSEdge]=None):
        super().__init__()
        self.__eels_edges = eels_edges or list()

    def insert_edge(self, index: int, eels_edge: EELSEdge) -> None:
        self.__eels_edges.insert(index, eels_edge)
        self.notify_insert_item("eels_edges", eels_edge, index)

    def append_edge(self, eels_edge: EELSEdge) -> None:
        self.insert_edge(len(self.__eels_edges), eels_edge)

    def remove_edge(self, index: int) -> None:
        eels_edge = self.__eels_edges[index]
        self.__eels_edges.remove(eels_edge)
        self.notify_remove_item("eels_edges", eels_edge, index)

    @property
    def eels_edges(self) -> typing.List[EELSEdge]:
        return self.__eels_edges


class EELSEdgeDisplay(Observable.Observable):
    """Display settings for an EELS edge."""

    def __init__(self, eels_edge: EELSEdge, *, is_visible: bool = True):
        super().__init__()
        self.__eels_edge = eels_edge
        self.__is_visible = is_visible

    @property
    def eels_edge(self) -> EELSEdge:
        return self.__eels_edge

    @property
    def is_visible(self) -> bool:
        return self.__is_visible

    @is_visible.setter
    def is_visible(self, value: bool) -> None:
        self.__is_visible = value


class EELSQuantificationDisplay(Observable.Observable):
    """Display settings for an EELS quantification."""

    def __init__(self, eels_quantification: EELSQuantification):
        super().__init__()
        self.__eels_quantification = eels_quantification
        self.__eels_edge_displays = list()

        for eels_edge in self.__eels_quantification.eels_edges:
            self.__eels_edge_displays.append(EELSEdgeDisplay(eels_edge))

        def eels_edge_inserted(key, value, before_index):
            if key == "eels_edges":
                eels_edge_display = EELSEdgeDisplay(value)
                self.__eels_edge_displays.insert(before_index, eels_edge_display)
                self.notify_insert_item("eels_edge_displays", eels_edge_display, before_index)

        def eels_edge_removed(key, value, index):
            if key == "eels_edges":
                eels_edge_display = self.__eels_edge_displays[index]
                self.__eels_edge_displays.remove(eels_edge_display)
                self.notify_remove_item("eels_edge_displays", eels_edge_display, index)

        self.__eels_quantification_item_inserted_event_listener = self.__eels_quantification.item_inserted_event.listen(eels_edge_inserted)
        self.__eels_quantification_item_removed_event_listener = self.__eels_quantification.item_removed_event.listen(eels_edge_removed)

    def close(self):
        self.__eels_quantification_item_inserted_event_listener.close()
        self.__eels_quantification_item_inserted_event_listener = None
        self.__eels_quantification_item_removed_event_listener.close()
        self.__eels_quantification_item_removed_event_listener = None

    @property
    def eels_quantification(self) -> EELSQuantification:
        return self.__eels_quantification

    @property
    def eels_edge_displays(self) -> typing.List[EELSEdgeDisplay]:
        return self.__eels_edge_displays


class EELSEdgeDisplayView:

    def __init__(self, eels_edge: EELSEdge):
        self.__eels_edge = eels_edge
        self.background_data_item = None
        self.signal_data_item = None
        self.signal_interval_graphic = None
        self.fit_interval_graphics = list()
        self.computation = None

    def show(self, document_model: DocumentModel.DocumentModel, eels_display_item: DisplayItem.DisplayItem, eels_data_item: DataItem.DataItem) -> None:

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
            signal_interval_graphic.interval = self.__eels_edge.signal_eels_interval.to_fractional_interval(eels_data_len, eels_calibration)
            eels_display_item.add_graphic(signal_interval_graphic)

        # create the fit interval graphics
        fit_interval_graphics = self.fit_interval_graphics
        for index, fit_eels_interval in enumerate(self.__eels_edge.fit_eels_intervals):
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

        # enable the legend display
        eels_display_item.set_display_property("legend_position", "top-right")

        # store values
        self.background_data_item = background_data_item
        self.signal_data_item = signal_data_item
        self.signal_interval_graphic = signal_interval_graphic
        self.fit_interval_graphics = fit_interval_graphics
        self.computation = computation

    def hide(self, document_model: DocumentModel.DocumentModel, eels_display_item: DisplayItem.DisplayItem) -> None:
        document_model.remove_computation(self.computation)
        eels_display_item.remove_graphic(self.signal_interval_graphic)
        for graphic in self.fit_interval_graphics:
            eels_display_item.remove_graphic(graphic)
        # these items should auto remove with the computation
        if self.background_data_item in document_model.data_items:
            document_model.remove_data_item(self.background_data_item)
        if self.signal_data_item in document_model.data_items:
            document_model.remove_data_item(self.signal_data_item)


class EELSQuantificationController:
    """Controller between a line plot display item and an EELS quantification display.

    Handles the following situations:
        - initial attachment
        - reading and writing
        - enabling/disabling line plot layers
    """

    def __init__(self, document_model: DocumentModel.DocumentModel, eels_display_item: DisplayItem.DisplayItem, eels_data_item: DataItem.DataItem, eels_quantification_display: EELSQuantificationDisplay):
        self.__document_model = document_model
        self.__eels_display_item = eels_display_item
        self.__eels_data_item = eels_data_item
        self.__eels_quantification_display = eels_quantification_display
        self.__eels_quantification = eels_quantification_display.eels_quantification
        self.__eels_edge_views_map = dict()

        # watch for EELS edge displays added or removed and configure appropriately

        def eels_edge_display_inserted(key, value, before_index):
            if key == "eels_edge_displays":
                eels_edge_display = value
                eels_edge = eels_edge_display.eels_edge
                eels_edge_display_view = self.__eels_edge_views_map.get(eels_edge)
                if not eels_edge_display_view:
                    eels_edge_display_view = EELSEdgeDisplayView(eels_edge)
                    self.__eels_edge_views_map[eels_edge] = eels_edge_display_view
                eels_edge_display_view.show(document_model, eels_display_item, eels_data_item)

        def eels_edge_display_removed(key, value, index):
            if key == "eels_edge_displays":
                eels_edge_display = value
                eels_edge = eels_edge_display.eels_edge
                eels_edge_display_view = self.__eels_edge_views_map.get(eels_edge)
                if eels_edge_display_view:
                    eels_edge_display_view.hide(document_model, eels_display_item)

        self.__eels_quantification_display_item_inserted_event_listener = self.__eels_quantification_display.item_inserted_event.listen(eels_edge_display_inserted)
        self.__eels_quantification_display_item_removed_event_listener = self.__eels_quantification_display.item_removed_event.listen(eels_edge_display_removed)

    def close(self):
        self.__eels_quantification_display_item_inserted_event_listener.close()
        self.__eels_quantification_display_item_inserted_event_listener = None
        self.__eels_quantification_display_item_removed_event_listener.close()
        self.__eels_quantification_display_item_removed_event_listener = None

    def add_eels_edge(self, eels_edge: EELSEdge) -> None:
        self.__eels_quantification.append_edge(eels_edge)

    def remove_eels_edge(self, eels_edge: EELSEdge) -> None:
        self.__eels_quantification.remove_edge(self.__eels_quantification.eels_edges.index(eels_edge))

    def add_eels_edge_from_interval_graphic(self, signal_interval_graphic: Graphics.IntervalGraphic) -> EELSEdge:
        # get the fractional signal interval from the graphic
        signal_interval = signal_interval_graphic.interval

        # calculate fit intervals ahead and behind the signal
        fit_ahead_interval = signal_interval[0] * 0.8, signal_interval[0] * 0.9
        fit_behind_interval = signal_interval[1] * 1.1, signal_interval[1] * 1.2

        # get length and calibration values from the EELS data item
        eels_data_len = self.__eels_data_item.data_shape[-1]
        eels_data_calibration = self.__eels_data_item.dimensional_calibrations[-1]

        # create the signal and two fit EELS intervals
        signal_eels_interval = EELSInterval.from_fractional_interval(eels_data_len, eels_data_calibration, signal_interval)
        fit_ahead_eels_interval = EELSInterval.from_fractional_interval(eels_data_len, eels_data_calibration, fit_ahead_interval)
        fit_behind_eels_interval = EELSInterval.from_fractional_interval(eels_data_len, eels_data_calibration, fit_behind_interval)

        # create the EELS edge object
        eels_edge = EELSEdge(signal_eels_interval=signal_eels_interval, fit_eels_intervals=[fit_ahead_eels_interval, fit_behind_eels_interval])

        # add the EELS edge display view
        eels_edge_display_view = EELSEdgeDisplayView(eels_edge)
        eels_edge_display_view.signal_interval_graphic = signal_interval_graphic
        self.__eels_edge_views_map[eels_edge] = eels_edge_display_view

        # add the EELS edge object to the quantification object
        self.add_eels_edge(eels_edge)

        # return the edge
        return eels_edge
