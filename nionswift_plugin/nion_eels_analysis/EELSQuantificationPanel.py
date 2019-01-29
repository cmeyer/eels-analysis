# standard libraries
import gettext
import typing

# third party libraries
# None

# local libraries
from nion.swift import DocumentController
from nion.swift.model import DisplayItem
from nion.ui import Declarative
from nion.ui import Window
from nion.utils import Event
from nion.utils import Geometry
from nion.utils import Observable

from . import EELSQuantificationController


_ = gettext.gettext


class DeclarativeWindow(Window.Window):

    def __init__(self, parent_window: Window.Window, ui_widget: Declarative.UIDescription, ui_handler):
        super().__init__(parent_window.ui, app=parent_window.app, parent_window=parent_window, window_style="tool")

        def request_close():
            # this may be called in response to the user clicking a button to close.
            # make sure that the button is not destructed as a side effect of closing
            # the window by queueing the close. and it is not possible to use event loop
            # here because the event loop limitations: not able to run both parent and child
            # event loops simultaneously.
            parent_window.queue_task(self.request_close)

        # make and attach closer for the handler; put handler into container closer
        self.__closer = Declarative.Closer()
        if ui_handler and hasattr(ui_handler, "close"):
            ui_handler._closer = Declarative.Closer()
            self.__closer.push_closeable(ui_handler)

        finishes = list()

        self.widget = Declarative.construct(parent_window.ui, self, ui_widget, ui_handler, finishes)

        self.attach_widget(self.widget)

        for finish in finishes:
            finish()
        if ui_handler and hasattr(ui_handler, "init_handler"):
            ui_handler.init_handler()
        if ui_handler and hasattr(ui_handler, "init_window"):
            ui_handler.init_window(request_close)

        self._create_menus()

        self.__ui_handler = ui_handler

    def show(self, *, size: Geometry.IntSize=None, position: Geometry.IntPoint=None) -> None:
        super().show(size=size, position=position)
        ui_handler = self.__ui_handler
        if ui_handler and hasattr(ui_handler, "did_show"):
            self.__ui_handler.did_show()

    def close(self) -> None:
        self.__closer.close()
        super().close()


class Handler(Observable.Observable):

    def __init__(self, document_controller: DocumentController.DocumentController, qm: EELSQuantificationController.EELSQuantificationManager):
        """
        Handle quantification being added or removed. the quantification selection should remain the same unless the
        selected quantification is deleted, in which case a default quantification should be chosen.

        Handle the case of the user trying to delete the last quantification. window should close in that case?
        """

        super().__init__()

        self.__document_controller = document_controller
        self.__eels_quantification_manager = qm
        self.__eels_edges_model = qm.get_eels_edges_model_for_display_item()

        def display_item_changed(display_item: DisplayItem.DisplayItem) -> None:
            self.__eels_edges_model.set_display_item(display_item)

        self.__focused_display_item_changed_event_listener = document_controller.focused_display_item_changed_event.listen(display_item_changed)

        def inserted(k, v, i):
            if k == "eels_edges":
                self.notify_insert_item("eels_edges", v, i)

        self.__item_inserted_event_listener = self.__eels_edges_model.item_inserted_event.listen(inserted)

        def removed(k, v, i):
            if k == "eels_edges":
                self.notify_remove_item("eels_edges", v, i)

        self.__item_inserted_event_listener = self.__eels_edges_model.item_inserted_event.listen(inserted)
        self.__item_removed_event_listener = self.__eels_edges_model.item_removed_event.listen(removed)

        self.__eels_edges_model.set_display_item(document_controller.focused_display_item)

    def close(self) -> None:
        self.__focused_display_item_changed_event_listener.close()
        self.__focused_display_item_changed_event_listener = None
        self.__item_inserted_event_listener.close()
        self.__item_inserted_event_listener = None
        self.__item_removed_event_listener.close()
        self.__item_removed_event_listener = None

    def init_window(self, request_close_fn: typing.Callable[[], None]) -> None:
        self.__request_close_fn = request_close_fn

    @property
    def eels_edges(self) -> typing.List[EELSQuantificationController.EELSEdge]:
        return list(self.__eels_edges_model.items)

    def create_handler(self, component_id: str, container=None, item=None, **kwargs):

        class EELSEdgeSectionHandler:

            def __init__(self, container, eels_edge: EELSQuantificationController.EELSEdge):
                self.container = container
                self.eels_edge = eels_edge
                self.property_changed_event = Event.Event()

                def property_changed(property_name: str) -> None:
                    if property_name == "signal_eels_interval":
                        self.property_changed_event.fire("signal_start")
                        self.property_changed_event.fire("signal_end")

                def item_inserted(key: str, value, before_index: int) -> None:
                    if key == "fit_eels_intervals":
                        self.property_changed_event.fire("fit1_start")
                        self.property_changed_event.fire("fit1_end")
                        self.property_changed_event.fire("fit2_start")
                        self.property_changed_event.fire("fit2_end")

                def item_removed(key: str, value, index: int) -> None:
                    if key == "fit_eels_intervals":
                        self.property_changed_event.fire("fit1_start")
                        self.property_changed_event.fire("fit1_end")
                        self.property_changed_event.fire("fit2_start")
                        self.property_changed_event.fire("fit2_end")

                def item_content_changed(key: str, value, index: int) -> None:
                    if key == "fit_eels_intervals":
                        self.property_changed_event.fire("fit1_start")
                        self.property_changed_event.fire("fit1_end")
                        self.property_changed_event.fire("fit2_start")
                        self.property_changed_event.fire("fit2_end")

                self.__property_changed_listener = self.eels_edge.property_changed_event.listen(property_changed)
                self.__item_inserted_listener = self.eels_edge.item_inserted_event.listen(item_inserted)
                self.__item_removed_listener = self.eels_edge.item_removed_event.listen(item_removed)
                self.__item_content_changed_listener = self.eels_edge.item_content_changed_event.listen(item_content_changed)

            def close(self) -> None:
                self.__property_changed_listener.close()
                self.__property_changed_listener = None
                self.__item_inserted_listener.close()
                self.__item_inserted_listener = None
                self.__item_removed_listener.close()
                self.__item_removed_listener = None
                self.__item_content_changed_listener.close()
                self.__item_content_changed_listener = None

            def init_handler(self):
                pass

            @property
            def is_displayed(self) -> bool:
                return self.eels_edge.is_visible

            @is_displayed.setter
            def is_displayed(self, value: bool) -> None:
                if value:
                    self.eels_edge.show()
                else:
                    self.eels_edge.hide()

            @property
            def edge_index_label(self) -> str:
                return f"EELS Edge #{self._index + 1}"

            @property
            def _index(self):
                return self.container.eels_edges.index(self.eels_edge)

            @property
            def signal_start(self) -> str:
                return f"{self.eels_edge.signal_eels_interval.start_ev:0.2f}"

            @signal_start.setter
            def signal_start(self, value: str) -> None:
                f_value = None
                try:
                    f_value = float(value)
                except ValueError as e:
                    pass
                if f_value is not None:
                    eels_interval = self.eels_edge.signal_eels_interval
                    eels_interval.start_ev = float(value)
                    self.eels_edge.signal_eels_interval = eels_interval

            @property
            def signal_end(self) -> str:
                return f"{self.eels_edge.signal_eels_interval.end_ev:0.2f}"

            @signal_end.setter
            def signal_end(self, value: str) -> None:
                f_value = None
                try:
                    f_value = float(value)
                except ValueError as e:
                    pass
                if f_value is not None:
                    eels_interval = self.eels_edge.signal_eels_interval
                    eels_interval.end_ev = float(value)
                    self.eels_edge.signal_eels_interval = eels_interval

            @property
            def fit1_start(self) -> str:
                if len(self.eels_edge.fit_eels_intervals) > 0:
                    return f"{self.eels_edge.fit_eels_intervals[0].start_ev:0.2f}"
                return str()

            @property
            def fit1_end(self) -> str:
                if len(self.eels_edge.fit_eels_intervals) > 0:
                    return f"{self.eels_edge.fit_eels_intervals[0].end_ev:0.2f}"
                return str()

            @property
            def fit2_start(self) -> str:
                if len(self.eels_edge.fit_eels_intervals) > 1:
                    return f"{self.eels_edge.fit_eels_intervals[1].start_ev:0.2f}"
                return str()

            @property
            def fit2_end(self) -> str:
                if len(self.eels_edge.fit_eels_intervals) > 1:
                    return f"{self.eels_edge.fit_eels_intervals[1].end_ev:0.2f}"
                return str()

        if component_id == "eels_edge":
            return EELSEdgeSectionHandler(container, item)

        return None

    @property
    def resources(self) -> typing.Dict:
        u = Declarative.DeclarativeUI()

        title_row = u.create_row(u.create_label(text="@binding(edge_index_label)"), u.create_stretch(), spacing=8)

        is_displayed_row = u.create_row(u.create_check_box(text=_("Displayed"), checked="@binding(is_displayed)"), u.create_stretch(), spacing=8)

        signal_row = u.create_row(u.create_label(text=_("Signal")), u.create_line_edit(text="@binding(signal_start)", width=60), u.create_line_edit(text="@binding(signal_end)", width=60), u.create_label(text="eV"), u.create_stretch(), spacing=8)

        fit1_row = u.create_row(u.create_label(text=_("Fit 1")), u.create_line_edit(text="@binding(fit1_start)", width=60), u.create_line_edit(text="@binding(fit1_end)", width=60), u.create_label(text="eV"), u.create_stretch(), spacing=8)

        fit2_row = u.create_row(u.create_label(text=_("Fit 2")), u.create_line_edit(text="@binding(fit2_start)", width=60), u.create_line_edit(text="@binding(fit2_end)", width=60), u.create_label(text="eV"), u.create_stretch(), spacing=8)

        column = u.create_column(u.create_spacing(8), title_row, is_displayed_row, signal_row, fit1_row, fit2_row, spacing=8)

        component = u.define_component(content=column, component_id="eels_edge")

        return {"eels_edge": component}


def open_eels_quantification_window(document_controller: DocumentController.DocumentController):
    document_model = document_controller.document_model

    qm = EELSQuantificationController.EELSQuantificationManager.get_instance(document_model)

    ui_handler = Handler(document_controller, qm)

    u = Declarative.DeclarativeUI()

    edges_column = u.create_column(items="eels_edges", item_component_id="eels_edge")

    column = u.create_column(edges_column, u.create_stretch(), margin=8)

    window = DeclarativeWindow(document_controller, column, ui_handler)
    window.show()

    document_controller.register_dialog(window)
