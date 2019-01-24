# standard libraries
import gettext
import typing

# third party libraries
# None

# local libraries
from nion.swift import DocumentController
from nion.ui import Declarative
from nion.ui import Window
from nion.utils import Geometry
from nion.utils import Model
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

        finishes = list()

        self.widget = Declarative.construct(parent_window.ui, None, ui_widget, ui_handler, finishes)

        self.attach_widget(self.widget)

        for finish in finishes:
            finish()
        if ui_handler and hasattr(ui_handler, "init_handler"):
            ui_handler.init_handler()
        if ui_handler and hasattr(ui_handler, "init_window"):
            ui_handler.init_window(request_close)

        self._create_menus()

        self.__ui_handler = ui_handler

    def about_to_close(self, geometry: str, state: str) -> None:
        # do this by overriding about_to_close because on_close is reserved for other purposes.
        ui_handler = self.__ui_handler
        if ui_handler and hasattr(ui_handler, "close"):
            ui_handler.close()
        super().about_to_close(geometry, state)

    def show(self, *, size: Geometry.IntSize=None, position: Geometry.IntPoint=None) -> None:
        super().show(size=size, position=position)
        ui_handler = self.__ui_handler
        if ui_handler and hasattr(ui_handler, "did_show"):
            self.__ui_handler.did_show()

    def close(self) -> None:
        super().close()



class Handler(Observable.Observable):

    def __init__(self, qm: EELSQuantificationController.EELSQuantificationManager):
        """
        Handle quantification being added or removed. the quantification selection should remain the same unless the
        selected quantification is deleted, in which case a default quantification should be chosen.

        Handle the case of the user trying to delete the last quantification. window should close in that case?
        """

        super().__init__()

        self.__eels_quantification_manager = qm

        self.eels_quantification_choices = Model.PropertyModel([])
        self.eels_quantification_index = Model.PropertyModel(0)

        self.__selected_eels_quantification = None

        if len(qm.eels_quantifications) == 0:
            qm.create_eels_quantification()

        def sync_quantification_choices():
            eels_quantification_str = _("EELS Quantification")
            self.eels_quantification_choices.value = [q.title or f"{eels_quantification_str} {i}" for i, q in enumerate(qm.eels_quantifications)]

        def sync_quantification(index: typing.Optional[int]) -> None:
            if index is not None:
                self.__selected_eels_quantification = self.__eels_quantification_manager.eels_quantifications[index]
            else:
                self.__selected_eels_quantification = None
            self.notify_property_changed("eels_quantification_title")
            self.notify_property_changed("eels_quantification_status")

        def eels_quantification_property_changed(property_name: str) -> None:
            sync_quantification_choices()

        def eels_quantifications_item_inserted(key: str, value, before_index: int) -> None:
            if key == "eels_quantification_data_structures":
                eels_quantification = value
                self.__eels_quantification_property_changed_listeners.insert(before_index, eels_quantification.property_changed_event.listen(eels_quantification_property_changed))
                sync_quantification_choices()
                if self.__selected_eels_quantification is not None:
                    self.eels_quantification_index.value = self.__eels_quantification_manager.eels_quantifications.index(self.__selected_eels_quantification)

        def eels_quantifications_item_removed(key: str, value, index: int) -> None:
            if key == "eels_quantification_data_structures":
                self.__eels_quantification_property_changed_listeners[index].close()
                del self.__eels_quantification_property_changed_listeners[index]
                sync_quantification_choices()
                if self.__selected_eels_quantification in self.__eels_quantification_manager.eels_quantifications:
                    self.eels_quantification_index.value = self.__eels_quantification_manager.eels_quantifications.index(self.__selected_eels_quantification)
                elif len(self.__eels_quantification_manager.eels_quantifications) > 0:
                    self.eels_quantification_index.value = 0
                else:
                    self.eels_quantification_index.value = None
            if len(self.__eels_quantification_manager.eels_quantifications) == 0:
                self.__request_close_fn()

        self.__eels_quantifications_item_inserted_event_listener = qm.eels_quantifications_model.item_inserted_event.listen(eels_quantifications_item_inserted)
        self.__eels_quantifications_item_removed_event_listener = qm.eels_quantifications_model.item_removed_event.listen(eels_quantifications_item_removed)
        self.__eels_quantification_property_changed_listeners = list()

        self.eels_quantification_index.on_value_changed = sync_quantification

        for index, eels_quantification in enumerate(qm.eels_quantifications):
            eels_quantifications_item_inserted("eels_quantification_data_structures", eels_quantification, index)

        sync_quantification(self.eels_quantification_index.value)

    def init_window(self, request_close_fn: typing.Callable[[], None]) -> None:
        self.__request_close_fn = request_close_fn

    @property
    def eels_quantification_title(self) -> str:
        return self.eels_quantification.title

    @eels_quantification_title.setter
    def eels_quantification_title(self, value: str) -> None:
        self.eels_quantification.title = value
        self.notify_property_changed("eels_quantification_title")

    @property
    def eels_quantification_status(self) -> str:
        return f"{len(self.eels_quantification.eels_edges)} {len(self.__eels_quantification_manager.get_eels_quantification_displays(self.eels_quantification))}"

    @property
    def eels_quantification(self) -> EELSQuantificationController.EELSQuantification:
        return self.__eels_quantification_manager.eels_quantifications[self.eels_quantification_index.value]

    def delete_eels_quantification(self, widget=None) -> None:
        self.__eels_quantification_manager.destroy_eels_quantification(self.eels_quantification)


def open_eels_quantification_window(document_controller: DocumentController.DocumentController):
    document_model = document_controller.document_model

    qm = EELSQuantificationController.EELSQuantificationManager.get_instance(document_model)

    ui_handler = Handler(qm)

    u = Declarative.DeclarativeUI()

    eels_quantification_choice = u.create_combo_box(items_ref="@binding(eels_quantification_choices.value)", current_index="@binding(eels_quantification_index.value)", width=200)

    delete_button = u.create_push_button(text=_("Delete"), on_clicked="delete_eels_quantification")

    eels_quantification_choice_row = u.create_row(eels_quantification_choice, u.create_stretch(), delete_button, u.create_push_button(text=_("Copy")), u.create_push_button(text=_("New")), spacing=4)

    eels_quantification_title_edit = u.create_line_edit(text="@binding(eels_quantification_title)")

    eels_quantification_edit_row = u.create_row(eels_quantification_title_edit, u.create_stretch(), spacing=4)

    eels_quantification_status_row = u.create_row(u.create_label(text="@binding(eels_quantification_status)"), u.create_stretch(), spacing=4)

    column = u.create_column(eels_quantification_choice_row, eels_quantification_edit_row, eels_quantification_status_row, u.create_stretch(), spacing=4, margin=8)

    window = DeclarativeWindow(document_controller, column, ui_handler)
    window.show()

    print(f"title = {ui_handler.eels_quantification_title} / {ui_handler.eels_quantification} / {ui_handler.eels_quantification.title}")

    document_controller.register_dialog(window)
