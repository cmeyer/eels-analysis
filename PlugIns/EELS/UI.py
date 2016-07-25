# standard libraries
import copy
import functools
import gettext
import operator
import typing
import uuid

# third party libraries
# None

# local libraries
import_ok = False
try:
    from nion.utils import Binding
    from nion.utils import Converter
    from nion.utils import Event
    from nion.swift import Application
    from nion.swift import HistogramPanel
    from nion.swift import Panel
    from nion.swift import Workspace
    from nion.swift.model import Connection
    from nion.swift.model import DataItem
    from nion.swift.model import DocumentModel
    from nion.swift.model import Graphics
    from nion.swift.model import Symbolic
    from EELSAnalysis import PeriodicTable
    import_ok = True
except ImportError:
    pass

_ = gettext.gettext


def processing_extract_signal(document_controller):
    display_specifier = document_controller.selected_display_specifier

    fit_region = DocumentModel.DocumentModel.make_region("fit", "interval", params={"label": _("Fit"), "interval": (0.2, 0.3)})
    signal_region = DocumentModel.DocumentModel.make_region("signal", "interval", params={"label": _("Signal"), "interval": (0.4, 0.5)})
    src = DocumentModel.DocumentModel.make_source(display_specifier.data_item, None, "src", _("Source"), regions=[fit_region, signal_region])
    data_item = document_controller.document_model.make_data_item_with_computation("vstack((extract_signal_from_polynomial_background({src}, signal.interval, (fit.interval, )), {src})", [src], [],
                                                                                   _("Background Subtracted"))
    if data_item:
        new_display_specifier = DataItem.DisplaySpecifier.from_data_item(data_item)
        document_controller.display_data_item(new_display_specifier)
        return data_item
    return None


def processing_subtract_linear_background(document_controller):
    display_specifier = document_controller.selected_display_specifier
    fit_region = DocumentModel.DocumentModel.make_region("fit", "interval", params={"label": _("Fit"), "interval": (0.2, 0.3)})
    src = DocumentModel.DocumentModel.make_source(display_specifier.data_item, None, "src", _("Source"), regions=[fit_region, ])
    data_item = document_controller.document_model.make_data_item_with_computation("vstack((subtract_linear_background({src}, fit.interval, (0, 1)), {src}))", [src], [],
                                                                                   _("Linear Background Subtracted"))
    if data_item:
        new_display_specifier = DataItem.DisplaySpecifier.from_data_item(data_item)
        document_controller.display_data_item(new_display_specifier)
        return data_item
    return None


def processing_subtract_background_signal(document_controller):
    display_specifier = document_controller.selected_display_specifier
    fit_region = DocumentModel.DocumentModel.make_region("fit", "interval", params={"label": _("Fit"), "interval": (0.2, 0.3), "graphic_id": "fit"})
    signal_region = DocumentModel.DocumentModel.make_region("signal", "interval", params={"label": _("Signal"), "interval": (0.4, 0.5), "graphic_id": "signal"})
    src = DocumentModel.DocumentModel.make_source(display_specifier.data_item, None, "src", _("Source"), regions=[fit_region, signal_region])
    data_item = document_controller.document_model.make_data_item_with_computation("s = extract_original_signal({src}, fit.interval, signal.interval)\nbg = subtract_background_signal({src}, fit.interval, signal.interval)\nvstack((s, bg, s - bg))", [src], [],
                                                                                   _("Background Subtracted"))
    if data_item:
        new_display_specifier = DataItem.DisplaySpecifier.from_data_item(data_item)
        document_controller.display_data_item(new_display_specifier)
        return data_item
    return None


def show_color_channels(document_controller):
    display_specifier = document_controller.selected_display_specifier
    display = display_specifier.display
    if display:
        names = (_("Red"), _("Green"), _("Blue"))
        for r in range(1, 4):
            region = Graphics.ChannelGraphic()
            region.label = names[r - 1]
            region.position = r / 4
            region.is_shape_locked = True
            display.add_graphic(region)


def filter_channel(document_controller):
    document_model = document_controller.document_model
    display_specifier = document_controller.selected_display_specifier
    data_item = display_specifier.data_item
    if data_item:
        display = data_item.maybe_data_source.displays[0]
        selected_graphics = display.selected_graphics
        selected_graphic = selected_graphics[0] if len(selected_graphics) == 1 else None
        selected_region = None
        for region in display.graphics:
            if region == selected_graphic:
                selected_region = region
                break
        if selected_region:
            src_data_items = document_model.get_source_data_items(data_item)
            if len(src_data_items) == 1:
                pick_data_item = src_data_items[0]
                src_data_items = document_model.get_source_data_items(pick_data_item)
                if len(src_data_items) == 1:
                    src_data_item = src_data_items[0]
                    fit_region = copy.deepcopy(data_item.maybe_data_source.computation.variables[1])
                    src = DocumentModel.DocumentModel.make_source(src_data_item, None, "src", _("Source"), use_display_data=False)
                    script = "sum(subtract_linear_background(src.data, fit.interval, signal.interval))"
                    new_data_item = document_model.make_data_item_with_computation(script, [src], [], _("Mapped"))
                    computation = new_data_item.maybe_data_source.computation
                    computation.create_object("signal", document_model.get_object_specifier(selected_region), label=_("Signal"))
                    computation.add_variable(fit_region)
                    if new_data_item:
                        new_display_specifier = DataItem.DisplaySpecifier.from_data_item(new_data_item)
                        document_controller.display_data_item(new_display_specifier)
                        return new_data_item
    return None


def filter_element(document_controller, f, s):
    document_model = document_controller.document_model
    display_specifier = document_controller.selected_display_specifier
    data_item = display_specifier.data_item
    pick_region = Graphics.EllipseGraphic()
    pick_region.size = 16 / data_item.maybe_data_source.data_and_calibration.data_shape[-2], 16 / data_item.maybe_data_source.data_and_calibration.data_shape[-1]
    pick_region.label = _("Pick")
    data_item.maybe_data_source.displays[0].add_graphic(pick_region)
    pick = document_model.get_pick_region_new(data_item, pick_region=pick_region)
    # pick = document_model.get_pick_new(data_item)
    if pick:
        pick_display_specifier = DataItem.DisplaySpecifier.from_data_item(pick)
        pick_display_specifier.display.display_type = "line_plot"
        fit_region = Graphics.IntervalGraphic()
        fit_region.label = _("Fit")
        fit_region.graphic_id = "fit"
        fit_region.interval = 0.2, 0.3
        pick_display_specifier.display.add_graphic(fit_region)
        signal_region = Graphics.IntervalGraphic()
        signal_region.label = _("Signal")
        signal_region.graphic_id = "signal"
        signal_region.interval = 0.4, 0.5
        pick_display_specifier.display.add_graphic(signal_region)
        script = "map_background_subtracted_signal(src.data, fit.interval, signal.interval)"
        src2 = DocumentModel.DocumentModel.make_source(data_item, None, "src", _("Source"), use_display_data=False)
        map = document_model.make_data_item_with_computation(script, [src2], [], _("Mapped"))
        if map:
            computation = map.maybe_data_source.computation
            computation.create_object("fit", document_model.get_object_specifier(fit_region), label="Fit")
            computation.create_object("signal", document_model.get_object_specifier(signal_region), label="Signal")
            pick_computation = pick.maybe_data_source.computation
            pick_computation.create_object("fit", document_model.get_object_specifier(fit_region), label="Fit")
            pick_computation.create_object("signal", document_model.get_object_specifier(signal_region), label="Signal")
            pick_computation.expression = "pick = sum(src.data * region_mask(src.data, region)[newaxis, ...], tuple(range(1, len(data_shape(src.data)))))\ns = make_signal_like(extract_original_signal(pick, fit.interval, signal.interval), pick)\nbg = make_signal_like(subtract_background_signal(pick, fit.interval, signal.interval), pick)\nvstack((pick, s - bg, bg))"
            # pick_computation.expression = "pick = pick(src.data, pick_region.position)\ns = make_signal_like(extract_original_signal(pick, fit.interval, signal.interval), pick)\nbg = make_signal_like(subtract_background_signal(pick, fit.interval, signal.interval), pick)\nvstack((pick, s - bg, bg))"
            document_controller.display_data_item(pick_display_specifier)
            document_controller.display_data_item(DataItem.DisplaySpecifier.from_data_item(map))

            src_data_and_metadata = data_item.maybe_data_source.data_and_calibration
            fit_region_start = src_data_and_metadata.dimensional_calibrations[0].convert_from_calibrated_value(f[0]) / src_data_and_metadata.data_shape[0]
            fit_region_end = src_data_and_metadata.dimensional_calibrations[0].convert_from_calibrated_value(f[1]) / src_data_and_metadata.data_shape[0]
            signal_region_start = src_data_and_metadata.dimensional_calibrations[0].convert_from_calibrated_value(s[0]) / src_data_and_metadata.data_shape[0]
            signal_region_end = src_data_and_metadata.dimensional_calibrations[0].convert_from_calibrated_value(s[1]) / src_data_and_metadata.data_shape[0]
            fit_region.interval = fit_region_start, fit_region_end
            signal_region.interval = signal_region_start, signal_region_end

def pick_new_edge(document_controller, model_data_item, elemental_mapping):
    document_model = document_controller.document_model
    pick_region = Graphics.EllipseGraphic()
    pick_region.size = 16 / model_data_item.maybe_data_source.data_and_calibration.data_shape[-2], 16 / model_data_item.maybe_data_source.data_and_calibration.data_shape[-1]
    pick_region.label = "{} {}".format(_("Pick"), str(elemental_mapping.electron_shell))
    model_data_item.maybe_data_source.displays[0].add_graphic(pick_region)
    pick_data_item = document_model.get_pick_region_new(model_data_item, pick_region=pick_region)
    # pick_data_item = document_model.get_pick_new(data_item)
    if pick_data_item:
        pick_data_item.title = "{} of {}".format(pick_region.label, model_data_item.title)
        pick_display_specifier = DataItem.DisplaySpecifier.from_data_item(pick_data_item)
        pick_display_specifier.display.display_type = "line_plot"
        fit_region = Graphics.IntervalGraphic()
        fit_region.label = _("Fit")
        fit_region.graphic_id = "fit"
        fit_region.interval = elemental_mapping.fit_interval
        pick_display_specifier.display.add_graphic(fit_region)
        signal_region = Graphics.IntervalGraphic()
        signal_region.label = _("Signal")
        signal_region.graphic_id = "signal"
        signal_region.interval = elemental_mapping.signal_interval
        pick_display_specifier.display.add_graphic(signal_region)
        # TODO: CHANGES VIA CONNECTIONS DON'T GET WRITTEN TO METADATA
        pick_computation = pick_data_item.maybe_data_source.computation
        pick_computation.create_object("mapping", document_model.get_object_specifier(elemental_mapping), label="Mapping")
        pick_computation.expression = "pick = sum(src.data * region_mask(src.data, region)[newaxis, ...], tuple(range(1, len(data_shape(src.data)))))\ns = make_signal_like(extract_original_signal(pick, mapping.fit_interval, mapping.signal_interval), pick)\nbg = make_signal_like(subtract_background_signal(pick, mapping.fit_interval, mapping.signal_interval), pick)\nvstack((pick, s - bg, bg))"
        pick_data_item.add_connection(Connection.PropertyConnection(elemental_mapping, "fit_interval", fit_region, "interval"))
        pick_data_item.add_connection(Connection.PropertyConnection(elemental_mapping, "signal_interval", signal_region, "interval"))
        document_controller.display_data_item(pick_display_specifier)

def map_new_edge(document_controller, model_data_item, elemental_mapping):
    document_model = document_controller.document_model
    script = "map_background_subtracted_signal(src.data, mapping.fit_interval, mapping.signal_interval)"
    src = DocumentModel.DocumentModel.make_source(model_data_item, None, "src", _("Source"), use_display_data=False)
    map_data_item = document_model.make_data_item_with_computation(script, [src], [], "{} {}".format(_("Map"), str(elemental_mapping.electron_shell)))
    if map_data_item:
        computation = map_data_item.maybe_data_source.computation
        computation.create_object("mapping", document_model.get_object_specifier(elemental_mapping), label="Mapping")
        document_controller.display_data_item(DataItem.DisplaySpecifier.from_data_item(map_data_item))


def build_menus(document_controller):
    document_controller.processing_menu.add_menu_item(_("Subtract Linear Background"), lambda: processing_subtract_linear_background(document_controller))
    document_controller.processing_menu.add_menu_item(_("Subtract Background Signal"), lambda: processing_subtract_background_signal(document_controller))
    document_controller.processing_menu.add_menu_item(_("Extract Signal"), lambda: processing_extract_signal(document_controller))
    document_controller.processing_menu.add_menu_item(_("Show Color Channels"), lambda: show_color_channels(document_controller))
    document_controller.processing_menu.add_menu_item(_("Filter Channel"), lambda: filter_channel(document_controller))
    document_controller.processing_menu.add_menu_item(_("Elemental Map (Si K)"), lambda: filter_element(document_controller, (1700, 1800), (1839, 2039)))
    document_controller.processing_menu.add_menu_item(_("Elemental Map (Ga L)"), lambda: filter_element(document_controller, (1100, 1200), (1220, 1420)))


if import_ok and Application.app is not None:
    Application.app.register_menu_handler(build_menus)  # called on import to make the menu entry for this plugin


class ElementalMapping:
    def __init__(self, electron_shell: PeriodicTable.ElectronShell=None, fit_interval=None, signal_interval=None):
        self.uuid = uuid.uuid4()
        self.__fit_interval = fit_interval
        self.__signal_interval = signal_interval
        self.__electron_shell = electron_shell
        self.property_changed_event = Event.Event()

    def close(self):
        pass

    def read_from_dict(self, properties):
        self.uuid = uuid.UUID(properties["uuid"])
        atomic_number = properties.get("atomic_number")
        shell_number = properties.get("shell_number")
        subshell_index = properties.get("subshell_index")
        self.__electron_shell = PeriodicTable.ElectronShell(atomic_number, shell_number, subshell_index)
        self.__fit_interval = properties.get("fit_interval", (0.4, 0.5))
        self.__signal_interval = properties.get("signal_interval", (0.5, 0.6))

    def write_to_dict(self):
        properties = dict()
        properties["type"] = "elemental-mapping"
        properties["uuid"] = str(self.uuid)
        properties["atomic_number"] = self.__electron_shell.atomic_number
        properties["shell_number"] = self.__electron_shell.shell_number
        properties["subshell_index"] = self.__electron_shell.subshell_index
        properties["fit_interval"] = copy.copy(self.__fit_interval)
        properties["signal_interval"] = copy.copy(self.__signal_interval)
        return properties

    def copy_from(self, other):
        self.electron_shell = copy.deepcopy(other.electron_shell)
        self.fit_interval = other.fit_interval
        self.signal_interval = other.signal_interval

    @property
    def electron_shell(self):
        return self.__electron_shell

    @electron_shell.setter
    def electron_shell(self, value):
        if self.__electron_shell != value:
            self.__electron_shell = value
            self.property_changed_event.fire("electron_shell", value)

    @property
    def fit_interval(self):
        return self.__fit_interval

    @fit_interval.setter
    def fit_interval(self, value):
        if self.__fit_interval != value:
            self.__fit_interval = value
            self.property_changed_event.fire("fit_interval", value)

    @property
    def signal_interval(self):
        return self.__signal_interval

    @signal_interval.setter
    def signal_interval(self, value):
        if self.__signal_interval != value:
            self.__signal_interval = value
            self.property_changed_event.fire("signal_interval", value)


elemental_mapping_computation_variable_type = Symbolic.ComputationVariableType('elemental_mapping', "ElementalMapping", ElementalMapping)

Symbolic.ComputationVariable.register_computation_variable_type(elemental_mapping_computation_variable_type)


def is_model(data_item):
    if data_item is not None:
        buffered_data_source = data_item.maybe_data_source
        if buffered_data_source:
            data_and_metadata = buffered_data_source.data_and_calibration
            if data_and_metadata:
                return data_and_metadata.is_data_3d
    return False


class ElementalMappingController:
    # only supports properties of elemental_mappings; no more complex structure allowed

    def __init__(self, document_model: DocumentModel.DocumentModel):

        self.__elemental_mapping_property_changed_listeners = dict()  # typing.Dict[uuid.UUID, Any]

        def item_inserted(key, value, before_index):
            # when a data item is inserted, construct ElementalMapping objects from the metadata
            # and store the element_mapping list on the data item.
            if key == "data_item":
                data_item = value
                if is_model(data_item):  # TODO: improve handling of metadata in derived items so as to not have to skip this
                    buffered_data_source = data_item.maybe_data_source
                    if buffered_data_source:
                        elemental_mappings = list()
                        metadata = buffered_data_source.metadata
                        elemental_mapping_dicts = metadata.get("elemental_mappings", list())
                        for elemental_mapping_dict in elemental_mapping_dicts:
                            elemental_mapping = ElementalMapping()
                            elemental_mapping.read_from_dict(elemental_mapping_dict)
                            elemental_mappings.append(elemental_mapping)
                            elemental_mapping_computation_variable_type.register_object(elemental_mapping)
                            data_item.persistent_object_context.register(elemental_mapping)  # TODO: check this again
                            self.__elemental_mapping_property_changed_listeners[elemental_mapping.uuid] = elemental_mapping.property_changed_event.listen(lambda k, v: self.__write_metadata(data_item))
                        setattr(data_item, "elemental_mappings", elemental_mappings)

        def item_removed(key, value, index):
            if key == "data_item":
                data_item = value
                if is_model(data_item):  # TODO: improve handling of metadata in derived items so as to not have to skip this
                    for elemental_mapping in getattr(data_item, "elemental_mappings", list()):
                        elemental_mapping.close()
                        elemental_mapping_computation_variable_type.unregister_object(elemental_mapping)
                        self.__elemental_mapping_property_changed_listeners[elemental_mapping.uuid].close()
                        del self.__elemental_mapping_property_changed_listeners[elemental_mapping.uuid]
                    delattr(value, "elemental_mappings")

        self.__item_inserted_listener = document_model.item_inserted_event.listen(item_inserted)
        self.__item_removed_listener = document_model.item_removed_event.listen(item_removed)

        for index, data_item in enumerate(document_model.data_items):
            item_inserted("data_item", data_item, index)

        document_model.rebind_computations()

    def close(self):
        self.__item_inserted_listener.close()
        self.__item_inserted_listener = None
        self.__item_removed_listener.close()
        self.__item_removed_listener = None

    def __write_metadata(self, data_item):
        buffered_data_source = data_item.maybe_data_source
        if buffered_data_source:
            metadata = buffered_data_source.metadata
            elemental_mapping_dicts = list()
            for elemental_mapping in getattr(data_item, "elemental_mappings", list()):
                elemental_mapping_dicts.append(elemental_mapping.write_to_dict())
            metadata["elemental_mappings"] = elemental_mapping_dicts
            buffered_data_source.set_metadata(metadata)

    def get_elemental_mappings(self, data_item):
        return getattr(data_item, "elemental_mappings")

    def add_elemental_mapping(self, data_item, elemental_mapping):
        # add the elemental_mapping to the list on the data item.
        # then update the metadata to reflect the new list.
        elemental_mappings = self.get_elemental_mappings(data_item)
        assert all(em.uuid != elemental_mapping.uuid for em in elemental_mappings)
        elemental_mappings.append(elemental_mapping)
        elemental_mapping_computation_variable_type.register_object(elemental_mapping)
        self.__elemental_mapping_property_changed_listeners[elemental_mapping.uuid] = elemental_mapping.property_changed_event.listen(lambda k, v: self.__write_metadata(data_item))
        data_item.persistent_object_context.register(elemental_mapping)  # TODO: check this again
        self.__write_metadata(data_item)

    def remove_elemental_mapping(self, data_item, elemental_mapping):
        # remove element_mapping with matching uuid.
        # then update the metadata to reflect the new list.
        elemental_mappings = self.get_elemental_mappings(data_item)
        assert any(em.uuid == elemental_mapping.uuid for em in elemental_mappings)
        elemental_mappings.remove(elemental_mapping)
        elemental_mapping_computation_variable_type.unregister_object(elemental_mapping)
        self.__elemental_mapping_property_changed_listeners[elemental_mapping.uuid].close()
        del self.__elemental_mapping_property_changed_listeners[elemental_mapping.uuid]
        self.__write_metadata(data_item)


def change_elemental_mapping(document_model, model_data_item, data_item, elemental_mapping):
    mapping_computation_variable = None
    pick_region_specifier = None
    computation = data_item.maybe_data_source.computation if data_item else None
    if computation:
        for computation_variable in computation.variables:
            if computation_variable.name == "mapping":
                mapping_computation_variable = computation_variable
            if computation_variable.name == "region":
                pick_region_specifier = computation_variable.specifier
    if mapping_computation_variable:
        mapping_computation_variable.specifier = document_model.get_object_specifier(elemental_mapping)
        for connection in copy.copy(data_item.connections):
            if connection.source_property in ("fit_interval", "signal_interval"):
                source_property = connection.source_property
                target_property = connection.target_property
                target = connection._target
                data_item.remove_connection(connection)
                new_connection = Connection.PropertyConnection(elemental_mapping, source_property, target, target_property)
                data_item.add_connection(new_connection)
        if pick_region_specifier:
            pick_region_value = document_model.resolve_object_specifier(pick_region_specifier)
            if pick_region_value:
                pick_region = pick_region_value.value
                pick_region.label = "{} {}".format(_("Pick"), str(elemental_mapping.electron_shell))
                data_item.title = "{} of {}".format(pick_region.label, model_data_item.title)
        else:
                data_item.title = "{} {} of {}".format(_("Map"), str(elemental_mapping.electron_shell), model_data_item.title)
        document_model.rebind_computations()


class ElementalMappingPanel(Panel.Panel):

    def __init__(self, document_controller, panel_id, properties):
        super().__init__(document_controller, panel_id, _("Elemental Mappings"))

        self.__elemental_mapping_panel_controller = ElementalMappingController(document_controller.document_model)

        ui = document_controller.ui

        self.__button_group = None

        column = ui.create_column_widget()

        elemental_mapping_column = ui.create_column_widget()

        add_column = ui.create_column_widget()

        column.add(elemental_mapping_column)

        column.add_spacing(12)

        column.add(add_column)

        column.add_spacing(12)

        column.add_stretch()

        self.widget = column

        model_data_item_ref = [None]  # type: typing.List[DataItem]
        current_data_item_ref = [None]  # type: typing.List[DataItem]

        def data_item_changed(data_item):
            current_data_item_ref[0] = data_item
            model_data_item = None
            current_elemental_mapping = None
            if is_model(data_item):
                model_data_item = data_item
            else:
                computation = data_item.maybe_data_source.computation if data_item else None
                if computation:
                    for computation_variable in computation.variables:
                        if computation_variable.name == "src":
                            src_data_item_value = document_controller.document_model.resolve_object_specifier(computation_variable.specifier)
                            src_data_item = src_data_item_value.data_item if src_data_item_value else None
                            if is_model(src_data_item):
                                model_data_item = src_data_item
                        if computation_variable.name == "mapping":
                            current_elemental_mapping_value = document_controller.document_model.resolve_object_specifier(computation_variable.specifier)
                            current_elemental_mapping = current_elemental_mapping_value.value if current_elemental_mapping_value else None
            model_data_item_ref[0] = model_data_item
            elemental_mapping_column.remove_all()
            add_column.remove_all()
            if self.__button_group:
                self.__button_group.close()
                self.__button_group = None
            if model_data_item:
                self.__button_group = ui.create_button_group()
                for index, elemental_mapping in enumerate(self.__elemental_mapping_panel_controller.get_elemental_mappings(model_data_item)):
                    row = ui.create_row_widget()
                    radio_button = None
                    label = None
                    electron_shell = elemental_mapping.electron_shell
                    text = electron_shell.to_long_str()
                    if not is_model(current_data_item_ref[0]):
                        radio_button = ui.create_radio_button_widget(text)
                        self.__button_group.add_button(radio_button, index)
                        if elemental_mapping == current_elemental_mapping:
                            radio_button.checked = True
                        radio_button.on_clicked = functools.partial(change_elemental_mapping, document_controller.document_model, model_data_item, current_data_item_ref[0], elemental_mapping)
                    else:
                        label = ui.create_label_widget(text)
                    delete_button = ui.create_push_button_widget(_("Delete"))
                    pick_button = ui.create_push_button_widget(_("Pick"))
                    map_button = ui.create_push_button_widget(_("Map"))
                    def pick_pressed(elemental_mapping):
                        if current_data_item_ref[0] == model_data_item:
                            pick_new_edge(document_controller, model_data_item, elemental_mapping)
                    def map_pressed(elemental_mapping):
                        if current_data_item_ref[0] == model_data_item:
                            map_new_edge(document_controller, model_data_item, elemental_mapping)
                    def delete_pressed(elemental_mapping):
                        self.__elemental_mapping_panel_controller.remove_elemental_mapping(model_data_item, elemental_mapping)
                        data_item_changed(current_data_item_ref[0])  # TODO: this should be automatic
                    delete_button.on_clicked = functools.partial(delete_pressed, elemental_mapping)
                    pick_button.on_clicked = functools.partial(pick_pressed, elemental_mapping)
                    map_button.on_clicked = functools.partial(map_pressed, elemental_mapping)
                    row.add_spacing(20)
                    if radio_button:
                        row.add(radio_button)
                        row.add_spacing(4)
                    elif label:
                        row.add(label)
                    if is_model(current_data_item_ref[0]):
                        row.add_spacing(12)
                        row.add(pick_button)
                        row.add_spacing(12)
                        row.add(map_button)
                    row.add_stretch()
                    row.add(delete_button)
                    row.add_spacing(12)
                    elemental_mapping_column.add(row)

                if is_model(current_data_item_ref[0]):

                    def add_edge(atomic_number, edge, f, s):
                        model_data_item = model_data_item_ref[0]
                        if model_data_item:
                            elemental_mapping = ElementalMapping(PeriodicTable.ElectronShell.from_eels_notation(atomic_number, edge), f, s)
                            self.__elemental_mapping_panel_controller.add_elemental_mapping(model_data_item, elemental_mapping)
                            data_item_changed(model_data_item)

                    atomic_number_widget = ui.create_combo_box_widget(items=PeriodicTable.PeriodicTable().get_elements_list(), item_getter=operator.itemgetter(1))

                    edge_widget = ui.create_combo_box_widget(items=PeriodicTable.PeriodicTable().get_edges_list(1), item_getter=operator.itemgetter(1))

                    add_button_widget = ui.create_push_button_widget(_("Add Edge"))

                    atomic_number_row = ui.create_row_widget()
                    atomic_number_row.add_spacing(20)
                    atomic_number_row.add(ui.create_label_widget(_("Element")))
                    atomic_number_row.add_spacing(8)
                    atomic_number_row.add(atomic_number_widget)
                    atomic_number_row.add_spacing(8)
                    atomic_number_row.add_stretch()

                    edge_row = ui.create_row_widget()
                    edge_row.add_spacing(20)
                    edge_row.add(ui.create_label_widget(_("Edge")))
                    edge_row.add_spacing(8)
                    edge_row.add(edge_widget)
                    edge_row.add_spacing(8)
                    edge_row.add_stretch()

                    add_button_row = ui.create_row_widget()
                    add_button_row.add_spacing(20)
                    add_button_row.add(add_button_widget)
                    add_button_row.add_spacing(8)
                    add_button_row.add_stretch()

                    add_si_k_button_widget = ui.create_push_button_widget(_("SI K"))
                    add_ga_l1_button_widget = ui.create_push_button_widget(_("Ge L"))

                    add_row = ui.create_row_widget()

                    add_row.add_spacing(20)
                    add_row.add(add_si_k_button_widget)
                    add_row.add_spacing(12)
                    add_row.add(add_ga_l1_button_widget)
                    add_row.add_spacing(20)
                    add_row.add_stretch()

                    add_column.add(atomic_number_row)
                    add_column.add(edge_row)
                    add_column.add(add_button_row)
                    add_column.add(add_row)

                    add_si_k_button_widget.on_clicked = functools.partial(add_edge, 14, "K", (0.644873550257732, 0.6952118234536082), (0.71484375, 0.8155202963917526))
                    add_ga_l1_button_widget.on_clicked = functools.partial(add_edge, 31, "L1", (0.3428439110824742, 0.3931821842783505), (0.40324983891752575, 0.5039263853092784))

                    def add_edge():
                        model_data_item = model_data_item_ref[0]
                        if model_data_item:
                            electron_shell = edge_widget.current_item[0]
                            binding_energy_eV = PeriodicTable.PeriodicTable().nominal_binding_energy_ev(electron_shell)
                            signal_interval_eV = binding_energy_eV, binding_energy_eV * 1.10
                            fit_interval_eV = binding_energy_eV * 0.93, binding_energy_eV * 0.98
                            buffered_data_source = model_data_item.maybe_data_source
                            if buffered_data_source:
                                data_and_metadata = buffered_data_source.data_and_calibration
                                if data_and_metadata and data_and_metadata.dimensional_calibrations is not None and len(data_and_metadata.dimensional_calibrations) > 0:
                                    calibration = data_and_metadata.dimensional_calibrations[0]
                                    if calibration.units == "eV":
                                        fit_region_start = calibration.convert_from_calibrated_value(fit_interval_eV[0]) / data_and_metadata.data_shape[0]
                                        fit_region_end = calibration.convert_from_calibrated_value(fit_interval_eV[1]) / data_and_metadata.data_shape[0]
                                        signal_region_start = calibration.convert_from_calibrated_value(signal_interval_eV[0]) / data_and_metadata.data_shape[0]
                                        signal_region_end = calibration.convert_from_calibrated_value(signal_interval_eV[1]) / data_and_metadata.data_shape[0]
                                        fit_interval = fit_region_start, fit_region_end
                                        signal_interval = signal_region_start, signal_region_end
                                        elemental_mapping = ElementalMapping(electron_shell, fit_interval, signal_interval)
                                        self.__elemental_mapping_panel_controller.add_elemental_mapping(model_data_item, elemental_mapping)
                                        data_item_changed(model_data_item)

                    add_button_widget.on_clicked = add_edge

                    def atomic_number_changed(item):
                        edge_widget.items = PeriodicTable.PeriodicTable().get_edges_list(item[0])

                    atomic_number_widget.on_current_item_changed = atomic_number_changed

        self.__target_data_item_stream = HistogramPanel.TargetDataItemStream(document_controller)
        self.__listener = self.__target_data_item_stream.value_stream.listen(data_item_changed)
        data_item_changed(self.__target_data_item_stream.value)

    def close(self):
        self.__listener.close()
        self.__listener = None
        self.__target_data_item_stream = None
        self.__elemental_mapping_panel_controller.close()
        self.__elemental_mapping_panel_controller = None
        if self.__button_group:
            self.__button_group.close()
            self.__button_group = None
        # continue up the chain
        super().close()

workspace_manager = Workspace.WorkspaceManager()
workspace_manager.register_panel(ElementalMappingPanel, "elemental-mapping-panel", _("Elemental Mappings"), ["left", "right"], "left")
