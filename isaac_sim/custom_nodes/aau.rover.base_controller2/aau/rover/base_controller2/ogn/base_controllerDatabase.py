"""Support for simplified access to data on nodes of type aau.rover.base_controller2.RoverBaseController

Controller for the rover base
"""

import omni.graph.core as og
import omni.graph.core._omni_graph_core as _og
import omni.graph.tools.ogn as ogn
import traceback
import sys
import numpy
class base_controllerDatabase(og.Database):
    """Helper class providing simplified access to data on nodes of type aau.rover.base_controller2.RoverBaseController

    Class Members:
        node: Node being evaluated

    Attribute Value Properties:
        Inputs:
            inputs.ang_vel
            inputs.lin_vel
        Outputs:
            outputs.steer_command
            outputs.velocity_command
    """
    # This is an internal object that provides per-class storage of a per-node data dictionary
    PER_NODE_DATA = {}
    # This is an internal object that describes unchanging attributes in a generic way
    # The values in this list are in no particular order, as a per-attribute tuple
    #     Name, Type, ExtendedTypeIndex, UiName, Description, Metadata,
    #     Is_Required, DefaultValue, Is_Deprecated, DeprecationMsg
    # You should not need to access any of this data directly, use the defined database interfaces
    INTERFACE = og.Database._get_interface([
        ('inputs:ang_vel', 'double', 0, 'Angular Velocity', 'Angular velocity', {ogn.MetadataKeys.DEFAULT: '0'}, True, 0, False, ''),
        ('inputs:lin_vel', 'double', 0, 'Linear Velocity', 'Linear velocity', {ogn.MetadataKeys.DEFAULT: '0'}, True, 0, False, ''),
        ('outputs:steer_command', 'double[]', 0, 'Steer Command', 'Steering angles for the boogie motors', {ogn.MetadataKeys.DEFAULT: '[]'}, True, [], False, ''),
        ('outputs:velocity_command', 'double[]', 0, 'Velocity Command', 'The angular velocity for the wheels', {ogn.MetadataKeys.DEFAULT: '[]'}, True, [], False, ''),
    ])
    class ValuesForInputs(og.DynamicAttributeAccess):
        LOCAL_PROPERTY_NAMES = {"ang_vel", "lin_vel", "_setting_locked", "_batchedReadAttributes", "_batchedReadValues"}
        """Helper class that creates natural hierarchical access to input attributes"""
        def __init__(self, node: og.Node, attributes, dynamic_attributes: og.DynamicAttributeInterface):
            """Initialize simplified access for the attribute data"""
            context = node.get_graph().get_default_graph_context()
            super().__init__(context, node, attributes, dynamic_attributes)
            self._batchedReadAttributes = [self._attributes.ang_vel, self._attributes.lin_vel]
            self._batchedReadValues = [0, 0]

        @property
        def ang_vel(self):
            return self._batchedReadValues[0]

        @ang_vel.setter
        def ang_vel(self, value):
            self._batchedReadValues[0] = value

        @property
        def lin_vel(self):
            return self._batchedReadValues[1]

        @lin_vel.setter
        def lin_vel(self, value):
            self._batchedReadValues[1] = value

        def __getattr__(self, item: str):
            if item in self.LOCAL_PROPERTY_NAMES:
                return object.__getattribute__(self, item)
            else:
                return super().__getattr__(item)

        def __setattr__(self, item: str, new_value):
            if item in self.LOCAL_PROPERTY_NAMES:
                object.__setattr__(self, item, new_value)
            else:
                super().__setattr__(item, new_value)

        def _prefetch(self):
            readAttributes = self._batchedReadAttributes
            newValues = _og._prefetch_input_attributes_data(readAttributes)
            if len(readAttributes) == len(newValues):
                self._batchedReadValues = newValues
    class ValuesForOutputs(og.DynamicAttributeAccess):
        LOCAL_PROPERTY_NAMES = { }
        """Helper class that creates natural hierarchical access to output attributes"""
        def __init__(self, node: og.Node, attributes, dynamic_attributes: og.DynamicAttributeInterface):
            """Initialize simplified access for the attribute data"""
            context = node.get_graph().get_default_graph_context()
            super().__init__(context, node, attributes, dynamic_attributes)
            self.steer_command_size = 0
            self.velocity_command_size = 0
            self._batchedWriteValues = { }

        @property
        def steer_command(self):
            data_view = og.AttributeValueHelper(self._attributes.steer_command)
            return data_view.get(reserved_element_count=self.steer_command_size)

        @steer_command.setter
        def steer_command(self, value):
            data_view = og.AttributeValueHelper(self._attributes.steer_command)
            data_view.set(value)
            self.steer_command_size = data_view.get_array_size()

        @property
        def velocity_command(self):
            data_view = og.AttributeValueHelper(self._attributes.velocity_command)
            return data_view.get(reserved_element_count=self.velocity_command_size)

        @velocity_command.setter
        def velocity_command(self, value):
            data_view = og.AttributeValueHelper(self._attributes.velocity_command)
            data_view.set(value)
            self.velocity_command_size = data_view.get_array_size()

        def _commit(self):
            _og._commit_output_attributes_data(self._batchedWriteValues)
            self._batchedWriteValues = { }
    class ValuesForState(og.DynamicAttributeAccess):
        """Helper class that creates natural hierarchical access to state attributes"""
        def __init__(self, node: og.Node, attributes, dynamic_attributes: og.DynamicAttributeInterface):
            """Initialize simplified access for the attribute data"""
            context = node.get_graph().get_default_graph_context()
            super().__init__(context, node, attributes, dynamic_attributes)
    def __init__(self, node):
        super().__init__(node)
        dynamic_attributes = self.dynamic_attribute_data(node, og.AttributePortType.ATTRIBUTE_PORT_TYPE_INPUT)
        self.inputs = base_controllerDatabase.ValuesForInputs(node, self.attributes.inputs, dynamic_attributes)
        dynamic_attributes = self.dynamic_attribute_data(node, og.AttributePortType.ATTRIBUTE_PORT_TYPE_OUTPUT)
        self.outputs = base_controllerDatabase.ValuesForOutputs(node, self.attributes.outputs, dynamic_attributes)
        dynamic_attributes = self.dynamic_attribute_data(node, og.AttributePortType.ATTRIBUTE_PORT_TYPE_STATE)
        self.state = base_controllerDatabase.ValuesForState(node, self.attributes.state, dynamic_attributes)
    class abi:
        """Class defining the ABI interface for the node type"""
        @staticmethod
        def get_node_type():
            get_node_type_function = getattr(base_controllerDatabase.NODE_TYPE_CLASS, 'get_node_type', None)
            if callable(get_node_type_function):
                return get_node_type_function()
            return 'aau.rover.base_controller2.RoverBaseController'
        @staticmethod
        def compute(context, node):
            try:
                per_node_data = base_controllerDatabase.PER_NODE_DATA[node.node_id()]
                db = per_node_data.get('_db')
                if db is None:
                    db = base_controllerDatabase(node)
                    per_node_data['_db'] = db
            except:
                db = base_controllerDatabase(node)

            try:
                compute_function = getattr(base_controllerDatabase.NODE_TYPE_CLASS, 'compute', None)
                if callable(compute_function) and compute_function.__code__.co_argcount > 1:
                    return compute_function(context, node)

                db.inputs._prefetch()
                db.inputs._setting_locked = True
                with og.in_compute():
                    return base_controllerDatabase.NODE_TYPE_CLASS.compute(db)
            except Exception as error:
                stack_trace = "".join(traceback.format_tb(sys.exc_info()[2].tb_next))
                db.log_error(f'Assertion raised in compute - {error}\n{stack_trace}', add_context=False)
            finally:
                db.inputs._setting_locked = False
                db.outputs._commit()
            return False
        @staticmethod
        def initialize(context, node):
            base_controllerDatabase._initialize_per_node_data(node)
            initialize_function = getattr(base_controllerDatabase.NODE_TYPE_CLASS, 'initialize', None)
            if callable(initialize_function):
                initialize_function(context, node)
        @staticmethod
        def release(node):
            release_function = getattr(base_controllerDatabase.NODE_TYPE_CLASS, 'release', None)
            if callable(release_function):
                release_function(node)
            base_controllerDatabase._release_per_node_data(node)
        @staticmethod
        def update_node_version(context, node, old_version, new_version):
            update_node_version_function = getattr(base_controllerDatabase.NODE_TYPE_CLASS, 'update_node_version', None)
            if callable(update_node_version_function):
                return update_node_version_function(context, node, old_version, new_version)
            return False
        @staticmethod
        def initialize_type(node_type):
            initialize_type_function = getattr(base_controllerDatabase.NODE_TYPE_CLASS, 'initialize_type', None)
            needs_initializing = True
            if callable(initialize_type_function):
                needs_initializing = initialize_type_function(node_type)
            if needs_initializing:
                node_type.set_metadata(ogn.MetadataKeys.EXTENSION, "aau.rover.base_controller2")
                node_type.set_metadata(ogn.MetadataKeys.UI_NAME, "Rover Base Controller")
                node_type.set_metadata(ogn.MetadataKeys.DESCRIPTION, "Controller for the rover base")
                node_type.set_metadata(ogn.MetadataKeys.LANGUAGE, "Python")
                base_controllerDatabase.INTERFACE.add_to_node_type(node_type)
        @staticmethod
        def on_connection_type_resolve(node):
            on_connection_type_resolve_function = getattr(base_controllerDatabase.NODE_TYPE_CLASS, 'on_connection_type_resolve', None)
            if callable(on_connection_type_resolve_function):
                on_connection_type_resolve_function(node)
    NODE_TYPE_CLASS = None
    GENERATOR_VERSION = (1, 17, 0)
    TARGET_VERSION = (2, 64, 7)
    @staticmethod
    def register(node_type_class):
        base_controllerDatabase.NODE_TYPE_CLASS = node_type_class
        og.register_node_type(base_controllerDatabase.abi, 1)
    @staticmethod
    def deregister():
        og.deregister_node_type("aau.rover.base_controller2.RoverBaseController")
