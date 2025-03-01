import sys
import re
import pickle
import numpy as np
import random
from PyQt5.QtCore import Qt, QPoint, QRect, QSize
from PyQt5.QtGui import QPainter, QColor, QPen, QCursor
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QLabel, QFrame,
    QDialog, QVBoxLayout, QHBoxLayout, QLineEdit, QMenu, QAction,
    QFileDialog, QInputDialog, QComboBox, QTableWidget, QTableWidgetItem,QInputDialog, QMessageBox
)


# Custom Node widget that acts like a button and stores extra metadata.
class NodeButton(QPushButton):
    def __init__(self, node_type, parent):
        super().__init__(node_type, parent)
        self.meta = {}  # dictionary to store additional metadata
        # --- Make background partially transparent so lines show through ---
        self.setStyleSheet("""
            QPushButton {
                background-color: rgba(255, 255, 255, 80); /* semi-transparent */
                border: 1px solid black;
                border-radius: 4px;
            }
        """)
        # Allows the stylesheet's transparent background to work
        self.setAttribute(Qt.WA_StyledBackground, True)
        # ---------------------------------------------------------
        self.setFixedSize(100, 50)
        self.setMouseTracking(True)
        self.drag_start = None

    def mousePressEvent(self, event):
        self.drag_start = event.pos()
        if event.button() == Qt.LeftButton:
            self.parent().parent().on_node_gui_input(event, self)
            event.accept()
        elif event.button() == Qt.RightButton:
            self.parent().parent()._on_layer_clicked(self)
            event.accept()
            return  # Prevent further propagation
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.drag_start is not None and event.buttons() & Qt.LeftButton:
            diff = event.pos() - self.drag_start
            self.move(self.pos() + diff)
            # Update circle positions only for rectangular mode
            if not self.meta.get("is_circular", False):
                for circle in self.meta.get("circle_nodes") or []:
                    offset = circle.meta.get("offset", QPoint(0, 0))
                    circle.move(self.pos() + offset)
            self.parent().update()  # redraw connections while dragging
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self.drag_start = None
        super().mouseReleaseEvent(event)


# A small red circle widget used for node connections.
class CircleLabel(QLabel):
    def __init__(self, parent_node, position_type, parent):
        super().__init__(parent)
        self.parent_node = parent_node
        self.position_type = position_type
        self.meta = {}
        self.setFixedSize(10, 10)
        self.setStyleSheet("background-color: red;")

    def mousePressEvent(self, event):
        # Delegate the event to the main application
        self.parent().parent()._on_circle_clicked(event, self.parent_node, self.position_type)
        event.accept()
        super().mousePressEvent(event)


# A panel used for representing neurons in circular mode.
class Panel(QFrame):
    def __init__(self, parent_node, parent):
        super().__init__(parent)
        self.parent_node = parent_node
        self.meta = {}
        self.setFixedSize(25, 25)
        # --- Make background partially transparent so lines show through ---
        self.setStyleSheet("background-color: rgba(255,153,51,80); border-radius: 10px;")
        self.setAttribute(Qt.WA_StyledBackground, True)
        # ---------------------------------------------------------

    def mousePressEvent(self, event):
        self.window()._on_neuron_gui_input(event, self.parent_node, self)
        if event.button() == Qt.RightButton:
            event.accept()  # Consume the event so it doesn't propagate to the parent.
            return
        super().mousePressEvent(event)


# The canvas widget where nodes are placed and connections drawn.
class CanvasWidget(QWidget):
    def __init__(self, main_node, parent=None):
        super().__init__(parent)
        self.main_node = main_node
        self.setMouseTracking(True)

    def paintEvent(self, event):
        painter = QPainter(self)
        for conn in self.main_node.connections:
            color = self.main_node.get_weight_color(conn["weight"])
            pen = QPen(color, 2)
            painter.setPen(pen)
            from_node = conn["from"]
            to_node = conn["to"]

            # Determine the starting position
            if from_node.meta.get("is_circular", False) and conn.get("from_panel_index", -1) != -1:
                panel = from_node.meta["circle_nodes"][conn["from_panel_index"]]
                from_pos = panel.mapTo(self, QPoint(panel.width() // 2, panel.height() // 2))
            else:
                from_pos = from_node.pos() + QPoint(110, 25)

            # Determine the ending position
            if to_node.meta.get("is_circular", False) and conn.get("to_panel_index", -1) != -1:
                panel = to_node.meta["circle_nodes"][conn["to_panel_index"]]
                to_pos = panel.mapTo(self, QPoint(panel.width() // 2, panel.height() // 2))
            else:
                to_pos = to_node.pos() + QPoint(-10, 25)

            painter.drawLine(from_pos, to_pos)
        painter.end()


# Main application window – it contains the canvas and menus.
class ML_IDE_MainNode(QMainWindow):
    def __init__(self):
        super().__init__()
        self.nodes = []  # list of created nodes
        self.connections = []  # list of connection dictionaries
        self.pending_connection = None  # stores a pending connection dict
        self.dragging_node = None
        self.node_id_counter = 0
        self.mouse_position = QPoint(0, 0)
        # Training/optimization variables:
        self.optimizer_type = "adam"
        self.learning_rate = 0.01
        self.training_data = []
        self.epochs = 1000
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_epsilon = 1e-8
        self.adam_t = 0
        self.np = np  # using numpy for random numbers
        self.initUI()

    def initUI(self):
        self.setWindowTitle("ML IDE")
        self.setGeometry(100, 100, 800, 600)
        # Create the canvas as the central widget.
        self.canvas = CanvasWidget(self, self)
        self.setCentralWidget(self.canvas)
        # Set up the canvas to have a custom context menu.
        self.canvas.setContextMenuPolicy(Qt.CustomContextMenu)
        self.canvas.customContextMenuRequested.connect(self.show_context_menu)
        self.create_menus()
        self.create_drag_and_drop_system()
        self.create_context_menu()
        print("ML IDE initialized")
        self.show()

    # --------------------- Node and Connection functions ---------------------
    def create_node(self, node_type, position):
        node = NodeButton(node_type, self.canvas)
        node.move(position)
        node.meta["id"] = self.node_id_counter
        self.node_id_counter += 1

        # Determine the node role and neuron count based on the string (e.g. "Input: 2 neurons")
        role = "Layer"  # default role
        neuron_count = 1  # default count
        if node_type.startswith("Input"):
            role = "Input"
        elif node_type.startswith("Output"):
            role = "Output"

        # Use regex to extract the number of neurons
        m = re.search(r":\s*(\d+)\s*neurons", node_type)
        if m:
            neuron_count = int(m.group(1))

        # Store the role and neuron count in meta for later use
        node.meta["node_role"] = role
        node.meta["neuron_count"] = neuron_count

        # Set activation function based on role
        if role == "Input":
            node.meta["activation"] = "linear"
        else:
            node.meta["activation"] = "Sigmoid"
        node.meta["bias"] = 0.0
        node.meta["activation_value"] = 0.0

        # Create connection circles based on node role.
        if role == "Input":
            self.create_connection_circle(node, QPoint(105, 20), "right")
        elif role == "Output":
            self.create_connection_circle(node, QPoint(-15, 20), "left")
        else:
            self.create_connection_circle(node, QPoint(-15, 20), "left")
            self.create_connection_circle(node, QPoint(105, 20), "right")
        node.show()

        # Save a copy of the original rectangular circles for later restoration.
        node.meta["rect_circles"] = node.meta["circle_nodes"].copy()

        self.nodes.append(node)
        return node

    def create_connection_circle(self, parent_node, offset, position_type):
        circle = CircleLabel(parent_node, position_type, self.canvas)
        circle.meta["offset"] = offset  # store the offset for later updates
        circle.move(parent_node.pos() + offset)
        circle.show()
        parent_node.meta[position_type + "_circle"] = circle
        if "circle_nodes" not in parent_node.meta:
            parent_node.meta["circle_nodes"] = []
        parent_node.meta["circle_nodes"].append(circle)
        return circle

    def _on_circle_clicked(self, event, parent_node, position_type):
        if event.button() == Qt.LeftButton:
            if position_type == "right":
                self.pending_connection = {"node": parent_node, "panel_index": -1}
            elif position_type == "left" and self.pending_connection:
                if self.pending_connection["node"] != parent_node:
                    self.connect_nodes(self.pending_connection["node"], parent_node,
                                       self.pending_connection["panel_index"], -1)
                self.pending_connection = None
        elif event.button() == Qt.RightButton:
            self.remove_connections_for_circle(parent_node, position_type)
            self.pending_connection = None

    def remove_connections_for_circle(self, node, position_type):
        for i in range(len(self.connections) - 1, -1, -1):
            conn = self.connections[i]
            if position_type == "right" and conn["from"] == node:
                self.connections.pop(i)
            elif position_type == "left" and conn["to"] == node:
                self.connections.pop(i)
        self.canvas.update()
        print("Removed connections for node", node.meta["id"], "(", position_type, "dot)")

    def connect_nodes(self, from_node, to_node, from_panel_index=-1, to_panel_index=-1):
        weight = self.np.random.uniform(-1.0, 1.0)
        conn = {
            "from": from_node,
            "to": to_node,
            "from_panel_index": from_panel_index,
            "to_panel_index": to_panel_index,
            "weight": weight,
            "m": 0.0,
            "v": 0.0
        }
        self.connections.append(conn)
        print("Connected node", from_node.meta["id"], "(panel", from_panel_index,
              ") to node", to_node.meta["id"], "(panel", to_panel_index, ") with weight", weight)
        self.canvas.update()

    def get_weight_color(self, weight):
        factor = abs(weight)
        if factor > 1.0:
            factor = 1.0
        red = int(factor * 255)
        green = int((1.0 - factor) * 255)
        return QColor(red, green, 0)

    def _draw(self):
        self.canvas.update()

    def on_node_gui_input(self, event, node):
        # Called from NodeButton.mousePressEvent
        if event.button() == Qt.LeftButton:
            self.dragging_node = node
        elif event.button() == Qt.RightButton:
            self._on_layer_clicked(node)

    def on_layer_clicked(self, node):
        # An alias if you prefer a function named on_layer_clicked
        self._on_layer_clicked(node)

    def _on_activation_selected(self, activation_dropdown, node):
        selected_activation = activation_dropdown.currentText()
        node.meta["activation"] = selected_activation

        # Store and use the base text without any activation details.
        base_text = node.meta.get("base_text", node.text().split("\n(")[0])
        node.meta["base_text"] = base_text  # Save for future updates
        node.setText(base_text)

        # Remove any existing activation label
        if "activation_label" in node.meta:
            node.meta["activation_label"].deleteLater()

        # Create a new label for the activation text
        activation_label = QLabel(f"({selected_activation})", node)
        text_height = node.fontMetrics().height()
        activation_label.move(10, 10 + text_height)
        activation_label.setStyleSheet("background-color: transparent;")
        activation_label.show()
        node.meta["activation_label"] = activation_label

        print("Activation function selected for", selected_activation)

    def randomize_weights(self, weight_range=None):
        """
        Randomizes the weights for all connections in the network.

        Parameters:
            weight_range (tuple): A tuple (min, max) specifying the range for random weights.
        """

        weight_range_1 = -0.5
        weight_range_2 = 0.5
        for conn in self.connections:
            # Randomize weight using a uniform distribution
            conn["weight"] = np.random.uniform(weight_range_1, weight_range_2)

            # If using Adam optimizer, reset momentum and velocity
            if self.optimizer_type == "adam":
                conn["m"] = 0.0
                conn["v"] = 0.0
        self.canvas.update()

    def feed_forward(self, input_vector):
        # Process input nodes (sorted by id for consistency)
        input_nodes = sorted(
            [node for node in self.nodes if node.meta.get("node_role") == "Input"],
            key=lambda n: n.meta["id"]
        )
        if len(input_vector) != sum(n.meta["neuron_count"] for n in input_nodes):
            raise ValueError("Input vector size mismatch with the number of input neurons!")
        index = 0
        for node in input_nodes:
            count = node.meta["neuron_count"]
            # Store input as a list (even if a single value)
            node.meta["activation_value"] = input_vector[index:index + count]
            index += count

        # Process hidden layer nodes ("Layer")
        hidden_nodes = [node for node in self.nodes if node.meta.get("node_role") == "Layer"]
        for node in hidden_nodes:
            net_input = 0.0
            for conn in self.connections:
                if conn["to"] == node:
                    from_node = conn["from"]
                    weight = conn["weight"]
                    from_activation = from_node.meta.get("activation_value", 0.0)
                    # If the source is multi-neuron, sum its activations
                    if isinstance(from_activation, list):
                        net_input += weight * sum(from_activation)
                    else:
                        net_input += weight * from_activation
            net_input += node.meta.get("bias", 0.0)
            # Use a nonlinear activation (e.g., Sigmoid) for hidden nodes
            activation_func = node.meta.get("activation", "Sigmoid")
            if activation_func == "Sigmoid":
                activated = 1.0 / (1.0 + np.exp(-net_input))
            elif activation_func == "ReLU":
                activated = max(0, net_input)
            elif activation_func == "Tanh":
                activated = np.tanh(net_input)
            else:  # fallback to linear
                activated = net_input
            node.meta["activation_value"] = activated

        # Process output nodes
        output_nodes = [node for node in self.nodes if node.meta.get("node_role") == "Output"]
        for node in output_nodes:
            net_input = 0.0
            for conn in self.connections:
                if conn["to"] == node:
                    from_node = conn["from"]
                    weight = conn["weight"]
                    from_activation = from_node.meta.get("activation_value", 0.0)
                    if isinstance(from_activation, list):
                        net_input += weight * sum(from_activation)
                    else:
                        net_input += weight * from_activation
            net_input += node.meta.get("bias", 0.0)
            # Ensure a nonlinear activation (e.g., Sigmoid) for output nodes
            activation_func = node.meta.get("activation", "Sigmoid")
            if activation_func == "Sigmoid":
                activated = 1.0 / (1.0 + np.exp(-net_input))
            elif activation_func == "ReLU":
                activated = max(0, net_input)
            elif activation_func == "Tanh":
                activated = np.tanh(net_input)
            else:  # fallback to linear
                activated = net_input
            node.meta["activation_value"] = activated

        # Collect and return output node values
        outputs = [node.meta.get("activation_value", 0.0) for node in output_nodes]
        return outputs

    def back_propagation(self, target_vector):
        """
        Adjusts weights and biases using backpropagation based on the given target vector.
        This version:
          - Computes deltas for output and hidden nodes.
          - Updates weights and biases using the computed deltas.
          - Clears the delta values at the end so they don't accumulate across samples.
        """
        # Get output nodes in a consistent order
        output_nodes = [node for node in self.nodes if node.meta.get("node_role") == "Output"]
        if len(target_vector) != len(output_nodes):
            raise ValueError("Mismatch between target vector length and number of output nodes!")

        # Compute delta for output nodes
        for i, node in enumerate(output_nodes):
            output = node.meta.get("activation_value", 0.0)
            target = target_vector[i]
            error = target - output
            activation_func = node.meta.get("activation", "Sigmoid")
            if activation_func == "ReLU":
                derivative = 1 if output > 0 else 0
            elif activation_func == "Sigmoid":
                derivative = output * (1 - output)
            elif activation_func == "Tanh":
                derivative = 1 - output ** 2
            elif activation_func == "Softmax":
                derivative = output * (1 - output)
            else:
                derivative = 1
            node.meta["delta"] = error * derivative

        # Compute delta for hidden nodes (nodes not in Input or Output)
        for node in self.nodes:
            if node.meta.get("node_role") not in ["Input", "Output"]:
                sum_delta = 0.0
                for conn in self.connections:
                    if conn["from"] == node and "delta" in conn["to"].meta:
                        sum_delta += conn["weight"] * conn["to"].meta["delta"]
                activation = node.meta.get("activation_value", 0.0)
                activation_func = node.meta.get("activation", "Sigmoid")
                if activation_func == "ReLU":
                    derivative = 1 if activation > 0 else 0
                elif activation_func == "Sigmoid":
                    derivative = activation * (1 - activation)
                elif activation_func == "Tanh":
                    derivative = 1 - activation ** 2
                elif activation_func == "Softmax":
                    derivative = activation * (1 - activation)
                else:
                    derivative = 1
                node.meta["delta"] = sum_delta * derivative

        # Update weights for all connections
        for conn in self.connections:
            from_node = conn["from"]
            to_node = conn["to"]
            from_activation = from_node.meta.get("activation_value", 0.0)
            if isinstance(from_activation, list):
                from_activation = sum(from_activation)
            delta = to_node.meta.get("delta", 0.0)
            # Gradient for weight is -delta * activation; updating as:
            # weight = weight - learning_rate * (-delta * activation) = weight + learning_rate * activation * delta
            gradient = -delta * from_activation

            if self.optimizer_type == "sgd":
                conn["weight"] -= self.learning_rate * gradient
            elif self.optimizer_type == "adam":
                self.adam_t += 1
                conn["m"] = self.adam_beta1 * conn["m"] + (1 - self.adam_beta1) * gradient
                conn["v"] = self.adam_beta2 * conn["v"] + (1 - self.adam_beta2) * (gradient ** 2)
                m_hat = conn["m"] / (1 - self.adam_beta1 ** self.adam_t)
                v_hat = conn["v"] / (1 - self.adam_beta2 ** self.adam_t)
                conn["weight"] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.adam_epsilon)
            else:
                conn["weight"] -= self.learning_rate * gradient

        # Update biases for each node (only once per node)
        for node in self.nodes:
            if node.meta.get("node_role") in ["Output", "Layer"] and "delta" in node.meta:
                # For our delta definition, bias update is: bias = bias + learning_rate * delta
                node.meta["bias"] += self.learning_rate * node.meta["delta"]

        # Clear delta values so that they don't accumulate between samples
        for node in self.nodes:
            if "delta" in node.meta:
                node.meta.pop("delta")

    def train_network(self):
        """
        Trains the network using stored training data.
        Additional modifications:
          - Shuffles training data each epoch.
          - Uses online (sample-by-sample) updates.
        """
        if not self.training_data:
            print("No training data available!")
            return
        for epoch in range(self.epochs):
            random.shuffle(self.training_data)
            total_loss = 0
            for data in self.training_data:
                input_vector = data["input"]
                target_vector = data["target"]

                # Forward pass
                outputs = self.feed_forward(input_vector)

                # Compute loss (Mean Squared Error)
                loss = sum((t - o) ** 2 for t, o in zip(target_vector, outputs)) / len(target_vector)
                total_loss += loss

                # Backpropagation update
                self.back_propagation(target_vector)

            avg_loss = total_loss / len(self.training_data)
            print(f"Epoch {epoch + 1}/{self.epochs} - Loss: {avg_loss:.6f}")
            self.canvas.update()

    def set_optimizer(self, new_optimizer):
        if new_optimizer == "sgd":
            self.optimizer_type = "sgd"
        elif new_optimizer == "adam":
            self.optimizer_type = "adam"
        else:
            self.optimizer_type = "sgd"
        print("Optimizer set to", self.optimizer_type)

    # ------------- Helper functions for connection cleanup -------------
    def _remove_rect_connections_for_node(self, node):
        for i in range(len(self.connections) - 1, -1, -1):
            conn = self.connections[i]
            if ((conn["from"] == node and conn["from_panel_index"] == -1) or
                    (conn["to"] == node and conn["to_panel_index"] == -1)):
                self.connections.pop(i)

    def _remove_circular_connections_for_node(self, node):
        for i in range(len(self.connections) - 1, -1, -1):
            conn = self.connections[i]
            if ((conn["from"] == node and conn["from_panel_index"] != -1) or
                    (conn["to"] == node and conn["to_panel_index"] != -1)):
                self.connections.pop(i)

    # --------------------- Save and Load functions ---------------------
    def save_project(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Project", "", "Save Files (*.save)")
        if file_path:
            data = {
                "nodes": [],
                "connections": [],
                "training_data": self.training_data
            }
            for node in self.nodes:
                node_data = {
                    "id": node.meta.get("id"),
                    "position": (node.x(), node.y()),
                    "is_circular": node.meta.get("is_circular", False),
                    "name": node.meta.get("original_text", node.text()),
                    "neuron_count": node.meta.get("neuron_count", 0),
                    "node_role": node.meta.get("node_role", ""),
                    "activation": node.meta.get("activation", "default")
                }
                if node_data["is_circular"] and "circle_nodes" in node.meta:
                    panels = node.meta["circle_nodes"]
                    panel_states = []
                    for panel in panels:
                        panel_states.append(panel.meta.get("active", True))
                    node_data["neuron_panels"] = panel_states
                data["nodes"].append(node_data)
            for conn in self.connections:
                conn_data = {
                    "from": conn["from"].meta.get("id"),
                    "to": conn["to"].meta.get("id"),
                    "from_panel_index": conn.get("from_panel_index", -1),
                    "to_panel_index": conn.get("to_panel_index", -1),
                    "weight": conn["weight"],
                    "m": conn.get("m", 0.0),
                    "v": conn.get("v", 0.0)
                }
                data["connections"].append(conn_data)
            with open(file_path, "wb") as f:
                pickle.dump(data, f)
            print("Project saved at", file_path)

    def load_project(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Project", "", "Save Files (*.save)")
        if file_path:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
            self.clear_project()
            node_map = {}
            # Recreate all nodes first.
            for node_data in data["nodes"]:
                new_node = self.create_node(node_data["name"], QPoint(*node_data["position"]))
                new_node.meta["id"] = node_data["id"]
                node_map[new_node.meta["id"]] = new_node

                new_node.meta["neuron_count"] = node_data.get("neuron_count", 0)
                if "node_role" in node_data:
                    new_node.meta["node_role"] = node_data["node_role"]
                else:
                    if node_data["name"].startswith("Input"):
                        new_node.meta["node_role"] = "Input"
                    elif node_data["name"].startswith("Output"):
                        new_node.meta["node_role"] = "Output"
                    else:
                        new_node.meta["node_role"] = "Layer"

                # Only apply activation if the node is a hidden layer ("Layer")
                if "activation" in node_data and (new_node.meta.get("node_role", "") == "Layer" or new_node.meta.get("node_role", "") == "Output"):
                    new_node.meta["activation"] = node_data["activation"]
                    # --- Update the UI to show the saved activation ---
                    activation_label = QLabel(f"({new_node.meta['activation']})", new_node)
                    text_height = new_node.fontMetrics().height()
                    activation_label.move(10, 10 + text_height)
                    activation_label.setStyleSheet("background-color: transparent;")
                    activation_label.show()
                    new_node.meta["activation_label"] = activation_label

                if node_data.get("is_circular", False):
                    self.toggle_node_state(new_node)
                    if "neuron_panels" in node_data and "circle_nodes" in new_node.meta:
                        panels = new_node.meta["circle_nodes"]
                        for i in range(min(len(panels), len(node_data["neuron_panels"]))):
                            panel = panels[i]
                            active_state = node_data["neuron_panels"][i]
                            panel.meta["active"] = active_state
                            if active_state:
                                panel.setStyleSheet("background-color: rgba(255,153,51,80); border-radius: 10px;")
                            else:
                                panel.setStyleSheet("background-color: rgb(128,128,128); border-radius: 10px;")
            for conn_data in data["connections"]:
                from_id = conn_data["from"]
                to_id = conn_data["to"]
                if from_id in node_map and to_id in node_map:
                    conn = {
                        "from": node_map[from_id],
                        "to": node_map[to_id],
                        "from_panel_index": conn_data.get("from_panel_index", -1),
                        "to_panel_index": conn_data.get("to_panel_index", -1),
                        "weight": conn_data["weight"],
                        "m": conn_data.get("m", 0.0),
                        "v": conn_data.get("v", 0.0)
                    }
                    self.connections.append(conn)
            self.training_data = data.get("training_data", [])
            self.canvas.update()
            print("Project loaded from", file_path)

    # --------------------- Menu and Context Menu functions ---------------------
    def create_menus(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")
        new_act = QAction("New Project", self)
        open_act = QAction("Open Project", self)
        save_act = QAction("Save Project", self)
        exit_act = QAction("Exit", self)
        new_act.triggered.connect(lambda: self.on_file_menu_selected(0))
        open_act.triggered.connect(lambda: self.on_file_menu_selected(1))
        save_act.triggered.connect(lambda: self.on_file_menu_selected(2))
        exit_act.triggered.connect(lambda: self.on_file_menu_selected(3))
        file_menu.addAction(new_act)
        file_menu.addAction(open_act)
        file_menu.addAction(save_act)
        file_menu.addAction(exit_act)

        compile_menu = menubar.addMenu("Compile")
        # New "Run Network" action for setting input values and retrieving outputs.
        run_act = QAction("Run Network", self)
        run_act.triggered.connect(self.run_network)
        compile_menu.addAction(run_act)

        rand_act = QAction("Randomize weights", self)
        rand_act.triggered.connect(self.randomize_weights)
        compile_menu.addAction(rand_act)

        train_act = QAction("Train Network", self)
        sgd_act = QAction("Set Optimizer to SGD", self)
        adam_act = QAction("Set Optimizer to Adam", self)
        add_training_act = QAction("Add Training Data", self)


        train_act.triggered.connect(lambda: self.on_compile_menu_selected(0))
        sgd_act.triggered.connect(lambda: self.on_compile_menu_selected(1))
        adam_act.triggered.connect(lambda: self.on_compile_menu_selected(2))
        add_training_act.triggered.connect(lambda: self.on_compile_menu_selected(3))

        compile_menu.addAction(train_act)
        compile_menu.addAction(sgd_act)
        compile_menu.addAction(adam_act)
        compile_menu.addAction(add_training_act)
        print("Menus created")

    def run_network(self):
        input_vector = []
        # Gather input values from all Input nodes.
        for node in self.nodes:
            if node.meta.get("node_role", "") == "Input":
                count = node.meta.get("neuron_count", 1)
                # Ask for comma-separated values for this node.
                text, ok = QInputDialog.getText(
                    self,
                    "Input Node",
                    f"Enter {count} values (comma separated) for Input Node {node.meta.get('id')}:"
                )
                if ok:
                    try:
                        values = [float(x.strip()) for x in text.split(",")]
                        if len(values) != count:
                            QMessageBox.warning(
                                self,
                                "Error",
                                f"Node {node.meta.get('id')} expects {count} values, but got {len(values)}."
                            )
                            return
                        # Save the input for this node and add it to the overall input vector.
                        node.meta["activation_value"] = values
                        input_vector.extend(values)
                    except ValueError:
                        QMessageBox.warning(self, "Error", "Please enter valid numbers.")
                        return

        # Process the collected input vector through the network.
        outputs = self.feed_forward(input_vector)
        # Display the outputs.
        QMessageBox.information(self, "Network Output", f"Output: {outputs}")

    def on_file_menu_selected(self, id):
        if id == 0:
            print("New Project selected")
            self.clear_project()
            print("New project initialized")
        elif id == 1:
            print("Open Project selected")
            self.load_project()
        elif id == 2:
            print("Save Project selected")
            self.save_project()
        elif id == 3:
            print("Exiting application")
            QApplication.quit()

    def on_compile_menu_selected(self, id):
        if id == 0:
            print("Train Network selected")
            self.train_network()
        elif id == 1:
            self.set_optimizer("sgd")
        elif id == 2:
            self.set_optimizer("adam")
        elif id == 3:
            self.add_training_data_table()

    def clear_project(self):
        for node in self.nodes:
            node.setParent(None)
        self.nodes.clear()
        self.connections.clear()
        self.training_data.clear()  # Clear the training data set
        self.canvas.update()
        print("New project initialized")

    def create_drag_and_drop_system(self):
        print("Drag and drop system initialized")

    def create_context_menu(self):
        self.context_menu = QMenu(self)
        self.context_menu.addAction("Add Input", lambda: self.ask_for_layer_neuron_count("Input"))
        self.context_menu.addAction("Add Output", lambda: self.ask_for_layer_neuron_count("Output"))
        self.context_menu.addAction("Add Hidden Layer", lambda: self.ask_for_layer_neuron_count("Layer"))
        print("Context menu created")

    def ask_for_layer_neuron_count(self, layer_type):
        count, ok = QInputDialog.getInt(self, "Layer Neuron Count",
                                        f"Enter the number of neurons for {layer_type} layer:")
        if ok and count > 0:
            self.create_node(f"{layer_type}: {count} neurons", self.mouse_position)

    def show_context_menu(self, pos):
        global_pos = self.canvas.mapToGlobal(pos)
        node_under_cursor = self.get_layer_under_mouse(global_pos)
        if not node_under_cursor:
            self.context_menu.move(global_pos)
            self.context_menu.show()

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            if event.isAccepted():
                return
            self.mouse_position = event.globalPos()
            node_under_cursor = self.get_layer_under_mouse(self.mouse_position)
            if not node_under_cursor:
                self.context_menu.move(self.mouse_position)
                self.context_menu.show()
        super().mousePressEvent(event)

    def get_layer_under_mouse(self, mouse_pos):
        for node in self.nodes:
            global_rect = QRect(node.mapToGlobal(QPoint(0, 0)), node.size())
            if global_rect.contains(mouse_pos):
                return node
            if "left_circle" in node.meta:
                left_dot = node.meta["left_circle"]
                global_rect = QRect(left_dot.mapToGlobal(QPoint(0, 0)), left_dot.size())
                if global_rect.contains(mouse_pos):
                    return node
            if "right_circle" in node.meta:
                right_dot = node.meta["right_circle"]
                global_rect = QRect(right_dot.mapToGlobal(QPoint(0, 0)), right_dot.size())
                if global_rect.contains(mouse_pos):
                    return node
        return None

    def _on_layer_clicked(self, node):
        popup_menu = QMenu(self)
        # Only allow activation selection for hidden layers (role "Layer")
        if node.meta.get("node_role", "") == "Layer" or node.meta.get("node_role", "") == "Output":
            popup_menu.addAction("Select Activation", lambda: self._show_activation_dialog(node))
        popup_menu.addAction("Change Visual Format", lambda: self.toggle_node_state(node))
        if node.meta.get("node_role", "") == "Output":
            popup_menu.addAction("Set Output Goal", lambda: self._show_output_goal_dialog(node))
        popup_menu.exec_(QCursor.pos())

    def _show_activation_dialog(self, node):
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Activation Function")
        layout = QVBoxLayout()
        combo = QComboBox()
        combo.addItems(["ReLU", "Sigmoid", "Tanh", "Softmax"])
        layout.addWidget(combo)
        btn = QPushButton("OK")
        btn.clicked.connect(lambda: [self._on_activation_selected(combo, node), dialog.accept()])
        layout.addWidget(btn)
        dialog.setLayout(layout)
        dialog.exec_()

    def _show_output_goal_dialog(self, node):
        dialog = QDialog(self)
        dialog.setWindowTitle("Set Output Goal")
        layout = QVBoxLayout()
        line = QLineEdit()
        line.setPlaceholderText("Enter desired output goal")
        layout.addWidget(line)
        btn = QPushButton("OK")
        btn.clicked.connect(lambda: [self._on_output_goal_entered(node, line), dialog.accept()])
        layout.addWidget(btn)
        dialog.setLayout(layout)
        dialog.exec_()

    def _on_output_goal_entered(self, node, goal_line):
        goal_value = goal_line.text()
        node.meta["goal"] = goal_value
        activated = node.meta.get("activation_value", 0.0)
        display_text = "Goal: " + str(goal_value) + "\nOutput: " + str(activated)
        if "text_label" in node.meta:
            node.meta["text_label"].setText(display_text)
        else:
            label = QLabel(display_text, node)
            label.move(10, node.height() + 5)
            label.show()
            node.meta["output_goal_label"] = label
        print("Output goal set for node", node.meta.get("id"))

    # -------------- Toggling between Rectangular and Circular --------------
    def toggle_node_state(self, node):
        if node.meta.get("is_circular", False):
            self._switch_node_to_rectangular(node)
            node.meta["is_circular"] = False
        else:
            node.meta["original_text"] = node.text()
            node.meta["original_size"] = node.size()
            self._switch_node_to_circular(node)
            node.meta["is_circular"] = True

    def _switch_node_to_circular(self, node):
        if "rect_circles" in node.meta:
            for c in node.meta["rect_circles"]:
                c.hide()
        rect_connections = []
        for conn in self.connections[:]:
            if (conn["from"] == node and conn["from_panel_index"] == -1) or \
               (conn["to"] == node and conn["to_panel_index"] == -1):
                rect_connections.append(conn)
                self.connections.remove(conn)
        node.meta["circle_nodes"] = []
        m = re.search(r"(?:Layer|Input|Output|Hidden Layer):\s*(\d+)\s*neurons", node.text())
        neuron_count = int(m.group(1)) if m else 1
        node.setText("")
        circle_nodes = []
        for i in range(neuron_count):
            panel = Panel(node, node)
            panel.move(50, 75 + i * 35)
            panel.meta["active"] = True
            panel.show()
            circle_nodes.append(panel)
        node.meta["circle_nodes"] = circle_nodes
        node.setFixedSize(node.width(), 35 * neuron_count + 75)
        label = QLabel(node.meta.get("original_text", ""), node)
        label.move(10, 10)
        label.show()
        node.meta["text_label"] = label

        for conn in rect_connections:
            if conn["from"] == node:
                for i in range(len(circle_nodes)):
                    new_conn = {
                        "from": node,
                        "to": conn["to"],
                        "from_panel_index": i,
                        "to_panel_index": conn.get("to_panel_index", -1),
                        "weight": conn["weight"],
                        "m": conn.get("m", 0.0),
                        "v": conn.get("v", 0.0)
                    }
                    self.connections.append(new_conn)
            if conn["to"] == node:
                for i in range(len(circle_nodes)):
                    new_conn = {
                        "from": conn["from"],
                        "to": node,
                        "from_panel_index": conn.get("from_panel_index", -1),
                        "to_panel_index": i,
                        "weight": conn["weight"],
                        "m": conn.get("m", 0.0),
                        "v": conn.get("v", 0.0)
                    }
                    self.connections.append(new_conn)
        self.canvas.update()

    def _switch_node_to_rectangular(self, node):
        for conn in self.connections:
            if conn["from"] == node and conn["from_panel_index"] != -1:
                conn["from_panel_index"] = -1
            if conn["to"] == node and conn["to_panel_index"] != -1:
                conn["to_panel_index"] = -1

        if "circle_nodes" in node.meta:
            for c in node.meta["circle_nodes"]:
                c.setParent(None)
            node.meta["circle_nodes"] = []
        if "rect_circles" in node.meta:
            for c in node.meta["rect_circles"]:
                c.show()
            node.meta["circle_nodes"] = node.meta["rect_circles"]
            if len(node.meta["rect_circles"]) > 0:
                node.meta["left_circle"] = node.meta["rect_circles"][0]
            if len(node.meta["rect_circles"]) > 1:
                node.meta["right_circle"] = node.meta["rect_circles"][1]
        if "original_text" in node.meta:
            node.setText(node.meta["original_text"])
            node.meta.pop("original_text", None)
        if "original_size" in node.meta:
            node.setFixedSize(node.meta["original_size"])
            node.meta.pop("original_size", None)
        if "text_label" in node.meta:
            node.meta["text_label"].setParent(None)
            node.meta["text_label"] = None
        self.canvas.update()

    def _on_neuron_gui_input(self, event, parent_node, neuron_panel):
        panels = parent_node.meta.get("circle_nodes", [])
        try:
            panel_index = panels.index(neuron_panel)
        except ValueError:
            panel_index = -1
        if event.button() == Qt.LeftButton:
            if not neuron_panel.meta.get("active", True):
                neuron_panel.meta["active"] = True
                neuron_panel.setStyleSheet("background-color: rgba(255,153,51,80); border-radius: 10px;")
                self.canvas.update()
                print("Reactivated neuron panel for node", parent_node.meta.get("id"))
                if neuron_panel.meta.get("saved_connections") is not None:
                    saved_conns = neuron_panel.meta["saved_connections"]
                    for connection in saved_conns:
                        self.connections.append(connection)
                    neuron_panel.meta["saved_connections"] = None
                    self.canvas.update()
                    print("Restored saved connections for node", parent_node.meta.get("id"))
            if self.pending_connection and self.pending_connection["node"] != parent_node:
                self.connect_nodes(self.pending_connection["node"], parent_node,
                                   self.pending_connection["panel_index"], panel_index)
                self.pending_connection = None
            else:
                self.pending_connection = {"node": parent_node, "panel_index": panel_index}
        elif event.button() == Qt.RightButton:
            self.remove_connections_for_neuron(parent_node, neuron_panel)
            self.pending_connection = None

    def remove_connections_for_neuron(self, node, neuron_panel):
        panels = node.meta.get("circle_nodes", [])
        try:
            panel_index = panels.index(neuron_panel)
        except ValueError:
            panel_index = -1
        saved_conns = neuron_panel.meta.get("saved_connections")
        if saved_conns is None:
            saved_conns = []
        for i in range(len(self.connections) - 1, -1, -1):
            c = self.connections[i]
            from_panel_idx = c.get("from_panel_index", -1)
            to_panel_idx = c.get("to_panel_index", -1)
            if c["from"] == node and from_panel_idx == panel_index:
                saved_conns.append(c)
                self.connections.pop(i)
            elif c["to"] == node and to_panel_idx == panel_index:
                saved_conns.append(c)
                self.connections.pop(i)
        neuron_panel.meta["saved_connections"] = saved_conns
        neuron_panel.meta["active"] = False
        neuron_panel.setStyleSheet("background-color: rgb(128,128,128); border-radius: 10px;")
        self.canvas.update()
        print("Deactivated neuron panel for node", node.meta.get("id"))

    # ------------------ Table-based approach to add training data ------------------
    def add_training_data_table(self):
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget, QTableWidgetItem

        input_neuron_count = 0
        output_neuron_count = 0
        for node in self.nodes:
            role = node.meta.get("node_role", "")
            count = node.meta.get("neuron_count", 0)
            if role == "Input":
                input_neuron_count += count
            elif role == "Output":
                output_neuron_count += count

        if input_neuron_count == 0:
            input_neuron_count = 1
        if output_neuron_count == 0:
            output_neuron_count = 1

        total_columns = input_neuron_count + output_neuron_count

        dialog = QDialog(self)
        dialog.setWindowTitle("Enter Training Data Table")
        layout = QVBoxLayout(dialog)

        table = QTableWidget(0, total_columns, dialog)
        input_headers = [f"Input {i + 1}" for i in range(input_neuron_count)]
        output_headers = [f"Output {i + 1}" for i in range(output_neuron_count)]
        table.setHorizontalHeaderLabels(input_headers + output_headers)
        layout.addWidget(table)

        for example in self.training_data:
            row_index = table.rowCount()
            table.insertRow(row_index)
            row_data = example["input"] + example["target"]
            for col, value in enumerate(row_data):
                table.setItem(row_index, col, QTableWidgetItem(str(value)))

        btn_layout = QHBoxLayout()
        add_row_btn = QPushButton("Add Row", dialog)
        ok_btn = QPushButton("OK", dialog)
        btn_layout.addWidget(add_row_btn)
        btn_layout.addWidget(ok_btn)
        layout.addLayout(btn_layout)

        def add_row():
            row_index = table.rowCount()
            table.insertRow(row_index)
            for col in range(total_columns):
                table.setItem(row_index, col, QTableWidgetItem(""))

        add_row_btn.clicked.connect(add_row)

        def on_ok():
            new_training_data = []
            for row in range(table.rowCount()):
                row_data = []
                complete_row = True
                for col in range(total_columns):
                    item = table.item(row, col)
                    if item is None or not item.text().strip():
                        complete_row = False
                        break
                    try:
                        value = float(item.text())
                        row_data.append(value)
                    except ValueError:
                        complete_row = False
                        break
                if complete_row and len(row_data) == total_columns:
                    input_values = row_data[:input_neuron_count]
                    target_values = row_data[input_neuron_count:]
                    new_training_data.append({"input": input_values, "target": target_values})
            self.training_data = new_training_data
            if new_training_data:
                print("Training data updated from table:", new_training_data)
            else:
                print("No complete training data rows were found.")
            dialog.accept()

        ok_btn.clicked.connect(on_ok)
        dialog.exec_()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ML_IDE_MainNode()
    sys.exit(app.exec_())
