extends Node2D

class_name ML_IDE_MainNode

# Main node for Machine Learning IDE
var nodes = []  # Store all created nodes
var connections = []  # Store connections between nodes
var dragging_node = null  # Track currently dragged node
var context_menu = null
var pending_connection = null  # Stores the node currently trying to connect

func _ready():
	print("ML IDE initialized")
	create_menus()
	create_drag_and_drop_system()
	create_context_menu()

func create_node(node_type: String, position: Vector2):
	var node = Button.new()
	node.text = node_type
	node.set_position(position)
	node.set_size(Vector2(100, 50))
	node.connect("gui_input", Callable(self, "_on_node_gui_input").bind(node))

	# Create left and right connection circles with corrected positions
	var left_circle = create_connection_circle(node, Vector2(-15, 20), "left")
	var right_circle = create_connection_circle(node, Vector2(140, 20), "right")

	# Add node to scene
	add_child(node)
	nodes.append(node)
	
	return node

func create_connection_circle(parent_node, offset: Vector2, position_type: String):
	var circle = ColorRect.new()
	circle.color = Color(1, 0, 0)  # Red
	circle.set_size(Vector2(10, 10))
	circle.set_position(offset)

	# Ensure interaction
	circle.connect("gui_input", Callable(self, "_on_circle_clicked").bind(parent_node, position_type))

	parent_node.add_child(circle)
	return circle

func _on_circle_clicked(event, parent_node, position_type):
	if event is InputEventMouseButton and event.button_index == MOUSE_BUTTON_LEFT and event.pressed:
		if position_type == "right":  # Start connection
			pending_connection = parent_node
		elif position_type == "left" and pending_connection:  # Finish connection
			if pending_connection != parent_node:
				connect_nodes(pending_connection, parent_node)
			pending_connection = null  # Reset pending connection

func connect_nodes(from_node: Node, to_node: Node):
	connections.append({"from": from_node, "to": to_node})
	print("Connected ", from_node.name, " to ", to_node.name)
	queue_redraw()  # Redraw connections

func _draw():
	for connection in connections:
		var from_pos = connection["from"].global_position + Vector2(140, 25)  # Right side
		var to_pos = connection["to"].global_position + Vector2(-5, 25)  # Left side
		draw_line(from_pos, to_pos, Color(0, 1, 0), 2)  # Green connection line

func _on_node_gui_input(event, node):
	if event is InputEventMouseButton:
		if event.button_index == MOUSE_BUTTON_LEFT and event.pressed:
			dragging_node = node
		elif event.button_index == MOUSE_BUTTON_RIGHT and event.pressed:
			_on_layer_clicked(node)
	elif event is InputEventMouseMotion and dragging_node:
		dragging_node.set_position(dragging_node.position + event.relative)
		queue_redraw()  # Update connections while dragging
	if event is InputEventMouseButton and not event.pressed:
		dragging_node = null

func _on_layer_clicked(node):
	var activation_dialog = AcceptDialog.new()
	activation_dialog.dialog_text = "Select Activation Function:"
	activation_dialog.min_size = Vector2(200,150)
	var activation_dropdown = OptionButton.new()
	activation_dropdown.add_item("ReLU")
	activation_dropdown.add_item("Sigmoid")
	activation_dropdown.add_item("Tanh")
	activation_dropdown.add_item("Softmax")
	activation_dialog.add_child(activation_dropdown)
	activation_dialog.connect("confirmed", Callable(self, "_on_activation_selected").bind(activation_dropdown, node))
	add_child(activation_dialog)
	activation_dialog.popup_centered()

func _on_activation_selected(activation_dropdown: OptionButton, node):
	var selected_activation = activation_dropdown.get_item_text(activation_dropdown.selected)
	if "\n(" in node.text:
		node.text = node.text.split("\n(")[0]
	node.text = node.text + "\n(" + selected_activation + ")"
	print("Activation function selected for ", node.text, ": ", selected_activation)

# Save and Load
func save_project():
	var file_dialog = FileDialog.new()
	file_dialog.file_mode = FileDialog.FILE_MODE_SAVE_FILE
	file_dialog.access = FileDialog.ACCESS_USERDATA
	file_dialog.filters = ["*.save ; Save Files"]
	file_dialog.connect("file_selected", Callable(self, "_on_save_file_selected"))
	add_child(file_dialog)
	file_dialog.set_size(Vector2(800, 600))
	file_dialog.popup_centered()
	
func _on_save_file_selected(file_path: String):
	var file = FileAccess.open(file_path, FileAccess.WRITE)

	var saved_nodes = []
	for node in nodes:
		saved_nodes.append({
			"id": node.get_meta("id"),
			"name": node.text,
			"position": node.position
		})

	var saved_connections = []
	for connection in connections:
		saved_connections.append({
			"from": connection["from"].get_meta("id"),
			"to": connection["to"].get_meta("id")
		})

	file.store_var({"nodes": saved_nodes, "connections": saved_connections})
	file.close()
	print("Project saved at ", file_path)

func load_project():
	var file_dialog = FileDialog.new()
	file_dialog.file_mode = FileDialog.FILE_MODE_OPEN_FILE
	file_dialog.access = FileDialog.ACCESS_USERDATA
	file_dialog.filters = ["*.save ; Save Files"]
	file_dialog.connect("file_selected", Callable(self, "_on_load_file_selected"))
	add_child(file_dialog)
	file_dialog.set_size(Vector2(800, 600))
	file_dialog.popup_centered()
	
func _on_load_file_selected(file_path: String):
	var file = FileAccess.open(file_path, FileAccess.READ)
	var data = file.get_var()
	file.close()

	clear_project()

	var node_map = {}

	# First, create all nodes and store them in a dictionary
	for node_data in data["nodes"]:
		var new_node = create_node(node_data["name"], node_data["position"])
		new_node.set_meta("id", node_data["id"])
		nodes.append(new_node)
		node_map[node_data["id"]] = new_node  # Store nodes by ID

	# Then, restore connections using the dictionary
	for conn_data in data["connections"]:
		if conn_data["from"] in node_map and conn_data["to"] in node_map:
			connect_nodes(node_map[conn_data["from"]], node_map[conn_data["to"]])

	print("Project loaded from ", file_path)
func create_menus():
	var menu_bar = MenuButton.new()
	menu_bar.text = "File"
	var popup = menu_bar.get_popup()
	popup.add_item("New Project", 0)
	popup.add_item("Open Project", 1)
	popup.add_item("Save Project", 2)
	popup.add_item("Exit", 3)
	popup.id_pressed.connect(on_menu_selected)
	add_child(menu_bar)
	print("Menu bar created")

func on_menu_selected(id: int):
	match id:
		0:
			print("New Project selected")
			clear_project()
			print("New project initialized")
		1:
			print("Open Project selected")
			load_project()
		2:
			print("Save Project selected")
			save_project()
		3:
			print("Exiting application")
			get_tree().quit()
			
func clear_project():
	# Free all nodes before clearing the list
	for node in nodes:
		node.queue_free()
	nodes.clear()

	# Clear connections
	connections.clear()
	queue_redraw()

	print("New project initialized")
	
func create_drag_and_drop_system():
	print("Drag and drop system initialized")
	for node in nodes:
		node.connect("gui_input", Callable(self, "_on_node_gui_input").bind(node))

func create_context_menu():
	context_menu = PopupMenu.new()
	context_menu.add_item("Add Layer", 0)
	context_menu.id_pressed.connect(_on_context_menu_selected)
	add_child(context_menu)

func _on_context_menu_selected(id: int):
	if id == 0:
		ask_for_neuron_count()

func ask_for_neuron_count():
	var input_dialog = AcceptDialog.new()
	input_dialog.dialog_text = "Enter the number of neurons:"
	var line_edit = LineEdit.new()
	input_dialog.add_child(line_edit)
	input_dialog.connect("confirmed", Callable(self, "_on_neuron_count_entered").bind(line_edit))
	add_child(input_dialog)
	input_dialog.popup_centered()

func _on_neuron_count_entered(line_edit: LineEdit):
	var mouse_position = get_global_mouse_position()
	var count = int(line_edit.text)
	if count > 0:
		var new_layer = create_node("Layer: " + str(count) + " neurons", mouse_position)

func _input(event):
	if event is InputEventMouseButton and event.button_index == MOUSE_BUTTON_RIGHT and event.pressed:
		var mouse_pos = get_global_mouse_position()
		var node_under_cursor = get_layer_under_mouse(mouse_pos)

		if not node_under_cursor:
			context_menu.set_position(mouse_pos)
			context_menu.popup()

func get_layer_under_mouse(mouse_pos: Vector2) -> Node:
	for node in nodes:
		if node.get_global_rect().has_point(mouse_pos):
			return node
	return null
