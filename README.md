# ML_Simulator

## Overview

ML_Simulator is a fully customizable Machine Learning IDE that empowers you to design, train, and test neural network models through an intuitive drag-and-drop interface. Built using Python and PyQt5, the IDE provides a visual environment for creating complex models with ease—ideal for both rapid prototyping and in-depth experimentation.

## Features

- **Drag-and-Drop Model Creation:** Build your neural network by visually connecting nodes.
- **Customizable Components:** Create and modify nodes with customizable settings such as activation functions, biases, and neuron counts.
- **Interactive Network Visualization:** Watch your network come to life with real-time updates as connections are made and weights are randomized.
- **Training and Evaluation:** Train your models using built-in backpropagation and feedforward routines, with support for both SGD and Adam optimizers.
- **Dynamic UI Elements:** Enjoy a polished, interactive UI that lets you add training data via a table interface, adjust node settings, and toggle between different visual representations.

## Installation

### Prerequisites

- **Python 3.x:** Ensure you have Python 3 installed on your system.
- **Required Libraries:**  
  - [PyQt5](https://pypi.org/project/PyQt5/)
  - [numpy](https://pypi.org/project/numpy/)

### Clone the Repository

Open your terminal and execute:

```bash
git clone https://github.com/YourUsername/ML_Simulator.git
cd ML_Simulator
```

### Install Dependencies

Install the necessary packages by running:

```bash
pip install -r requirements.txt
```

## Usage

To launch ML_Simulator, simply run:

```bash
python3 main.py
```

Once started, use the drag-and-drop interface to add nodes, set up network connections, and train your machine learning models interactively.

## Roadmap

- **Real-Time Training Visualization:** Add live training updates to monitor progress as the model learns.
- **Prebuilt Model Templates:** Integrate templates for popular architectures such as CNNs, RNNs, etc.
- **Model Export/Import:** Enable features for saving and loading trained models for future use.
- **Enhanced UI/UX:** Continuously refine the interface to boost usability and performance.
[TODO](./TODO)

## Contributing

Contributions are welcome! To get involved:

1. Fork the repository.
2. Create a feature branch.
3. Submit a pull request with your enhancements.

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
