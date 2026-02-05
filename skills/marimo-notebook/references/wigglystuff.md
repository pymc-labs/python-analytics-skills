# Wigglystuff Widget Reference

Creative AnyWidgets for marimo and other Python notebook environments.

## Installation

```bash
uv pip install wigglystuff
# or
pip install wigglystuff
```

## Widgets

### Slider2D
Two-dimensional slider returning x/y coordinates.
```python
from wigglystuff import Slider2D
import marimo as mo

slider = mo.ui.anywidget(Slider2D())
slider  # Display

# Access values
slider.x, slider.y
```

### Matrix
Interactive matrix input with configurable dimensions.
```python
from wigglystuff import Matrix
import marimo as mo

matrix = mo.ui.anywidget(Matrix(rows=3, cols=3, step=0.1))
matrix

# Get matrix values
matrix.data  # Returns nested list
```

### Paint
MS Paint-style drawing canvas.
```python
from wigglystuff import Paint
import marimo as mo

# Empty canvas
paint = mo.ui.anywidget(Paint(width=400, height=300))

# Start with image
paint = mo.ui.anywidget(Paint.from_path("image.png"))

# Get result
paint.to_pil()     # PIL Image
paint.to_base64()  # Base64 string
```

### EdgeDraw
Draw edges/connections between nodes.
```python
from wigglystuff import EdgeDraw
import marimo as mo

nodes = [{"id": "a", "x": 50, "y": 50}, {"id": "b", "x": 150, "y": 50}]
edge_draw = mo.ui.anywidget(EdgeDraw(nodes=nodes))
edge_draw

# Get edges
edge_draw.edges  # List of {"source": ..., "target": ...}
```

### SortableList
Drag-and-drop reorderable list.
```python
from wigglystuff import SortableList
import marimo as mo

items = ["First", "Second", "Third"]
sortable = mo.ui.anywidget(SortableList(items=items))
sortable

# Get current order
sortable.items
```

### ColorPicker
Interactive color selection.
```python
from wigglystuff import ColorPicker
import marimo as mo

picker = mo.ui.anywidget(ColorPicker(value="#ff0000"))
picker

# Get color
picker.value  # "#ff0000" format
```

### KeystrokeWidget
Capture keyboard input.
```python
from wigglystuff import KeystrokeWidget
import marimo as mo

keys = mo.ui.anywidget(KeystrokeWidget())
keys

# Get last keystroke
keys.key        # Key pressed
keys.ctrl       # Modifier states
keys.shift
keys.alt
keys.meta
```

### GamepadWidget
Capture gamepad/controller input.
```python
from wigglystuff import GamepadWidget
import marimo as mo

gamepad = mo.ui.anywidget(GamepadWidget())
gamepad

# Get state
gamepad.buttons  # Button states
gamepad.axes     # Joystick positions
```

### SpeechToText (WebkitSpeechToTextWidget)
Voice transcription using browser speech API.
```python
from wigglystuff import WebkitSpeechToTextWidget
import marimo as mo

speech = mo.ui.anywidget(WebkitSpeechToTextWidget())
speech

# Control and read
speech.listening = True   # Start recording
speech.transcript         # Get transcribed text
```

### CopyToClipboard
Button to copy content to clipboard.
```python
from wigglystuff import CopyToClipboard
import marimo as mo

copy_btn = mo.ui.anywidget(CopyToClipboard(content="Text to copy"))
copy_btn
```

### WebcamCapture
Capture images from webcam.
```python
from wigglystuff import WebcamCapture
import marimo as mo

webcam = mo.ui.anywidget(WebcamCapture())
webcam

# Get captured image
webcam.image  # Base64 encoded image
```

### CellTour
Interactive tour/walkthrough of notebook cells.
```python
from wigglystuff import CellTour
import marimo as mo

tour = mo.ui.anywidget(CellTour(steps=[
    {"cell": "cell_1", "message": "This is step 1"},
    {"cell": "cell_2", "message": "This is step 2"},
]))
tour
```

## Integration Pattern

All wigglystuff widgets integrate with marimo via `mo.ui.anywidget()`:

```python
from wigglystuff import SomeWidget
import marimo as mo

# Wrap the widget
widget = mo.ui.anywidget(SomeWidget(...))

# Display it
widget

# Access its value reactively in another cell
widget.value  # or specific properties like widget.x, widget.data, etc.
```

Changes to widget values trigger marimo's reactive execution.
