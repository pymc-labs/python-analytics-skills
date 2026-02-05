# Marimo UI Components Reference

## Input Elements

### Text & Numbers
```python
mo.ui.text(value="", label="Label", placeholder="hint")
mo.ui.text_area(value="", label="Label", rows=4)
mo.ui.number(value=0, start=0, stop=100, step=1, label="Label")
mo.ui.slider(start=0, stop=100, value=50, step=1, label="Label")
mo.ui.range_slider(start=0, stop=100, value=[25, 75], step=1)
```

### Selection
```python
mo.ui.checkbox(value=False, label="Label")
mo.ui.switch(value=False, label="Label")
mo.ui.radio(options=["a", "b", "c"], value="a", label="Label")
mo.ui.dropdown(options=["a", "b", "c"], value="a", label="Label")
mo.ui.dropdown(options={"One": 1, "Two": 2}, value="One")  # Dict mapping
mo.ui.multiselect(options=["a", "b", "c"], value=["a"], label="Label")
```

### Date & Time
```python
mo.ui.date(value=None, label="Date")
mo.ui.date_range(start=None, stop=None)
```

### Files & Media
```python
mo.ui.file(filetypes=[".csv", ".json", ".parquet"], multiple=False, label="Upload")
mo.ui.microphone()  # Audio recording
```

### Buttons
```python
mo.ui.button(label="Click", on_click=lambda _: print("clicked"))
mo.ui.run_button(label="Run")  # Only triggers on click, not reactive
```

### From DataFrame Series
```python
mo.ui.dropdown.from_series(df["column"])
mo.ui.slider.from_series(df["column"])
mo.ui.multiselect.from_series(df["column"])
mo.ui.range_slider.from_series(df["column"])
```

## Data Display

### Tables
```python
mo.ui.table(data)                         # Interactive with row selection
mo.ui.table(data, selection=None)         # No selection
mo.ui.table(data, selection="single")     # Single row selection
mo.ui.table(data, selection="multi")      # Multiple row selection
mo.ui.table(data, pagination=True, page_size=10)
table.value                               # Selected rows as DataFrame
```

### DataFrames
```python
mo.ui.dataframe(df)                       # Full dataframe explorer
mo.ui.data_explorer(df)                   # Interactive data explorer
```

### Charts (Interactive with Selection)
```python
# Altair
chart = alt.Chart(df).mark_point().encode(x="x", y="y")
selection = mo.ui.altair_chart(chart)
selection.value  # DataFrame of selected points

# Plotly
fig = px.scatter(df, x="x", y="y")
selection = mo.ui.plotly(fig)
selection.value  # Selected data
```

## Layout

### Stacks
```python
mo.hstack([elem1, elem2], justify="start", align="center", gap=1)
mo.vstack([elem1, elem2], align="stretch", gap=1)
# justify: start, center, end, space-between, space-around
# align: start, center, end, stretch
```

### Containers
```python
mo.accordion({"Section 1": content1, "Section 2": content2})
mo.tabs({"Tab 1": content1, "Tab 2": content2})
mo.carousel([item1, item2, item3])
mo.sidebar([content], footer=footer_content)
mo.nav_menu({"Home": home, "About": about})
```

### Grid
```python
mo.ui.batch(
    mo.md("**X**: {x}\n**Y**: {y}"),
    x=mo.ui.slider(0, 10),
    y=mo.ui.slider(0, 10)
)
```

## Output & Formatting

### Markdown
```python
mo.md("# Header")
mo.md(f"Value: **{variable}**")           # Interpolation
mo.md(f"Result: {mo.as_html(widget)}")    # Embed widgets
```

### Callouts
```python
mo.callout("Info message", kind="info")
mo.callout("Warning!", kind="warn")
mo.callout("Error!", kind="danger")
mo.callout("Success!", kind="success")
mo.callout("Note", kind="neutral")
```

### Status
```python
mo.status.spinner(title="Loading...")
mo.status.progress_bar(value=0.5)
```

### Media
```python
mo.image(src="path.png", alt="description", width=400)
mo.image(src=pil_image)                   # PIL Image
mo.video(src="path.mp4")
mo.audio(src="path.mp3")
mo.pdf(src="path.pdf")
mo.download(data, filename="file.csv", label="Download")
```

### Trees & Code
```python
mo.tree({"a": 1, "b": {"c": 2}})          # Tree view
mo.plain_text("text")
mo.show_code()                            # Show cell source
mo.lazy(expensive_element)                # Lazy loading
```

## Composite Elements

### Forms
```python
form = mo.ui.form(
    mo.md("""
    **Name**: {name}
    **Age**: {age}
    **Subscribe**: {subscribe}
    """).batch(
        name=mo.ui.text(),
        age=mo.ui.number(start=0, stop=120),
        subscribe=mo.ui.checkbox()
    ),
    submit_button_label="Submit"
)
form.value  # Dict after submission: {"name": ..., "age": ..., "subscribe": ...}
```

### Arrays & Dictionaries
```python
mo.ui.array([mo.ui.text()] * 3)                    # List of elements
mo.ui.array([mo.ui.text()], add_button=True)       # Growable
mo.ui.dictionary({"key1": mo.ui.text(), "key2": mo.ui.number()})
```

### Batched Elements
```python
batch = mo.md("""
X: {x}
Y: {y}
""").batch(x=mo.ui.slider(0, 10), y=mo.ui.slider(0, 10))
batch.value  # {"x": ..., "y": ...}
```

## Control Flow

```python
mo.stop(condition, fallback_output)       # Stop if condition True
mo.stop(df is None, mo.md("*No data*"))
```

## AnyWidget Integration

```python
from some_anywidget import CustomWidget
widget = mo.ui.anywidget(CustomWidget(...))
widget.value  # Access as dict
widget.property  # Access specific properties
```

## Accessing Values

All UI elements expose `.value`:
```python
slider = mo.ui.slider(0, 100, value=50)
slider.value  # 50, updates reactively

dropdown = mo.ui.dropdown(["a", "b"])
dropdown.value  # Selected option

table = mo.ui.table(df)
table.value  # Selected rows as DataFrame
```

## Event Handlers

```python
mo.ui.button(on_click=lambda event: do_something())
mo.ui.text(on_change=lambda value: handle_change(value))
```
