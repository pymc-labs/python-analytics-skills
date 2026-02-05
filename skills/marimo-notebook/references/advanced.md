# Marimo Advanced Features Reference

## SQL Integration

Install: `pip install "marimo[sql]"`

```python
@app.cell
def _(mo, threshold):
    # Query Python dataframes by name
    result = mo.sql(f"""
        SELECT * FROM my_dataframe
        WHERE value > {threshold.value}
        LIMIT 100
    """)
    return (result,)

# External sources
mo.sql("""
    SELECT * FROM 's3://bucket/file.parquet';
    SELECT * FROM read_csv('data.csv');
""")
```

### Supported Sources
SQL can query: Python dataframes (polars/pandas), DuckDB, PostgreSQL, MySQL, SQLite, Snowflake, BigQuery, Redshift, CSV/Parquet files, S3

### Escaping Braces
Escape literal braces with `{{}}`:
```sql
SELECT unnest([{{'a': 42, 'b': 84}}]);
```

### Output Types
Options: `native` (DuckDB relation), `lazy-polars`, `polars`, `pandas`, `auto`

## External Data Access

### CLI Arguments
```python
# Pass arguments after --
# python notebook.py -- --size 100 --name test

args = mo.cli_args()  # Dict of arguments
size = mo.cli_args().get("size") or 100
```

### URL Query Parameters
```python
params = mo.query_params()
params["key"]  # Read
params.set("key", "value")  # Write (updates URL, triggers reactivity)
```

### File Watching
```python
# Re-runs cell when file changes
watched_file = mo.watch.file("config.json")

# Re-runs when directory structure changes
watched_dir = mo.watch.directory("data/")
```

## Embedding Notebooks

```python
# Import another notebook
from other_notebook import app as other_app

@app.cell
async def _(other_app):
    # Must call from different cell than import
    result = await other_app.embed()
    return result.output, result.defs

# Run programmatically with overrides
outputs, defs = other_app.run(defs={"param": value})
```

## App Metadata

```python
meta = mo.app_meta()
meta.mode   # "edit", "run", "script", "test", or None
meta.theme  # "light" or "dark"
meta.request  # HTTP request info (headers, cookies, params)
```

## HTML Manipulation

```python
html = mo.Html("<div>Custom HTML</div>")
html.text  # Get underlying string

# Styling
styled = html.style({"color": "red", "font-size": "20px"})
styled = html.callout(kind="info")

# iframes (for scripts that need execution)
mo.iframe(html_with_scripts, width="100%", height="400px")
```

## Configuration

### Environment Variables
- `MARIMO_OUTPUT_MAX_BYTES` (default 8MB)
- `MARIMO_STD_STREAM_MAX_BYTES` (default 1MB)
- `MARIMO_SKIP_UPDATE_CHECK`
- `MARIMO_SQL_DEFAULT_LIMIT`

### User Config (`.marimo.toml`)
- Runtime: autorun, lazy mode, SQL output format
- Display: theme, font size, output placement
- Editor: hotkeys, vim mode, copilot, formatting

### App Config (in notebook file)
- Width, title, custom CSS, HTML head

## Deployment

```bash
# Docker
marimo run notebook.py --host 0.0.0.0 --port 8080

# Health endpoints
/health    # 200 OK
/healthz   # Alternative
/api/status  # JSON status
```

Platforms: Docker, Kubernetes, HuggingFace, Railway, SkyPilot, Slurm/HPC

## Testing

```bash
# Run as pytest
pytest notebook.py

# With doctest
python -m doctest notebook.py
```

marimo notebooks are pure Python—test like any Python module.

## Routing (Multi-Page Apps)

```python
mo.routes({
    "#/": render_home,
    "#/about": render_about,
    mo.routes.CATCH_ALL: render_home,
})

# Combined with navigation sidebar
mo.sidebar([
    mo.nav_menu({
        "#/": f"{mo.icon('lucide:home')} Home",
        "#/about": f"{mo.icon('lucide:user')} About",
    }, orientation="vertical")
])
```

## Lazy Loading

Defer expensive components until visible (useful for tabs/accordions):

```python
mo.lazy(expensive_component)                      # Object
mo.lazy(lambda: compute_expensive())              # Function (deferred)
mo.lazy(async_func, show_loading_indicator=True)  # Async with spinner
```
