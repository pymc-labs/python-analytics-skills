import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
        # Data Analysis Notebook

        This notebook provides a template for exploratory data analysis.
        """
    )
    return


@app.cell
def _():
    import polars as pl

    return (pl,)


@app.cell
def _(mo):
    file_input = mo.ui.file(filetypes=[".csv", ".parquet"], label="Upload data file")
    file_input
    return (file_input,)


@app.cell
def _(file_input, pl):
    if file_input.value:
        contents = file_input.contents()
        name = file_input.name()
        if name.endswith(".csv"):
            df = pl.read_csv(contents)
        elif name.endswith(".parquet"):
            df = pl.read_parquet(contents)
        else:
            df = None
    else:
        df = None
    df
    return (df,)


@app.cell
def _(df, mo):
    mo.stop(df is None, mo.md("Upload a file to begin analysis"))
    mo.md(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
    return


@app.cell
def _(df, mo):
    mo.stop(df is None)
    mo.ui.table(df.head(100))
    return


if __name__ == "__main__":
    app.run()
