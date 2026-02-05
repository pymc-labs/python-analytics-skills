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
        # Notebook Title

        Description of what this notebook does.
        """
    )
    return


@app.cell
def _():
    # Your code here
    pass
    return


if __name__ == "__main__":
    app.run()
