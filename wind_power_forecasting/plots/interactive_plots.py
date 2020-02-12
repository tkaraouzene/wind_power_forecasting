from typing import Optional, Union, List

import pandas as pd
import plotly.graph_objs as go
import plotly.offline as po
from plotly.subplots import make_subplots


def plot_offline_figure(fig, name='', filename='chart.html'):
    po.plot(fig, name, filename=filename)


def plot_timeseries(data: Union[pd.DataFrame, pd.Series], subplots: Optional[List[List[str]]] = None,
                    name: str = 'timeseries', mode='lines', title='', filename='chart.html'):
    """

    Parameters
    ----------
    data: pandas.Series to plot WITH TIME AS INDEX, or an union of series and/or dataframes.

    subplots: if several DataFrames to plot

    name: name of the figure, by default equal to 'timeseries'

    mode: gives the shape of the plot, 'lines', 'markers', ... see plotly documentation

    title: title of the plot

    Returns
    -------
    fig:
    """

    if isinstance(data, pd.Series):
        plots = [go.Scatter(x=data.index, y=data, name=data.name, mode=mode)]
        fig = go.Figure(data=plots, layout=go.Layout(title=title))
    else:
        fig = make_subplots(rows=len(subplots), cols=1, shared_xaxes=True, print_grid=False,
                            subplot_titles=title)

        idx_subplot = 1
        for col_list in subplots:
            for col in col_list:
                trace = go.Scatter(x=data.index, y=data[col], name=col, mode=mode)
                fig.append_trace(trace, idx_subplot, 1)
            idx_subplot += 1

    plot_offline_figure(fig, name=name, filename=filename)
    return fig
