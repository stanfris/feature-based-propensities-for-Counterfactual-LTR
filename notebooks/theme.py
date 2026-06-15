import altair as alt
from altair import theme


@theme.register("latex", enable=True)
def theme():
    return {
        "config": {
            "title": {
                "font": "serif",
                "fontWeight": "normal",
                "fontSize": 16,
            },
            "axis": {
                "titleFont": "serif",
                "titleFontWeight": "normal",
                "titleFontSize": 16,
                "labelFont": "serif",
                "labelFontWeight": "normal",
                "labelFontSize": 16,
            },
            "headerColumn": {
                "titleFont": "serif",
                "titleFontWeight": "normal",
                "titleFontSize": 16,
                "labelFont": "serif",
                "labelFontWeight": "normal",
                "labelFontSize": 16,
            },
            "headerRow": {
                "titleFont": "serif",
                "titleFontWeight": "normal",
                "titleFontSize": 16,
                "labelFont": "serif",
                "labelFontWeight": "normal",
                "labelFontSize": 16,
            },
            "legend": {
                "titleFont": "serif",
                "titleFontWeight": "normal",
                "titleFontSize": 16,
                "labelFont": "serif",
                "labelFontWeight": "normal",
                "labelFontSize": 16,
            },
            "text": {
                "font": "serif",
                "fontSize": 14,
            },
        },
    }


alt.themes.register("latex", theme)
alt.themes.enable("latex")
