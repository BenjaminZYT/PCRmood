import dash
from dash import dcc, html, Input, Output, State, callback_context
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Sleep between API calls to avoid Yahoo Finance rate limiting
SLEEP_SECONDS = 5

# -----------------------------------------------------------------------------    
def get_djia():
    url = 'https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average'
    try:
        tables = pd.read_html(url)
        for table in tables:
            if 'Symbol' in table.columns:
                return table, table['Symbol'].tolist()
    except Exception as e:
        print(f"Error fetching DJIA table: {e}")
        return pd.DataFrame(), []
    return pd.DataFrame(), []

djia_df, djia_tickers = get_djia()

if not djia_tickers:
    dropdown_options = [{'label': 'No data available', 'value': ''}]
else:
    dropdown_options = [{'label': ticker, 'value': ticker} for ticker in djia_tickers]

month_options = [{'label': f"{m} months back", 'value': m} for m in [6, 12, 18, 24, 30]]

# -----------------------------------------------------------------------------    
def get_recent_options(ticker_obj, hours=96):
    cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(hours=hours)
    calls_list, puts_list = [], []
    for expiry in ticker_obj.options:
        try:
            chain = ticker_obj.option_chain(expiry)
            calls_df = chain.calls.copy()
            puts_df = chain.puts.copy()
            calls_df['lastTradeDate'] = pd.to_datetime(calls_df['lastTradeDate'], errors='coerce')
            puts_df['lastTradeDate'] = pd.to_datetime(puts_df['lastTradeDate'], errors='coerce')
            calls_recent = calls_df[calls_df['lastTradeDate'] > cutoff]
            puts_recent = puts_df[puts_df['lastTradeDate'] > cutoff]
            if not calls_recent.empty:
                calls_list.append(calls_recent)
            if not puts_recent.empty:
                puts_list.append(puts_recent)
        except Exception as e:
            print(f"Error fetching options chain for expiry {expiry}: {e}")
    return pd.concat(calls_list) if calls_list else pd.DataFrame(), pd.concat(puts_list) if puts_list else pd.DataFrame()

# -----------------------------------------------------------------------------    
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1("Stock Sentiment Analyzer: PCR-Based Market Mood Dashboard"),
    html.Div([
        html.Label("Enter ticker symbol:"),
        dcc.Input(id="ticker-input", type="text", placeholder="Enter ticker symbol", value=""),
    ]),
    html.Br(),
    html.Div([
        html.Label("Or select a DJIA ticker:"),
        dcc.Dropdown(id="ticker-dropdown", options=dropdown_options, placeholder="Select a DJIA ticker", value=None)
    ]),
    html.Br(),
    html.Div([
        html.Label("Select number of months for historical (nonadjusted) closing prices plot:"),
        dcc.Dropdown(id="months-dropdown", options=month_options, placeholder="Select number of months", value=None)
    ]),
    html.Br(),
    html.Div([
        html.Button("Go", id="go-button", n_clicks=0, style={"marginRight": "20px"}),
        html.Button("Reset", id="reset-button", n_clicks=0)
    ]),
    html.Br(),
    html.Div(id="error-message", style={"color": "red", "marginTop": "20px"}),
    html.Div(id="main-content")
])

# -----------------------------------------------------------------------------    
@app.callback(
    [Output("error-message", "children"),
     Output("main-content", "children"),
     Output("ticker-input", "value"),
     Output("ticker-dropdown", "value"),
     Output("months-dropdown", "value")],
    [Input("go-button", "n_clicks"),
     Input("reset-button", "n_clicks")],
    [State("ticker-input", "value"),
     State("ticker-dropdown", "value"),
     State("months-dropdown", "value")]
)
def update_output(go_clicks, reset_clicks, ticker_input, ticker_dropdown, months_selected):
    ctx = callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == "reset-button":
        return "", "", "", None, None

    error_messages = []

    if (not ticker_input) and (not ticker_dropdown):
        return "Please enter a ticker symbol or select one from the dropdown.", "", ticker_input, ticker_dropdown, months_selected

    if ticker_input and ticker_dropdown:
        if ticker_input.strip().upper() != ticker_dropdown.strip().upper():
            return "Ticker input and dropdown selection do not match.", "", ticker_input, ticker_dropdown, months_selected

    if not months_selected:
        return "Please select the number of months for historical data.", "", ticker_input, ticker_dropdown, months_selected

    ticker = ticker_dropdown.strip().upper() if ticker_dropdown else ticker_input.strip().upper()

    try:
        ticker_obj = yf.Ticker(ticker)
        end_date = pd.Timestamp.now(tz='UTC')
        start_date = end_date - pd.Timedelta(days=months_selected * 30)
        hist = ticker_obj.history(start=start_date, end=end_date, interval='1d', auto_adjust=False)
    except Exception as e:
        return f"Error fetching data for {ticker}: {e}", "", ticker_input, ticker_dropdown, months_selected

    if hist.empty:
        return f"No historical data found for ticker {ticker}.", "", ticker_input, ticker_dropdown, months_selected

    y_col = "Adj Close" if "Adj Close" in hist.columns else "Close"
    fig_main = px.line(hist, x=hist.index, y=y_col, title="")
    fig_main.update_layout(
        annotations=[
            dict(
                text="Created by Benjamin Zu Yao Teoh | February 2025 | Alpharetta, GA",
                xref="paper", yref="paper", x=0.5, y=1.05, showarrow=False,
                font=dict(size=9, color="lightgray")
            )
        ],
        xaxis_title="Date", yaxis_title=""
    )

    pcr_info = []
    try:
        calls_combined, puts_combined = get_recent_options(ticker_obj, hours=96)
        if calls_combined.empty and puts_combined.empty:
            pcr_info.append(html.P("No options traded in the last 96 hours (4 days) for this ticker."))
        else:
            total_call_vol = calls_combined['volume'].sum()
            total_put_vol = puts_combined['volume'].sum()
            pcr_vol = total_put_vol / total_call_vol if total_call_vol else None
            pcr_info.append(html.P(f"Current PCR (Volume) for {ticker}: {pcr_vol:.2f}" if pcr_vol is not None else "PCR (Volume): N/A"))
    except Exception as e:
        pcr_info.append(html.P(f"Error computing options data: {e}"))

    subplot_tickers = ["AMZN", "AAPL", "NVDA", "TSLA"]
    fig_sub = make_subplots(rows=2, cols=2, subplot_titles=subplot_tickers)
    subplot_pcr_info = []
    row, col = 1, 1

    for sub_ticker in subplot_tickers:
        try:
            time.sleep(SLEEP_SECONDS)  # <<< prevent rate-limiting
            sub_obj = yf.Ticker(sub_ticker)
            sub_hist = sub_obj.history(start=start_date, end=end_date, interval='1d', auto_adjust=False)

            if not sub_hist.empty:
                sub_y_col = "Adj Close" if "Adj Close" in sub_hist.columns else "Close"
                fig_sub.add_trace(
                    go.Scatter(
                        x=sub_hist.index,
                        y=sub_hist[sub_y_col],
                        mode="lines",
                        # line=dict(color="purple"),
                        showlegend=False
                    ),
                    row=row, col=col
                )
            else:
                fig_sub.add_annotation(text="No data", row=row, col=col)

            calls_sub, puts_sub = get_recent_options(sub_obj, hours=96)
            if calls_sub.empty and puts_sub.empty:
                subplot_pcr_info.append(html.P(f"{sub_ticker} - No options traded in the last 96 hours."))
            else:
                total_call_vol_sub = calls_sub['volume'].sum()
                total_put_vol_sub = puts_sub['volume'].sum()
                pcr_vol_sub = total_put_vol_sub / total_call_vol_sub if total_call_vol_sub else None
                subplot_pcr_info.append(html.P(
                    f"{sub_ticker} - PCR (Volume): {pcr_vol_sub:.2f}" if pcr_vol_sub is not None else f"{sub_ticker} - PCR (Volume): N/A"
                ))
        except Exception as e:
            subplot_pcr_info.append(html.P(f"{sub_ticker} - Error: {e}"))

        if col == 2:
            col = 1
            row += 1
        else:
            col += 1

    fig_sub.update_layout(
        autosize=True,
        title_text="Historical Closing Price (not adjusted)"
    )

    content = [
        html.H2(f"Historical Closing Price Data for {ticker} (not adjusted)"),
        dcc.Graph(figure=fig_main),
        html.H3("Put-Call Ratios for Options (Traded in Last 96 Hours (4 days))"),
        html.Div(pcr_info),
        html.P(
            ["Rule of Thumb: PCR = 0.7: neutral || PCR > 0.7: bearish || PCR < 0.7: bullish. See ",
             html.A("Investopedia explanation.",
                    href="https://www.investopedia.com/ask/answers/06/putcallratio.asp",
                    target="_blank")],
            style={'fontSize': '14px'}
        ),
        html.H2("Corresponding Plots for Amazon, Apple, Nvidia, and Tesla"),
        dcc.Graph(figure=fig_sub),
        html.H3("Put-Call Ratios (for options traded in the last 4 days)"),
        html.Div(subplot_pcr_info)
    ]

    return "", content, ticker_input, ticker_dropdown, months_selected

# -----------------------------------------------------------------------------    
if __name__ == "__main__":
    app.run_server(debug=True)
