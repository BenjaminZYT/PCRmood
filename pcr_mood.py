import dash
from dash import dcc, html, Input, Output, State, callback_context
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------------------------------------------------------
# Helper function to retrieve the DJIA tickers from Wikipedia
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

# Define month options (6, 12, 18, 24, and 30 months)
month_options = [{'label': f"{m} months back", 'value': m} for m in [6, 12, 18, 24, 30]]

# -----------------------------------------------------------------------------
# Helper function to get options traded in the last 48 hours.
def get_recent_options(ticker_obj, hours=96):
    # Create a timezone-aware cutoff timestamp in UTC.
    cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(hours=hours)
    calls_list = []
    puts_list = []
    # Loop through all available expiry dates
    for expiry in ticker_obj.options:
        try:
            chain = ticker_obj.option_chain(expiry)
            calls_df = chain.calls.copy()
            puts_df = chain.puts.copy()
            # Convert the lastTradeDate column to datetime (the data is timezone-aware)
            calls_df['lastTradeDate'] = pd.to_datetime(calls_df['lastTradeDate'], errors='coerce')
            puts_df['lastTradeDate'] = pd.to_datetime(puts_df['lastTradeDate'], errors='coerce')
            # Filter for options traded in the last 48 hours
            calls_recent = calls_df[calls_df['lastTradeDate'] > cutoff]
            puts_recent = puts_df[puts_df['lastTradeDate'] > cutoff]
            if not calls_recent.empty:
                calls_list.append(calls_recent)
            if not puts_recent.empty:
                puts_list.append(puts_recent)
        except Exception as e:
            print(f"Error fetching options chain for expiry {expiry}: {e}")
    calls_all = pd.concat(calls_list) if calls_list else pd.DataFrame()
    puts_all = pd.concat(puts_list) if puts_list else pd.DataFrame()
    return calls_all, puts_all

# -----------------------------------------------------------------------------
# Create the Dash app layout
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1("Stock Sentiment Analyzer: PCR-Based Market Mood Dashboard"),
    
    # Input field for ticker symbol
    html.Div([
        html.Label("Enter ticker symbol:"),
        dcc.Input(id="ticker-input", type="text", placeholder="Enter ticker symbol", value=""),
    ]),
    html.Br(),
    
    # Dropdown to select a DJIA ticker
    html.Div([
        html.Label("Or select a DJIA ticker:"),
        dcc.Dropdown(
            id="ticker-dropdown",
            options=dropdown_options,
            placeholder="Select a DJIA ticker",
            value=None  # Initially not selected
        )
    ]),
    html.Br(),
    
    # Dropdown to choose the number of months (6–30 months in 6‐month increments)
    html.Div([
        html.Label("Select number of months for historical (nonadjusted) closing prices plot:"),
        dcc.Dropdown(
            id="months-dropdown",
            options=month_options,
            placeholder="Select number of months",
            value=None  # Initially not selected
        )
    ]),
    html.Br(),
    
    # Go and Reset buttons
    html.Div([
        html.Button("Go", id="go-button", n_clicks=0, style={"marginRight": "20px"}),
        html.Button("Reset", id="reset-button", n_clicks=0)
    ]),
    html.Br(),
    
    # Div to display any error messages
    html.Div(id="error-message", style={"color": "red", "marginTop": "20px"}),
    
    # Main content that will hold the graphs and PCR information
    html.Div(id="main-content")
])

# -----------------------------------------------------------------------------
# Combined callback to handle both "Go" and "Reset" actions.
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
    
    # If the Reset button was clicked, clear all inputs and outputs.
    if button_id == "reset-button":
        return "", "", "", None, None

    error_messages = []
    
    # Validate that at least one ticker is provided.
    if (not ticker_input) and (not ticker_dropdown):
        error_messages.append("Please enter a ticker symbol or select one from the dropdown.")
        return " ".join(error_messages), "", ticker_input, ticker_dropdown, months_selected
    
    # If both are provided, ensure they match.
    if ticker_input and ticker_dropdown:
        if ticker_input.strip().upper() != ticker_dropdown.strip().upper():
            error_messages.append("Ticker input and dropdown selection do not match. Please select one ticker.")
            return " ".join(error_messages), "", ticker_input, ticker_dropdown, months_selected
    
    # Validate that the number of months has been selected.
    if not months_selected:
        error_messages.append("Please select the number of months for historical data.")
        return " ".join(error_messages), "", ticker_input, ticker_dropdown, months_selected

    # Determine which ticker to use.
    ticker = ticker_dropdown.strip().upper() if ticker_dropdown else ticker_input.strip().upper()
    
    # Fetch historical data from Yahoo Finance.
    try:
        ticker_obj = yf.Ticker(ticker)
        end_date = pd.Timestamp.now(tz='UTC')
        # Use an approximate month duration (30 days per month)
        start_date = end_date - pd.Timedelta(days=months_selected * 30)
        hist = ticker_obj.history(start=start_date, end=end_date, interval='1d')
    except Exception as e:
        error_messages.append(f"Error fetching data for {ticker}: {e}")
        return " ".join(error_messages), "", ticker_input, ticker_dropdown, months_selected

    if hist.empty:
        error_messages.append(f"No historical data found for ticker {ticker}. Please enter a proper ticker or select from the dropdown.")
        return " ".join(error_messages), "", ticker_input, ticker_dropdown, months_selected

    # Decide which column to use: use "Adj Close" if available, otherwise use "Close".
    y_col = "Adj Close" if "Adj Close" in hist.columns else "Close"
    y_label = "Adj. Close (in USD)" if y_col == "Adj Close" else "Close (in USD)"
    
    # Create the main historical chart.
    fig_main = px.line(hist, x=hist.index, y=y_col, title="")
    fig_main.update_layout(
        annotations=[
            dict(
                text="Created by Benjamin Zu Yao Teoh | February 2025 | Alpharetta, GA",
                xref="paper",
                yref="paper",
                x=0.5,
                y=1.05,  # adjust this value if needed
                showarrow=False,
                font=dict(size=9, color="lightgray")
            )
        ]
    )
    fig_main.update_layout(xaxis_title="Date", yaxis_title="")

    # -----------------------------------------------------------------------------
    # Compute put-call ratios (PCR) for the selected ticker using options traded in the last 48 hours.
    pcr_info = []
    try:
        calls_combined, puts_combined = get_recent_options(ticker_obj, hours=96)
        if calls_combined.empty and puts_combined.empty:
            pcr_info.append(html.P("No options traded in the last 96 hours (4 days) for this ticker."))
        else:
            num_calls = len(calls_combined)
            num_puts = len(puts_combined)
            total_call_oi = calls_combined['openInterest'].sum()
            total_put_oi = puts_combined['openInterest'].sum()
            total_call_vol = calls_combined['volume'].sum()
            total_put_vol = puts_combined['volume'].sum()
            pcr_oi = total_put_oi / total_call_oi if total_call_oi != 0 else None
            pcr_vol = total_put_vol / total_call_vol if total_call_vol != 0 else None

#             pcr_info.append(html.P(f"Number of Call Options: {num_calls}"))
#             pcr_info.append(html.P(f"Number of Put Options: {num_puts}"))
#             pcr_info.append(html.P(f"PCR (Open Interest): {pcr_oi:.2f}" if pcr_oi is not None else "PCR (Open Interest): N/A"))
            pcr_info.append(html.P(f"Current PCR (Volume) for {ticker}: {pcr_vol:.2f}" if pcr_vol is not None else "PCR (Volume): N/A"))
    except Exception as e:
        pcr_info.append(html.P(f"Error computing options data: {e}"))

    # -----------------------------------------------------------------------------
    # Create a 2x2 subplot matrix for AMZN, AAPL, NVDA, and TSLA.
    subplot_tickers = ["AMZN", "AAPL", "NVDA", "TSLA"]
    fig_sub = make_subplots(rows=2, cols=2, subplot_titles=subplot_tickers)
    subplot_pcr_info = []
    row = 1
    col = 1
    for sub_ticker in subplot_tickers:
        try:
            sub_obj = yf.Ticker(sub_ticker)
            sub_hist = sub_obj.history(start=start_date, end=end_date, interval='1d')
            if not sub_hist.empty:
                # Use "Adj Close" if available; otherwise "Close"
                sub_y_col = "Adj Close" if "Adj Close" in sub_hist.columns else "Close"
                fig_sub.add_trace(
                    go.Scatter(
                        x=sub_hist.index,
                        y=sub_hist[sub_y_col],
                        mode="lines",
                        line=dict(color="purple"),
                        showlegend=False
                    ),
                    row=row, col=col
                )
            else:
                fig_sub.add_annotation(text="No data", row=row, col=col)
            
            # Compute PCR for each subplot ticker using options traded in the last 48 hours.
            calls_sub, puts_sub = get_recent_options(sub_obj, hours=96)
            if calls_sub.empty and puts_sub.empty:
                subplot_pcr_info.append(html.P(f"{sub_ticker} - No options traded in the last 96 hours (4 days)."))
            else:
                num_calls_sub = len(calls_sub)
                num_puts_sub = len(puts_sub)
                total_call_oi_sub = calls_sub['openInterest'].sum()
                total_put_oi_sub = puts_sub['openInterest'].sum()
                total_call_vol_sub = calls_sub['volume'].sum()
                total_put_vol_sub = puts_sub['volume'].sum()
                pcr_oi_sub = total_put_oi_sub / total_call_oi_sub if total_call_oi_sub != 0 else None
                pcr_vol_sub = total_put_vol_sub / total_call_vol_sub if total_call_vol_sub != 0 else None
                
#                 subplot_pcr_info.append(
#                     html.P(f"{sub_ticker} - PCR (OI): {pcr_oi_sub:.2f}" if pcr_oi_sub is not None else f"{sub_ticker} - PCR (OI): N/A")
#                 )
                subplot_pcr_info.append(
                    html.P(f"{sub_ticker} - PCR (Volume): {pcr_vol_sub:.2f}" if pcr_vol_sub is not None else f"{sub_ticker} - PCR (Volume): N/A")
                )
        except Exception as e:
            subplot_pcr_info.append(html.P(f"{sub_ticker} - Error: {e}"))
        
        # Advance to the next subplot cell
        if col == 2:
            col = 1
            row += 1
        else:
            col += 1

    fig_sub.update_layout(
        autosize=True,  # Allows the figure to scale with the browser
        title_text="Historical Closing Price (not adjusted)"
    )

    # -----------------------------------------------------------------------------
    # Assemble all components into the main content layout.
    content = [
        html.H2(f"Historical Closing Price Data for {ticker} (not adjusted)"),
        dcc.Graph(figure=fig_main),
        html.H3("Put-Call Ratios for Options (Traded in Last 96 Hours (4 days))"),
        html.Div(pcr_info),
        html.P(
            [
                "Rule of Thumb: PCR = 0.7: neutral || PCR > 0.7: bearish || PCR < 0.7: bullish. See ",
                html.A(
                    "Investopedia explanation.",
                    href="https://www.investopedia.com/ask/answers/06/putcallratio.asp",
                    target="_blank"
                )
            ],
            style={'fontSize': '14px'}
        ),
        html.H2("Corresponding Plots for Amazon, Apple, Nvidia, and Tesla"),
        dcc.Graph(figure=fig_sub),
        html.H3("Put-Call Ratios (for options traded in the last 4 days)"),
        html.Div(subplot_pcr_info)
    ]
    
    # Return outputs without altering the user's input values.
    return "", content, ticker_input, ticker_dropdown, months_selected

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app.run_server(debug=True)
