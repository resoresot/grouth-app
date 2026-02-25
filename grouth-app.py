import flet as ft
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import time
from datetime import datetime, timedelta
import threading
import traceback


# --- HTMLレポート生成（リッチデザイン版） ---
def generate_html_report(df: pd.DataFrame, period: str):
    # レイアウトを 3行構成に変更してグラフを大きく見せる
    # 1行目: 累積損益 (全体幅)
    # 2行目: R倍数分布 | 保有日数分布
    # 3行目: 年別損益 (全体幅)
    
    fig = make_subplots(
        rows=3, 
        cols=2, 
        subplot_titles=(
            "<b>累積損益推移</b> (Equity Curve)", 
            "<b>R倍数分布</b> (Risk/Reward Distribution)", 
            "<b>保有日数分布</b> (Holding Days)",
            "<b>年別損益</b> (Yearly P/L)"
        ),
        specs=[
            [{"colspan": 2}, None],  # Row 1: Equity
            [{}, {}],                # Row 2: R Dist, Days Dist
            [{"colspan": 2}, None]   # Row 3: Yearly
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    # 1. 累積損益 (Area Chartでリッチに)
    fig.add_trace(
        go.Scatter(
            x=df["EntryDate"], 
            y=df["CumPL"], 
            name="累積損益",
            mode='lines',
            fill='tozeroy', # 塗りつぶし
            line=dict(width=2, color='#00CC96'), # 鮮やかなグリーン
            hovertemplate='%{x}<br>累積: %{y:,.0f}円'
        ), 
        row=1, col=1
    )

    # 2. R倍数分布 (Histogram)
    fig.add_trace(
        go.Histogram(
            x=df["R"], 
            nbinsx=40, 
            name="R倍数",
            marker_color='#636EFA', # Plotly Blue
            opacity=0.8,
            hovertemplate='R倍数: %{x}<br>回数: %{y}'
        ), 
        row=2, col=1
    )
    # 建値(0)のライン
    fig.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.5, row=2, col=1)

    # 3. 保有日数分布 (Histogram)
    fig.add_trace(
        go.Histogram(
            x=df["Days"], 
            nbinsx=20, 
            name="保有日数",
            marker_color='#AB63FA', # Purple
            opacity=0.8,
            hovertemplate='日数: %{x}<br>回数: %{y}'
        ), 
        row=2, col=2
    )

    # 4. 年別損益 (勝ち負けで色分け)
    df["Year"] = pd.to_datetime(df["EntryDate"]).dt.year
    yearly = df.groupby("Year")["PL"].sum()
    # 利益なら緑、損失なら赤
    colors = ['#EF553B' if x < 0 else '#00CC96' for x in yearly.values]
    
    fig.add_trace(
        go.Bar(
            x=yearly.index, 
            y=yearly.values, 
            name="年別",
            marker_color=colors,
            hovertemplate='Year: %{x}<br>PL: %{y:,.0f}円'
        ), 
        row=3, col=1
    )

    # --- デザインの全体適用 (Dark Theme) ---
    fig.update_layout(
        template="plotly_dark", # ダークテーマ適用
        height=1100, # 縦長にして見やすく
        title={
            'text': f"<b>Growth Strategy Backtest Report</b><br><span style='font-size:14px; color:#cccccc'>Period: {period}</span>",
            'y':0.96,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        font=dict(family="Meiryo, sans-serif", size=12),
        margin=dict(l=50, r=50, t=100, b=50),
        showlegend=False, # 凡例はグラフタイトルで分かるので消してスッキリさせる
        hovermode="x unified" # ホバー時に縦線が出て見やすく
    )
    
    # 軸のフォーマット（円マークやカンマ区切り）
    fig.update_yaxes(tickformat=",", title="円", row=1, col=1)
    fig.update_yaxes(tickformat=",", title="円", row=3, col=1)
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#333', row=1, col=1)

    fig.write_html("report_gui.html")


# --- バックテスト実行ロジック ---
def run_backtest_logic(settings, page: ft.Page, progress_bar: ft.ProgressBar, log_text: ft.Text, result_area: ft.Text):
    ticker_file = "growth_tickers.txt"

    if not os.path.exists(ticker_file):
        log_text.value = f"エラー: {ticker_file} が見つかりません。"
        log_text.color = "red"
        progress_bar.visible = False
        page.update()
        return

    with open(ticker_file, "r", encoding="utf-8") as f:
        tickers = [line.strip() for line in f if line.strip()]

    total_count = len(tickers)
    if total_count == 0:
        log_text.value = "エラー: 銘柄リストが空です (growth_tickers.txt)。"
        log_text.color = "red"
        progress_bar.visible = False
        page.update()
        return

    # 設定値
    budget = int(settings["budget"])
    start_date = settings["start_date"]
    end_date = settings["end_date"]

    ma_short = int(settings["ma_short"])
    ma_long = int(settings["ma_long"])
    rsi_min = int(settings["rsi_min"])
    rsi_max = int(settings["rsi_max"])
    atr_sl = float(settings["atr_sl"])
    atr_tp = float(settings["atr_tp"])
    use_trailing = bool(settings["use_trailing"])

    all_results = []

    log_text.value = f"検証開始: {total_count}銘柄 / {start_date} ～ {end_date}"
    log_text.color = "white"
    progress_bar.value = 0
    progress_bar.visible = True
    page.update()

    # 進捗更新を少し間引く
    last_update_t = time.time()

    for i, ticker in enumerate(tickers):
        try:
            # 進捗
            progress_bar.value = (i + 1) / total_count

            # UI更新は0.2秒ごと＋5銘柄ごと
            if (i + 1) % 5 == 0 or (time.time() - last_update_t) > 0.2:
                log_text.value = f"分析中 ({i+1}/{total_count}): {ticker}"
                page.update()
                last_update_t = time.time()

            # データ取得（ネットワークなので時間がかかります）
            data = yf.download(ticker, start=start_date, end=end_date, progress=False, threads=False)

            if data is None or data.empty or len(data) < 50:
                continue

            # MultiIndex対策
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            # ATR
            high_low = data["High"] - data["Low"]
            ranges = pd.concat(
                [
                    high_low,
                    np.abs(data["High"] - data["Close"].shift()),
                    np.abs(data["Low"] - data["Close"].shift()),
                ],
                axis=1,
            )
            data["ATR"] = np.max(ranges, axis=1).rolling(14).mean()

            # MA
            data["MA_S"] = data["Close"].rolling(ma_short).mean()
            data["MA_L"] = data["Close"].rolling(ma_long).mean()

            # RSI
            delta = data["Close"].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            data["RSI"] = 100 - (100 / (1 + rs))

            # Volume change
            data["VolChg"] = data["Volume"].pct_change()

            # エントリー条件
            entry_signal = (
                (data["MA_S"] > data["MA_L"])
                & (data["MA_S"].shift(1) <= data["MA_L"].shift(1))
                & (data["RSI"] > rsi_min)
                & (data["RSI"] < rsi_max)
                & (data["VolChg"] > 0.3)
            )

            sig_dates = data.index[entry_signal]

            for sig_date in sig_dates:
                try:
                    idx = data.index.get_loc(sig_date) + 1
                    if idx >= len(data):
                        continue

                    entry_price = float(data.iloc[idx]["Open"])
                    atr = float(data.loc[sig_date, "ATR"])
                    if np.isnan(atr) or entry_price <= 0:
                        continue

                    share_count = budget / entry_price

                    initial_risk = atr * atr_sl
                    current_sl = entry_price - initial_risk
                    tp_price = entry_price + (atr * atr_tp)

                    highest_price = entry_price
                    future = data.iloc[idx + 1 :]
                    status = "未決済"
                    exit_price = float(data.iloc[-1]["Close"])
                    holding_days = len(future)

                    for day_count, (_, row) in enumerate(future.iterrows(), 1):
                        curr_high = float(row["High"])
                        curr_low = float(row["Low"])

                        if use_trailing:
                            highest_price = max(highest_price, curr_high)
                            current_sl = max(current_sl, highest_price - initial_risk)

                        if curr_low <= current_sl:
                            status = "損切/TS"
                            exit_price = current_sl
                            holding_days = day_count
                            break
                        elif (not use_trailing) and (curr_high >= tp_price):
                            status = "利確"
                            exit_price = tp_price
                            holding_days = day_count
                            break

                    pl_yen = (exit_price - entry_price) * share_count
                    risk_yen = initial_risk * share_count
                    r_mult = pl_yen / risk_yen if risk_yen != 0 else 0

                    all_results.append(
                        {
                            "Ticker": ticker,
                            "EntryDate": sig_date.date(),
                            "Status": status,
                            "PL": pl_yen,
                            "R": round(float(r_mult), 2),
                            "Days": int(holding_days),
                        }
                    )
                except Exception:
                    continue

            time.sleep(0.01)

        except Exception:
            continue

    # 結果反映
    if all_results:
        df = pd.DataFrame(all_results).sort_values("EntryDate")
        df["CumPL"] = df["PL"].cumsum()

        generate_html_report(df, f"{start_date} to {end_date}")
        df.to_csv("app_trade_details.csv", index=False, encoding="utf-8-sig")

        total_pl = float(df["PL"].sum())
        win_rate = (len(df[df["PL"] > 0]) / len(df) * 100) if len(df) > 0 else 0
        avg_r = float(df["R"].mean())

        result_text = (
            f"【検証完了】\n"
            f"総トレード数: {len(df)} 回\n"
            f"合計損益: {total_pl:,.2f} 円\n"
            f"勝率: {win_rate:.1f} %\n"
            f"平均R倍数: {avg_r:.2f} R"
        )

        result_area.value = result_text
        result_area.color = "green" if total_pl > 0 else "red"
        log_text.value = "完了: 'report_gui.html' と 'app_trade_details.csv' を生成しました。"
        log_text.color = "white"
    else:
        log_text.value = "条件に合うトレードが見つかりませんでした。"
        log_text.color = "white"
        result_area.value = ""
        result_area.color = None

    progress_bar.visible = False
    page.update()


def main(page: ft.Page):
    page.title = "Growth Strategy Tester"
    page.theme_mode = ft.ThemeMode.DARK
    page.window_width = 500
    page.window_height = 900

    # --- 日付設定 ---
    today = datetime.now()
    default_start = today - timedelta(days=365 * 5)

    start_date_val = ft.Text(default_start.strftime("%Y-%m-%d"), size=16)
    end_date_val = ft.Text(today.strftime("%Y-%m-%d"), size=16)

    def change_start(e):
        if e.control.value:
            corrected_date = e.control.value + timedelta(hours=12)
            start_date_val.value = corrected_date.strftime("%Y-%m-%d")
            page.update()

    def change_end(e):
        if e.control.value:
            corrected_date = e.control.value + timedelta(hours=12)
            end_date_val.value = corrected_date.strftime("%Y-%m-%d")
            page.update()

    date_picker_start = ft.DatePicker(
        on_change=change_start,
        first_date=datetime(2010, 1, 1),
        last_date=datetime(2030, 12, 31),
    )
    date_picker_end = ft.DatePicker(
        on_change=change_end,
        first_date=datetime(2010, 1, 1),
        last_date=datetime(2030, 12, 31),
    )

    # 0.81.0: overlayに追加して open=True で開く
    page.overlay.append(date_picker_start)
    page.overlay.append(date_picker_end)

    btn_start_date = ft.FilledButton(
        "開始日",
        icon="calendar_month",
        on_click=lambda _: (setattr(date_picker_start, "open", True), page.update()),
    )
    btn_end_date = ft.FilledButton(
        "終了日",
        icon="calendar_month",
        on_click=lambda _: (setattr(date_picker_end, "open", True), page.update()),
    )

    # --- 資金 ---
    budget_input = ft.TextField(
        label="1トレード予算 (円)",
        value="100000",
        keyboard_type=ft.KeyboardType.NUMBER,
    )

    # --- スライダーと数値表示 ---
    txt_atr_sl = ft.Text("1.5", size=16, weight="bold")
    txt_atr_tp = ft.Text("3.0", size=16, weight="bold")
    txt_ma_s = ft.Text("5", size=16)
    txt_ma_l = ft.Text("25", size=16)

    def update_text(e, target_text: ft.Text, fmt="{:.1f}"):
        target_text.value = fmt.format(float(e.control.value))
        page.update()

    atr_sl_sl = ft.Slider(
        min=0.5,
        max=5.0,
        divisions=45,
        value=1.5,
        label=None,  # 吹き出しは消す（0.81で丸め表示が起きやすい）
        on_change=lambda e: update_text(e, txt_atr_sl, "{:.1f}"),
    )
    atr_tp_sl = ft.Slider(
        min=1.0,
        max=10.0,
        divisions=90,
        value=3.0,
        label=None,
        on_change=lambda e: update_text(e, txt_atr_tp, "{:.1f}"),
    )
    ma_s_sl = ft.Slider(
        min=3,
        max=50,
        divisions=47,
        value=5,
        label=None,
        on_change=lambda e: update_text(e, txt_ma_s, "{:.0f}"),
    )
    ma_l_sl = ft.Slider(
        min=10,
        max=200,
        divisions=190,
        value=25,
        label=None,
        on_change=lambda e: update_text(e, txt_ma_l, "{:.0f}"),
    )

    rsi_range = ft.RangeSlider(min=0, max=100, divisions=100, start_value=40, end_value=60, label="{value}%")
    trailing_sw = ft.Switch(label="トレイリングストップを使用する", value=True)

    # --- 実行ボタン ---
    start_btn = ft.FilledButton(
        "バックテスト実行",
        icon="play_arrow",
        height=50,
        style=ft.ButtonStyle(
            shape=ft.RoundedRectangleBorder(radius=10),
            bgcolor="blue",
            color="white",
        ),
    )

    # --- プログレスバー（高さ固定） ---
    progress_bar = ft.ProgressBar(value=0, visible=False)
    progress_container = ft.Container(content=progress_bar, height=5)

    # --- ログと結果 ---
    log_text = ft.Text("待機中...", size=12, color="grey")  # 初期表示
    result_area = ft.Text("", size=16, weight=ft.FontWeight.BOLD)

    def on_click_start(e):
        # クリック直後にUIが更新されるように、処理は別スレッドで回す
        start_btn.disabled = True
        progress_bar.visible = True
        progress_bar.value = 0
        log_text.value = "検証準備中..."
        log_text.color = "white"
        result_area.value = ""
        page.update()

        settings = {
            "start_date": start_date_val.value,
            "end_date": end_date_val.value,
            "budget": budget_input.value,
            "ma_short": ma_s_sl.value,
            "ma_long": ma_l_sl.value,
            "rsi_min": rsi_range.start_value,
            "rsi_max": rsi_range.end_value,
            "atr_sl": atr_sl_sl.value,
            "atr_tp": atr_tp_sl.value,
            "use_trailing": trailing_sw.value,
        }

        def worker():
            try:
                run_backtest_logic(settings, page, progress_bar, log_text, result_area)
            except Exception as ex:
                log_text.value = f"エラー: {ex}"
                log_text.color = "red"
                # 例外詳細をファイルに出力
                with open("app_error.log", "w", encoding="utf-8") as f:
                    f.write(traceback.format_exc())
            finally:
                start_btn.disabled = False
                progress_bar.visible = False
                page.update()

        threading.Thread(target=worker, daemon=True).start()

    start_btn.on_click = on_click_start

    # --- メインレイアウト ---
    page.add(
        ft.Column(
            [
                ft.Text("グロース全銘柄 バックテスト", size=24, weight=ft.FontWeight.BOLD),
                ft.Divider(),
                ft.Text("期間設定", size=16, color="cyan"),
                ft.Row([btn_start_date, start_date_val], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                ft.Row([btn_end_date, end_date_val], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                ft.Container(height=10),
                ft.Text("資金設定", size=16, color="cyan"),
                budget_input,
                ft.Divider(),
                ft.Text("エントリー条件", size=16, color="cyan"),
                ft.Row([ft.Text("短期MA期間"), txt_ma_s], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                ma_s_sl,
                ft.Row([ft.Text("長期MA期間"), txt_ma_l], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                ma_l_sl,
                ft.Text("RSI 範囲 (%)"),
                rsi_range,
                ft.Divider(),
                ft.Text("イグジット設定", size=16, color="cyan"),
                ft.Row([ft.Text("損切幅 (ATR倍率)"), txt_atr_sl], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                atr_sl_sl,
                ft.Row([ft.Text("利確目安 (ATR倍率)"), txt_atr_tp], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                atr_tp_sl,
                trailing_sw,
                ft.Divider(),
                start_btn,
                ft.Container(height=10),
                progress_container,
                log_text,
                ft.Divider(),
                ft.Container(
                    content=result_area,
                    padding=10,
                    border=ft.Border.all(1, "grey"),
                    border_radius=5,
                ),
                ft.Container(height=50),
            ],
            scroll=ft.ScrollMode.AUTO,
            expand=True,
        )
    )


ft.run(main)