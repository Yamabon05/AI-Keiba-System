import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import joblib
import os
import re
import time
import optuna
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from tabulate import tabulate
import warnings
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# 警告を非表示
warnings.filterwarnings('ignore')

class KeibaSystem:
    def __init__(self, csv_path='all_keiba_results.csv'):
        self.csv_path = csv_path

        # ★ 追加: 予想ログファイルのパス
        self.log_path = 'prediction_log.csv'

        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        # ★ 特徴量定義 (26個)
        self.features = [
            '枠 番', '馬 番', '斤量', '体重', '体重増減', '騎手ID', '場ID', '馬場ID', 'トラックID',
            '馬_前3走平均着順', '馬_同場複勝率', '馬_同馬場複勝率', 'レース番号', 'クラスチェンジ', 
            '斤量変化', '距離変化', '前走タイム指数', 'タイム指数順位', '前走着順順位', '騎手の勢い',
            '距離変化区分', '着順変動', '乗り替わり', '騎手_本賞_連対率', 
            'タイム指数_レース内差', '斤量_レース内差'
        ]

        print("📁 システムを初期化中...")
        if os.path.exists(self.csv_path):
            self.history_df = pd.read_csv(self.csv_path, dtype={'race_id': str})
            self.history_df = self.history_df.dropna(subset=['race_id'])
            print(f"   過去データ読み込み完了: {len(self.history_df)} 行")
        else:
            print("   過去データファイルがありません。新規作成します。")
            self.history_df = pd.DataFrame()

        # LightGBMのハイパーパラメータ
        self.best_params = {
            'objective': 'lambdarank', 
            'metric': 'ndcg', 
            'ndcg_eval_at': [1, 3],
            'verbosity': -1, 
            'boosting_type': 'gbdt', 
            'random_state': 42,
            'learning_rate': 0.05, 
            'num_leaves': 31,
            'max_depth': 7, 
            'min_child_samples': 20, 
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }
        
        self.models = self.load_models()
        
        # マスタデータえrwfm
        self.place_dict = {"01":"札幌","02":"函館","03":"福島","04":"新潟","05":"東京","06":"中山","07":"中京","08":"京都","09":"阪神","10":"小倉"}
        self.cond_dict = {"良": "良", "稍": "稍重", "重": "重", "不": "不良"}
        self.weather_list = ["晴", "曇", "雨", "小雨", "雪", "小雪"]
        
        print("✅ システム起動完了。")

    def load_models(self):
        models = []
        seeds = [42, 100, 2024]
        for seed in seeds:
            path = f'production_model_2026_seed_{seed}.pkl'
            if os.path.exists(path):
                try:
                    models.append(joblib.load(path))
                except:
                    print(f"⚠️ モデル {path} の読み込みに失敗しました。再学習してください。")
        return models

    # ==========================================
    # データ加工ロジック (特徴量生成)
    # ==========================================
    def process_data(self, df):
        df = df.copy()
        if 'race_id' not in df.columns: return df
        
        # race_idのクリーニング
        df['race_id'] = df['race_id'].astype(str).str.replace('.0', '', regex=False)
        df = df[df['race_id'] != 'nan']
        
        if len(df) == 0: return df

        # --- 数値変換ヘルパー ---
        def time_to_seconds(time_str):
            try:
                if pd.isna(time_str): return None
                s_str = str(time_str)
                if ':' in s_str:
                    m, s = s_str.split(':')
                    return int(m) * 60 + float(s)
                return float(s_str)
            except: return None

        def split_weight(weight_str):
            try:
                s_weight = str(weight_str)
                if pd.isna(s_weight) or '計不' in s_weight: return 470, 0
                w = s_weight.split('(')[0]
                diff = s_weight.split('(')[1].replace(')', '') if '(' in s_weight else 0
                return int(w), int(diff)
            except: return 470, 0

        # --- 基本列の処理 ---
        if '馬体重' in df.columns:
            df['体重'], df['体重増減'] = zip(*df['馬体重'].apply(split_weight))
        else:
            df['体重'], df['体重増減'] = 470, 0

        if 'タイム' in df.columns:
            df['タイム秒'] = df['タイム'].apply(time_to_seconds)
        
        cols = ['単勝', '人 気', '着 順', '枠 番', '馬 番', '斤量']
        for col in cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # ターゲット変数 (着順)
        if '着 順' in df.columns:
            df['is_top3'] = df['着 順'].apply(lambda x: 1 if x <= 3 else 0)
            df['is_win'] = (df['着 順'] == 1).astype(int)
            df['rank_label'] = (18 - df['着 順']).clip(lower=0)

        # --- 特徴量エンジニアリング (Base) ---
        # レース番号抽出
        try:
            df['レース番号'] = df['race_id'].str[-2:].astype(int)
        except:
            df['レース番号'] = 11

        # ソートして過去成績を計算 (馬ごとの時系列)
        df = df.sort_values(['馬名', 'race_id'])

        # --- 1. 過去成績 & トレンド (Trend) ---
        df['馬_前3走平均着順'] = df.groupby('馬名')['着 順'].transform(lambda x: x.shift(1).rolling(3).mean())
        df['馬_同場複勝率'] = df.groupby(['馬名', 'site'])['is_top3'].transform(lambda x: x.shift(1).expanding().mean()).fillna(0)
        df['馬_同馬場複勝率'] = df.groupby(['馬名', 'condition'])['is_top3'].transform(lambda x: x.shift(1).expanding().mean()).fillna(0)
        
        # タイム指数
        if 'タイム秒' in df.columns:
            # そのレースの平均タイムとの差
            race_avg_time = df.groupby('race_id')['タイム秒'].transform('mean')
            df['タイム指数'] = race_avg_time - df['タイム秒'] 
            # 前走の指数を取得
            df['前走タイム指数'] = df.groupby('馬名')['タイム指数'].shift(1)
        
        # 変化量・勢い
        df['斤量変化'] = df.groupby('馬名')['斤量'].diff().fillna(0)
        df['距離変化'] = df.groupby('馬名')['distance'].diff().fillna(0)
        # 距離変化をカテゴリ化 (1:延長, -1:短縮, 0:同距離)
        df['距離変化区分'] = df['距離変化'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        
        # 着順の推移 (前走 - 2走前) マイナスなら良化
        df['2走前着順'] = df.groupby('馬名')['着 順'].shift(2)
        df['着順変動'] = df.groupby('馬名')['着 順'].shift(1) - df['2走前着順']
        
        # クラス・騎手
        df['前走_レース番号'] = df.groupby('馬名')['レース番号'].shift(1)
        df['クラスチェンジ'] = df['レース番号'] - df['前走_レース番号'].fillna(df['レース番号'])
        
        # 騎手の勢い (直近20走)
        df['騎手の勢い'] = df.groupby('騎手')['is_win'].transform(lambda x: x.shift(1).rolling(20).mean()).fillna(0)
        
        # --- 2. 騎手・相性 (Interaction) ---
        # 乗り替わりフラグ
        df['前走騎手'] = df.groupby('馬名')['騎手'].shift(1)
        df['乗り替わり'] = (df['騎手'] != df['前走騎手']).astype(int)
        
        # 騎手×コース相性 (同場連対率)
        df['騎手_本賞_連対率'] = df.groupby(['騎手', 'site'])['is_top3'].transform(lambda x: x.shift(1).expanding().mean()).fillna(0)

        # --- 3. レース内相対評価 (Context) ---
        # ここからはレース単位で比較するためソート順を変更
        df = df.sort_values(['race_id', '馬 番'])
        
        # 前走タイム指数のレース内偏差値化 (平均との差)
        # ※ fillna(0) は初出走などでデータがない場合
        df['タイム指数_レース内差'] = df.groupby('race_id')['前走タイム指数'].transform(lambda x: x - x.mean()).fillna(0)
        
        # 斤量のレース内比較 (周りより重いか軽いか)
        df['斤量_レース内差'] = df.groupby('race_id')['斤量'].transform(lambda x: x - x.mean())

        # 既存の順位系
        df['タイム指数順位'] = df.groupby('race_id')['前走タイム指数'].rank(ascending=False)
        df['前走着順順位'] = df.groupby('race_id')['馬_前3走平均着順'].rank(ascending=True)

        # Label Encoding
        le = LabelEncoder()
        for col in ['site', 'condition', 'track', '騎手']:
            df[col] = df[col].astype(str).fillna('unknown')
            id_col = col + 'ID' if col != 'site' else '場ID'
            if col == 'condition': id_col = '馬場ID'
            if col == 'track': id_col = 'トラックID'
            # 未知のラベル対応のためfit済みのものは使わず毎回変換(簡易的)
            df[id_col] = le.fit_transform(df[col])

        return df

    # ==========================================
    # スクレイピングヘルパー
    # ==========================================
    def get_race_info(self, soup):
        try:
            text = ""
            data_intro = soup.find('div', class_='RaceData01')
            if data_intro: text = data_intro.get_text(strip=True)
            else:
                data_intro_res = soup.find('div', class_='data_intro')
                if data_intro_res: text = data_intro_res.get_text(strip=True)

            if not text: return "芝", 1600, "晴", "良"

            track = "芝" if "芝" in text else "ダート" if "ダ" in text else "障害"
            dist_match = re.search(r"(\d+)m", text)
            distance = int(dist_match.group(1)) if dist_match else 1600
            
            weather = "晴"
            for w in self.weather_list:
                if w in text: weather = w; break
            
            condition = "良"
            for k, v in self.cond_dict.items():
                if k in text: condition = v; break
            
            return track, distance, weather, condition
        except: return "芝", 1600, "晴", "良"

    def fetch_race_data(self, race_id, mode='shutuba'):
        time.sleep(1) # マナーとして1秒待機
        url_base = "https://race.netkeiba.com/race/shutuba.html" if mode == 'shutuba' else "https://race.netkeiba.com/race/result.html"
        url = f"{url_base}?race_id={race_id}"
        try:
            res = requests.get(url, headers=self.headers)
            res.encoding = 'EUC-JP'
            if len(res.text) < 500: return None, None, None
            soup = BeautifulSoup(res.text, 'html.parser')
            track, distance, weather, condition = self.get_race_info(soup)
            
            dfs = pd.read_html(res.text)
            raw_df = None
            for d in dfs:
                # 列名のクリーニング (改行やスペースを削除)
                if isinstance(d.columns, pd.MultiIndex): 
                    d.columns = [str(c[0]).replace('\n','').replace(' ','') for c in d.columns]
                else:
                    d.columns = [str(c).replace('\n','').replace(' ','') for c in d.columns]
                
                # 馬名があるテーブルを探す
                if '馬名' in d.columns: 
                    raw_df = d; break
            
            return raw_df, (track, distance, weather, condition), soup
        except: return None, None, None

    # ==========================================
    # ★ Seleniumを使った強力なデータ取得 (新規追加)
    # ==========================================
    def fetch_data_selenium(self, race_id):
        print("   🌍 Seleniumでブラウザを起動してデータを取得中...")
        
        # ブラウザの設定 (ヘッドレスモード: 画面を表示せずに実行)
        options = Options()
        options.add_argument('--headless') # 画面を出したい場合はこの行をコメントアウト
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        # ユーザーエージェント偽装
        options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

        driver = webdriver.Chrome(options=options)
        wait = WebDriverWait(driver, 20)
        
        url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
        
        try:
            driver.get(url)
            
            # テーブルが表示されるまで待機
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '.ShutubaTable')))
            
            # ページのソースをBeautifulSoupに渡す
            soup = BeautifulSoup(driver.page_source, 'lxml')
            
            # --- ここからご提示のコードのロジック ---
            shutuba_table = []
            
            # ヘッダー取得
            header_tr = soup.select_one('.ShutubaTable > thead > tr')
            if header_tr:
                headers = [th.text.strip().split('\n')[0] for th in header_tr.select('th')]
                # 余計な列があれば調整 (参照コードに合わせて11列目までなど)
                # headers = headers[:15] 
                shutuba_table.append(headers)
            
            # データ行取得
            rows = soup.select('.ShutubaTable > tbody > tr')
            for tr in rows:
                row_data = []
                tds = tr.select('td')
                
                # 馬番、馬名、オッズなどが含まれる主要な列を取得
                for i, td in enumerate(tds):
                    # 印の列などはテキストが特殊な場合があるが、基本はtext.strip()
                    txt = td.text.strip()
                    
                    # オッズ列(人気順などが入る場合があるためクラスで判別も可だが、順番で取る)
                    # 参照コードでは td.select_one('.selectBox') などの処理があったので踏襲
                    if 'selectBox' in str(td):
                        val = td.select_one('.selectBox')
                        txt = val.text.strip() if val else '--'
                    
                    # オッズが "---" の場合や、更新中などの改行を除去
                    txt = txt.split('\n')[0]
                    row_data.append(txt)
                
                shutuba_table.append(row_data)
                
            return shutuba_table, soup

        except Exception as e:
            print(f"   ⚠️ Seleniumエラー: {e}")
            return None, None
        finally:
            driver.quit()

    # ==========================================
    # ★ 予想機能 (Selenium / 列ズレ防止・完全版)
    # ==========================================
    def predict_race(self, race_id):
        if not self.models:
            print("❌ モデルがありません。「3」を選択して学習してください。")
            return

        print(f"\n🚀 レースID {race_id} の予想を開始します (Selenium Mode)")

        # 1. Seleniumでデータを取得
        data_list, soup = self.fetch_data_selenium(race_id)
        
        if not data_list or len(data_list) < 2:
            print("⚠️ データの取得に失敗しました。")
            return

        # 2. DataFrame化 (ヘッダーを信用せず、インデックス位置で強制マッピング)
        # Netkeibaの出馬表の標準的な並び順 (0始まり):
        # 0:枠, 1:馬番, 2:印, 3:馬名, 4:性齢, 5:斤量, 6:騎手, 7:厩舎, 8:体重, 9:オッズ, 10:人気
        
        raw_data = data_list[1:] # ヘッダー除外
        df = pd.DataFrame(raw_data)
        
        # 列数が足りない場合のガード
        if len(df.columns) < 11:
            print(f"⚠️ 列数が不足しています (現在: {len(df.columns)}列)。データの整合性に注意してください。")
            # 足りない列を空文字で埋める
            for i in range(len(df.columns), 11):
                df[i] = ""

        # 強制的に列名を割り当て (列名ズレ防止のため位置指定)
        # 必要な列だけをピンポイントで抜き出す
        df_clean = pd.DataFrame()
        df_clean['枠 番'] = df.iloc[:, 0]
        df_clean['馬 番'] = df.iloc[:, 1]
        df_clean['馬名']   = df.iloc[:, 3]
        df_clean['斤量']   = df.iloc[:, 5]
        df_clean['騎手']   = df.iloc[:, 6]
        df_clean['単勝']   = df.iloc[:, 9] # オッズ
        df_clean['人 気']  = df.iloc[:, 10]
        
        # 体重の処理 (8列目)
        if len(df.columns) > 8:
            df_clean['馬体重'] = df.iloc[:, 8]
        else:
            df_clean['馬体重'] = "470(0)"

        df = df_clean.copy()

        # 3. 数値変換とクリーニング (ここでNaNを撲滅する)
        def safe_float(x):
            try: return float(str(x).replace('---', '0'))
            except: return 0.0

        def safe_int(x):
            try: return int(float(str(x)))
            except: return 0

        # 単勝: "---" や文字化けを 0.0 に
        df['単勝'] = df['単勝'].apply(safe_float)
        
        # 馬番: 数値化できない行は削除 (ここが重要)
        df['馬 番'] = pd.to_numeric(df['馬 番'], errors='coerce')
        df = df.dropna(subset=['馬 番']) # 馬番がない行はゴミデータとして捨てる
        df['馬 番'] = df['馬 番'].astype(int)
        
        # 斤量
        df['斤量'] = df['斤量'].apply(safe_float)
        
        # 人気
        df['人 気'] = df['人 気'].apply(safe_int)

        # 情報取得
        track, distance, weather, condition = self.get_race_info(soup)
        place_code = race_id[4:6]
        site_name = self.place_dict.get(place_code, "東京")
        
        print(f"   📍 {site_name} / {track} {distance}m / {weather} / {condition}")
        
        valid_odds = (df['単勝'] > 0).sum()
        if valid_odds > 0:
            print(f"   ✅ オッズ取得成功: {valid_odds}頭")
        else:
            print("   ⚠️ オッズが取得できませんでした (0.0として処理します)")

        # -------------------------------------------------------
        # 予測プロセス
        # -------------------------------------------------------
        df['site'] = site_name
        df['distance'] = distance
        df['track'] = track
        df['condition'] = condition
        df['weather'] = weather
        df['race_id'] = str(race_id)
        
        # 過去データ結合
        full_df = pd.concat([self.history_df, df], axis=0).drop_duplicates(subset=['race_id', '馬名'], keep='last')
        processed_full = self.process_data(full_df)
        processed_df = processed_full[processed_full['race_id'] == str(race_id)].copy()
        
        if len(processed_df) == 0:
            print("❌ 処理できるデータがありません (馬番の特定に失敗した可能性があります)。")
            return

        preds = []
        try:
            for model in self.models:
                preds.append(model.predict(processed_df[self.features]))
        except Exception as e:
            print(f"❌ 予測エラー: {e}")
            return
        
        processed_df['score'] = np.mean(preds, axis=0)
        std = processed_df['score'].std()
        processed_df['score_z'] = (processed_df['score'] - processed_df['score'].mean()) / std if std != 0 else 0
        
        results = processed_df.copy().sort_values('score_z', ascending=False)
        
        # 出力
        buy_list = []
        full_list = []
        
        for i, (_, row) in enumerate(results.iterrows()):
            score = row['score_z']
            odds_val = float(row['単勝'])
            
            status = ""
            is_buy = False
            
            if score > 1.5:
                if odds_val == 0: status = "⏳ 待機"
                elif odds_val >= 30.0: status = "🧪 爆穴"
                elif odds_val >= 5.0: status = "🔥🔥 妙味"
                else: status = "⚠️ 本命"
                if odds_val > 0: is_buy = True
            elif score > 0.8:
                status = "△ 抑え"

            odds_str = f"{odds_val:.1f}" if odds_val > 0 else "--"
            
            # 【修正】ここでも念のため安全に変換
            try:
                umaban = int(row['馬 番'])
            except:
                umaban = 0

            row_data = [i+1, umaban, row['馬名'], f"{score:.2f}", odds_str, status]
            full_list.append(row_data)
            if is_buy: buy_list.append(row_data)

        headers = ["Rank", "No.", "Name", "AI(Z)", "Odds", "Status"]
        
        print("\n" + "="*55)
        print(" 🎯 推奨馬 (Buying Target)")
        print("="*55)
        if buy_list:
            print(tabulate(buy_list, headers=headers, tablefmt="fancy_grid"))
        else:
            print("   (推奨馬なし)")

        print("\n" + "-"*55)
        print(" 📋 全頭分析")
        print("-"*55)
        print(tabulate(full_list, headers=headers, tablefmt="simple"))

        # ★ 追加: 予想結果をCSVに保存
        # results という変数がソート済みの結果DataFrameとして存在している箇所を利用
        if 'results' in locals():
            # ステータス列を追加してから保存
            # (predict_race内のループで判定したstatusをDataFrameに戻すのは手間なので
            #  簡易的にここで再計算するか、ループ内でリスト化したものをDataFrame化するのが安全ですが
            #  ここではresults DataFrameに最低限の情報を付与して渡します)
            
            # 簡易実装: results DFには status列がないので、ここで計算して付与
            def get_status(row):
                s = row['score_z']
                o = float(row['単勝'])
                if s > 1.5:
                    if o >= 30: return "爆穴"
                    elif o >= 5: return "妙味"
                    return "本命"
                elif s > 0.8: return "抑え"
                return "-"
                
            results['status'] = results.apply(get_status, axis=1)
            self.save_prediction_log(race_id, results)

            # ★ 追加: HTMLレポートも出力する
            self.save_html_report(race_id, results)


        
    # ==========================================
    # データ収集
    # ==========================================
    def add_result(self, race_id):
        print(f"\n📥 結果データを取得中... (ID: {race_id})")
        time.sleep(1)
        url = f"https://db.netkeiba.com/race/{race_id}"
        
        try:
            res = requests.get(url, headers=self.headers)
            res.encoding = 'EUC-JP'
            soup = BeautifulSoup(res.text, 'html.parser')
            
            track, distance, weather, condition = self.get_race_info(soup)
            place_name = self.place_dict.get(race_id[4:6], "その他")
            
            dfs = pd.read_html(res.text)
            if not dfs: return
            
            df = None
            for d in dfs:
                if isinstance(d.columns, pd.MultiIndex): d.columns = [str(c[0]) for c in d.columns]
                d.columns = [str(c).replace(' ', '') for c in d.columns]
                if '着順' in d.columns: df = d; break
            
            if df is None: return

            df['race_id'] = str(race_id)
            df['site'] = place_name
            df['track'] = track
            df['distance'] = distance
            df['weather'] = weather
            df['condition'] = condition
            
            rename_map = {'着順': '着 順', '枠番': '枠 番', '馬番': '馬 番', '人気': '人 気'}
            for k, v in rename_map.items():
                if k in df.columns: df = df.rename(columns={k: v})

            # 重複チェック
            if str(race_id) in self.history_df['race_id'].astype(str).unique():
                print("⚠️ このレースは既に登録されています。")
                return

            # CSV追記
            df.to_csv(self.csv_path, mode='a', header=not os.path.exists(self.csv_path), index=False, encoding="utf-8-sig")
            print("💾 保存しました。")
            
            # メモリ上のデータも更新
            self.history_df = pd.concat([self.history_df, df], axis=0)
            
            # 自動再学習するか確認
            if input("🔄 モデルを再学習しますか？ (y/n): ").lower() == 'y':
                self.retrain_models()

        except Exception as e:
            print(f"❌ エラー: {e}")

    # ==========================================
    # ★ 厳密な再学習 (時系列分割 & Early Stopping)
    # ==========================================
    def retrain_models(self):
        print("\n🔄 学習プロセスを開始します (Tuning -> Eval -> Finalize)...")
        
        # 1. データ準備
        if os.path.exists(self.csv_path):
            self.history_df = pd.read_csv(self.csv_path, dtype={'race_id': str}).dropna(subset=['race_id'])
        
        full_df = self.process_data(self.history_df).sort_values('race_id')
        
        if len(full_df) < 50:
            print("⚠️ データが少なすぎます。最低でも5レース分ほど集めてください。")
            return

        # データを分割 (チューニング & 検証用)
        unique_ids = full_df['race_id'].unique()
        split_idx = int(len(unique_ids) * 0.8)
        
        train_ids = unique_ids[:split_idx]
        test_ids = unique_ids[split_idx:]
        
        df_train = full_df[full_df['race_id'].isin(train_ids)]
        df_test = full_df[full_df['race_id'].isin(test_ids)].copy()
        
        q_train = df_train.groupby('race_id').size().to_list()
        q_test = df_test.groupby('race_id').size().to_list()
        
        dtrain = lgb.Dataset(df_train[self.features], label=df_train['rank_label'], group=q_train)
        dtest = lgb.Dataset(df_test[self.features], label=df_test['rank_label'], group=q_test, reference=dtrain)

        # ==========================================
        # Phase 0: パラメータ自動チューニング (Optuna)
        # ==========================================
        print("\n[Phase 0] 現在のデータに最適なパラメータを探索中 (Optuna)...")
        
        def objective(trial):
            params = {
                'objective': 'lambdarank',
                'metric': 'ndcg',
                'ndcg_eval_at': [1, 3],
                'verbosity': -1,
                'boosting_type': 'gbdt',
                'random_state': 42,
                'feature_pre_filter': False, 
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'max_depth': trial.suggest_int('max_depth', 5, 12),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            }
            
            # 高速化のため、ここでの学習回数は少なめに
            model = lgb.train(
                params, dtrain, 
                valid_sets=[dtest], 
                valid_names=['eval'],
                num_boost_round=1000,
                callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
            )
            return model.best_score['eval']['ndcg@1']

        # 探索実行 (20回試行)
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=20) 
        
        print(f"   ✨ 最適パラメータ発見: NDCG@1 = {study.best_value:.4f}")
        # クラスのパラメータを更新
        self.best_params.update(study.best_params)
        self.best_params['feature_pre_filter'] = False

        # ==========================================
        # Phase 1: 性能評価 & 最適回数決定
        # ==========================================
        print("\n[Phase 1] 新パラメータで直近20%をテストします...")
        
        seeds = [42, 100, 2024]
        val_models = []
        best_iterations = [] 
        
        for seed in seeds:
            params = self.best_params.copy()
            params['random_state'] = seed
            
            model = lgb.train(
                params, dtrain, 
                valid_sets=[dtest], 
                valid_names=['eval'],
                num_boost_round=5000,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(period=0)
                ]
            )
            val_models.append(model)
            best_iterations.append(model.best_iteration)
        
        # 評価指標の表示
        preds = []
        for model in val_models:
            preds.append(model.predict(df_test[self.features]))
        df_test['score'] = np.mean(preds, axis=0)
        
        results = []
        for rid, group in df_test.groupby('race_id'):
            ai_top1 = group.sort_values('score', ascending=False).iloc[0]
            is_win = 1 if ai_top1['着 順'] == 1 else 0
            is_top3 = 1 if ai_top1['着 順'] <= 3 else 0
            try: odds = float(ai_top1['単勝'])
            except: odds = 0
            return_val = odds * 100 if is_win else 0
            results.append({'win': is_win, 'top3': is_top3, 'return': return_val})
            
        total = len(results)
        wins = sum(r['win'] for r in results)
        top3 = sum(r['top3'] for r in results)
        ret_money = sum(r['return'] for r in results)
        bet_money = total * 100
        
        print("\n" + "="*50)
        print(f" 📊 検証結果 (未学習データ {total} レース)")
        print("="*50)
        print(f" 🎯 1着 的中率 : {(wins/total)*100:.1f}%")
        print(f" 🥉 3着内率    : {(top3/total)*100:.1f}%")
        print(f" 💰 単勝回収率 : {(ret_money/bet_money)*100:.1f}%")
        print("="*50)
        
        # ==========================================
        # Phase 2: 全データでの本番学習
        # ==========================================
        print("\n[Phase 2] 最適化された設定で全データを学習し、モデルを保存します...")
        
        # 少し多めに回す
        optimal_round = int(np.mean(best_iterations) * 1.1)
        print(f"   👉 設定ブースティング回数: {optimal_round}回")
        
        # 全データセット
        q_full = full_df.groupby('race_id').size().to_list()
        dtrain_full = lgb.Dataset(full_df[self.features], label=full_df['rank_label'], group=q_full)
        
        self.models = [] 
        
        for seed in seeds:
            print(f"   🌱 Seed {seed} Final Training...")
            params = self.best_params.copy()
            params['random_state'] = seed
            
            model = lgb.train(
                params, dtrain_full,
                num_boost_round=optimal_round,
                callbacks=[lgb.log_evaluation(period=0)]
            )
            joblib.dump(model, f'production_model_2026_seed_{seed}.pkl')
            self.models.append(model)
            
        print("✨ 完全再学習が完了しました。")

    # ==========================================
    # ★ 精度検証 (Evaluation)
    # ==========================================
    def evaluate_performance(self):
        print("\n📊 モデル精度と回収率の検証を開始します...")
        
        df = self.process_data(self.history_df).sort_values('race_id')
        unique_ids = df['race_id'].unique()
        
        if len(unique_ids) < 10:
            print("⚠️ 検証するにはデータがもっと必要です。")
            return

        test_size = int(len(unique_ids) * 0.2)
        test_ids = unique_ids[-test_size:]
        test_df = df[df['race_id'].isin(test_ids)].copy()
        
        preds = []
        for model in self.models:
            preds.append(model.predict(test_df[self.features]))
        test_df['score'] = np.mean(preds, axis=0)

        results = []
        for rid, group in test_df.groupby('race_id'):
            ai_top1 = group.sort_values('score', ascending=False).iloc[0]
            
            is_win = 1 if ai_top1['着 順'] == 1 else 0
            is_top3 = 1 if ai_top1['着 順'] <= 3 else 0
            
            try: odds = float(ai_top1['単勝'])
            except: odds = 0
            
            return_val = odds * 100 if is_win else 0
            results.append({'win': is_win, 'top3': is_top3, 'return': return_val})

        total = len(results)
        wins = sum(r['win'] for r in results)
        top3 = sum(r['top3'] for r in results)
        ret_money = sum(r['return'] for r in results)
        bet_money = total * 100
        
        print("-" * 40)
        print(f" 🧪 検証結果 (最新 {total} レース)")
        print("-" * 40)
        print(f" 🎯 1着 的中率 : {(wins/total)*100:.1f}% ({wins}/{total})")
        print(f" 🥉 3着内率    : {(top3/total)*100:.1f}% ({top3}/{total})")
        print(f" 💰 単勝回収率 : {(ret_money/bet_money)*100:.1f}% (投:{bet_money} -> 回:{int(ret_money)})")
        print("-" * 40)

    # ==========================================
    # ★ モデル監査 (落とし穴チェック)
    # ==========================================
    def audit_model(self):
        print("\n🕵️‍♀️ モデルの健全性を監査します (Audit)...")
        
        # 1. データの準備 (検証用データ)
        df = self.process_data(self.history_df).sort_values('race_id')
        unique_ids = df['race_id'].unique()
        test_size = int(len(unique_ids) * 0.2)
        test_ids = unique_ids[-test_size:]
        test_df = df[df['race_id'].isin(test_ids)].copy()
        
        # 予測
        preds = []
        for model in self.models:
            preds.append(model.predict(test_df[self.features]))
        test_df['score'] = np.mean(preds, axis=0)

        # 集計
        results = []
        for rid, group in test_df.groupby('race_id'):
            ai_top1 = group.sort_values('score', ascending=False).iloc[0]
            is_win = 1 if ai_top1['着 順'] == 1 else 0
            try: odds = float(ai_top1['単勝'])
            except: odds = 0.0
            
            results.append({
                'race_id': rid, 
                'win': is_win, 
                'odds': odds, 
                'return': odds * 100 if is_win else 0,
                'horse': ai_top1['馬名']
            })

        total_bet = len(results) * 100
        total_return = sum(r['return'] for r in results)
        base_recovery = (total_return / total_bet) * 100

        # --- ① マグレ当たり排除テスト ---
        # 当たったレースを配当が高い順にソート
        hits = sorted([r for r in results if r['win']], key=lambda x: x['odds'], reverse=True)
        
        print(f"\n1️⃣ 高配当への依存度チェック")
        print(f"   現在の回収率: {base_recovery:.1f}%")
        
        if hits:
            # Top 3の高配当を表示
            print("   [内訳] 配当が高かった的中 Top 3:")
            for i, h in enumerate(hits[:3]):
                print(f"     {i+1}. {h['horse']} (単勝 {h['odds']}倍)")
            
            # Top 3を除外した場合の回収率
            top3_return_sum = sum(h['return'] for h in hits[:3])
            audit_return = total_return - top3_return_sum
            audit_recovery = (audit_return / total_bet) * 100
            
            print(f"   👉 もし上位3本が外れていたら -> 回収率: {audit_recovery:.1f}%")
            if audit_recovery < 100:
                print("      ⚠️ 警告: 少数の大穴に依存しています。運の要素が強いです。")
            else:
                print("      ✅ 合格: ラッキーパンチ抜きでも勝てています。")
        else:
            print("   (的中なし)")

        # --- ② オッズ低下 (スリッページ) テスト ---
        print(f"\n2️⃣ オッズ不利シミュレーション")
        # 全ての払い戻しを 0.9倍 (10%減) して計算
        penalty_return = total_return * 0.9
        penalty_recovery = (penalty_return / total_bet) * 100
        print(f"   👉 常にオッズが10%低かったら -> 回収率: {penalty_recovery:.1f}%")
        
        if penalty_recovery > 100:
            print("      ✅ 合格: オッズ低下の誤差を含めてもプラスです。")
        else:
            print("      ⚠️ 注意: オッズが少し下がるだけでマイナス転落の危険があります。")

        # --- ③ リーク (カンニング) チェック ---
        print(f"\n3️⃣ データリーク(不正解)の確認")
        # 特徴量重要度を表示
        if self.models:
            importance = pd.DataFrame()
            importance['feature'] = self.features
            importance['gain'] = self.models[0].feature_importance(importance_type='gain')
            importance = importance.sort_values('gain', ascending=False).head(5)
            
            print("   [重要度 Top 5 特徴量]")
            for _, row in importance.iterrows():
                print(f"     - {row['feature']}")
            
            print("   👀 チェックポイント:")
            print("      ・ここに「当日の着順」「当日のタイム」などが含まれていませんか？")
            print("      ・「前走〜」や「オッズ(確定前)」ならOKです。")

    # ==========================================
    # ★ 追加機能: 同日一括処理 (Batch Processing)
    # ==========================================
    def process_day_all(self, race_id, mode='predict'):
        """
        指定されたrace_idの日付・場所情報を使い、1Rから12Rまでを一括処理する
        mode: 'predict' (予想) or 'result' (結果追加)
        """
        # IDのバリデーション (簡易)
        sid = str(race_id)
        if len(sid) != 12:
            print("⚠️ エラー: レースIDは12桁で入力してください (例: 202406010101)")
            return

        # 末尾2桁(レース番号)を除外してベースIDを作成
        base_id = sid[:-2]
        
        print(f"\n🔄 開催日一括処理を開始します... (Base ID: {base_id}**)")
        print(f"   モード: {'全レース予想' if mode == 'predict' else '全レース結果収集'}")

        # 1R〜12Rまでループ
        for i in range(1, 13):
            current_race_num = f"{i:02}"
            target_id = base_id + current_race_num
            
            print(f"\n{'='*60}")
            print(f" 🏇 {current_race_num}R (ID: {target_id}) の処理中...")
            print(f"{'='*60}")
            
            if mode == 'predict':
                self.predict_race(target_id)
            elif mode == 'result':
                self.add_result(target_id)
            
            # サーバー負荷軽減のため、少し長めに待機
            print("   💤 待機中 (Access Interval)...")
            time.sleep(2)
        
        print(f"\n✅ 全レースの処理が完了しました。")

    # ==========================================
    # ★ 新規追加: 予想ログの保存
    # ==========================================
    def save_prediction_log(self, race_id, df_results):
        import datetime
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # ログに残すデータを作成
        log_data = df_results.copy()
        log_data['timestamp'] = now
        log_data['race_id'] = str(race_id)
        
        # 必要な列だけに絞る
        cols = ['timestamp', 'race_id', '馬 番', '馬名', 'score_z', '単勝', 'status']
        # 存在しない列があれば埋める（エラー回避）
        for c in cols:
            if c not in log_data.columns: log_data[c] = 0
            
        save_df = log_data[cols].rename(columns={
            '馬 番': 'umaban',
            '馬名': 'horse_name', 
            '単勝': 'predicted_odds',
            'status': 'ai_status'
        })
        
        # 結果記入用の空列を追加
        save_df['actual_rank'] = np.nan   # 実際の着順
        save_df['actual_odds'] = np.nan   # 確定オッズ
        save_df['return_amount'] = 0      # 払戻金
        
        # 既存ログがあれば読み込んで、今回のレースIDの分を一度消す（上書きのため）
        if os.path.exists(self.log_path):
            existing_log = pd.read_csv(self.log_path, dtype={'race_id': str})
            # 同じレースIDの記録があれば削除して、新しい予想に入れ替える
            existing_log = existing_log[existing_log['race_id'] != str(race_id)]
            final_df = pd.concat([existing_log, save_df], axis=0)
        else:
            final_df = save_df

        final_df.to_csv(self.log_path, index=False, encoding='utf-8-sig')
        print(f"   📝 予想ログを保存しました: {self.log_path}")

    # ==========================================
    # ★ 決定版: 予想結果の答え合わせ (オッズ列名 自動補正機能付き)
    # ==========================================
    def settle_predictions(self, race_id):
        if not os.path.exists(self.log_path):
            print("⚠️ 予想ログファイルがありません。先に予想を行ってください。")
            return

        print(f"\n💰 レース結果を照合中... (ID: {race_id})")
        
        # 1. ネット競馬の「速報結果」ページを取得
        url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
        
        try:
            res = requests.get(url, headers=self.headers)
            res.encoding = 'EUC-JP'
            
            dfs = pd.read_html(res.content)
            if not dfs:
                print("❌ 結果データのテーブルが見つかりません。")
                return
            
            # 結果テーブルを探す
            result_df = None
            for d in dfs:
                # 列名のクリーニング（改行やスペースを除去）
                cols = [str(c).replace('\n', '').replace(' ', '') for c in d.columns]
                d.columns = cols
                
                # ★ 強力な補正: 「単勝オッズ」や「単勝」が含まれる列を探して「単勝」に統一
                odds_col = next((c for c in cols if '単勝' in c), None)
                if odds_col:
                    d = d.rename(columns={odds_col: '単勝'})

                # 「着順」と「馬番」があるテーブルを採用
                if '着順' in d.columns and '馬番' in d.columns:
                    result_df = d
                    break
            
            if result_df is None:
                print("⏳ まだレース結果が確定していないか、ページ構造が異なります。")
                return

            # --- データ型変換 ---
            # 馬番
            result_df['馬番'] = pd.to_numeric(result_df['馬番'], errors='coerce')
            result_df = result_df.dropna(subset=['馬番'])
            result_df['馬番'] = result_df['馬番'].astype(int)
            
            # 着順
            def parse_rank(x):
                try: return int(x)
                except: return 99
            
            result_df['着順_num'] = result_df['着順'].apply(parse_rank)
            
            # 単勝オッズ（数値化）
            if '単勝' in result_df.columns:
                result_df['単勝'] = pd.to_numeric(result_df['単勝'], errors='coerce').fillna(0.0)
            else:
                result_df['単勝'] = 0.0

            # 2. ログファイルを読み込み
            log_df = pd.read_csv(self.log_path, dtype={'race_id': str})
            
            target_mask = log_df['race_id'] == str(race_id)
            if not target_mask.any():
                print("⚠️ このレースの予想ログが見つかりません。")
                return

            # 3. マージして結果を更新
            buy_keywords = ['本命', '妙味', '爆穴', '🔥🔥', '⚠️', '🧪']
            count_updated = 0
            
            for idx, row in log_df[target_mask].iterrows():
                try:
                    umaban = int(row['umaban'])
                    match = result_df[result_df['馬番'] == umaban]
                    
                    if len(match) > 0:
                        top_row = match.iloc[0]
                        actual_rank = int(top_row['着順_num'])
                        final_odds = float(top_row['単勝'])
                        
                        # ログ更新
                        log_df.at[idx, 'actual_rank'] = actual_rank
                        log_df.at[idx, 'actual_odds'] = final_odds
                        
                        # 払い戻し計算 (購入対象ステータスのみ)
                        status = str(row['ai_status'])
                        if any(k in status for k in buy_keywords):
                            if actual_rank == 1:
                                log_df.at[idx, 'return_amount'] = int(final_odds * 100)
                            else:
                                log_df.at[idx, 'return_amount'] = 0
                        else:
                             log_df.at[idx, 'return_amount'] = 0
                        
                        count_updated += 1
                except:
                    continue

            # 保存
            log_df.to_csv(self.log_path, index=False, encoding='utf-8-sig')
            
            # --- 今回の結果レポート ---
            race_log = log_df[target_mask]
            buy_mask = race_log['ai_status'].astype(str).apply(lambda x: any(k in x for k in buy_keywords))
            
            this_bet = len(race_log[buy_mask]) * 100
            this_return = race_log['return_amount'].sum()
            this_balance = this_return - this_bet
            
            winner_row = result_df[result_df['着順_num']==1]
            winner_name = winner_row['馬名'].values[0] if len(winner_row)>0 else '不明'
            winner_odds = winner_row['単勝'].values[0] if len(winner_row)>0 else 0.0
            
            print("-" * 30)
            print(f" 🏁 {race_id} 結果更新 ({count_updated}頭)")
            print(f" 🥇 1着: {winner_name} (単勝 {winner_odds}倍)")  # ★ここにオッズが出るはずです
            print(f" 🎫 投資: {this_bet}円 -> 回収: {int(this_return)}円")
            print(f" 📊 収支: {'+' if this_balance >= 0 else ''}{int(this_balance)}円")
            
            # --- 全期間トータル ---
            total_log = log_df.dropna(subset=['actual_rank'])
            total_buy_mask = total_log['ai_status'].astype(str).apply(lambda x: any(k in x for k in buy_keywords))
            total_bet_calc = len(total_log[total_buy_mask]) * 100
            total_return_calc = total_log[total_buy_mask]['return_amount'].sum()
            total_balance_calc = total_return_calc - total_bet_calc
            
            print("=" * 30)
            print(f" 💰 全期間トータル収支: {'+' if total_balance_calc >= 0 else ''}{int(total_balance_calc):,} 円")
            print("=" * 30)
                
        except Exception as e:
            print(f"❌ エラーが発生しました: {e}")
            import traceback
            traceback.print_exc()

    # ==========================================
    # ★ 修正版: 検索機能付きHTMLレポート出力
    # ==========================================
    def save_html_report(self, race_id, df):
        import datetime
        today_str = datetime.datetime.now().strftime('%Y%m%d')
        filename = f"report_{today_str}.html"
        
        now_str = datetime.datetime.now().strftime('%H:%M:%S')
        
        # HTMLヘッダー (CSS + JS)
        html_header = """
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>AI Keiba Report</title>
            <style>
                body { font-family: "Helvetica Neue", Arial, sans-serif; background-color: #f4f4f9; color: #333; padding: 20px; padding-top: 80px; }
                
                /* 検索バーのスタイル (上部固定) */
                .search-container {
                    position: fixed; top: 0; left: 0; width: 100%;
                    background: #343a40; padding: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.2);
                    z-index: 1000; display: flex; align-items: center; justify-content: center;
                }
                .search-box {
                    width: 300px; padding: 10px; border-radius: 5px; border: none; font-size: 16px;
                    outline: none;
                }
                .search-label { color: white; margin-right: 10px; font-weight: bold; }
                
                /* カードスタイル */
                .race-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin-bottom: 30px; animation: fadeIn 0.5s; }
                h2 { border-bottom: 2px solid #007bff; padding-bottom: 10px; color: #0056b3; margin-top: 0; }
                .meta { color: #666; font-size: 0.9em; margin-bottom: 15px; }
                
                /* テーブル */
                table { width: 100%; border-collapse: collapse; margin-top: 10px; }
                th { background-color: #007bff; color: white; padding: 10px; text-align: left; }
                td { padding: 10px; border-bottom: 1px solid #ddd; }
                
                /* ステータス色 */
                .status-fire { background-color: #ffebee; color: #c62828; font-weight: bold; border-left: 5px solid #d32f2f; }
                .status-warn { background-color: #fff3e0; color: #ef6c00; font-weight: bold; border-left: 5px solid #f57c00; }
                .status-safe { background-color: #e3f2fd; color: #1565c0; border-left: 5px solid #1976d2; }
                .status-wait { color: #999; }
                
                @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
            </style>
            <script>
                // 検索フィルタ機能
                function filterRaces() {
                    var input = document.getElementById('raceInput').value.toLowerCase();
                    var cards = document.getElementsByClassName('race-card');
                    
                    for (var i = 0; i < cards.length; i++) {
                        var raceId = cards[i].getAttribute('data-race-id');
                        var text = cards[i].innerText.toLowerCase();
                        
                        // IDが一致するか、テキスト(11Rなど)が含まれていれば表示
                        if (raceId.includes(input) || text.includes(input)) {
                            cards[i].style.display = "";
                        } else {
                            cards[i].style.display = "none";
                        }
                    }
                }
            </script>
        </head>
        <body>
        
        <div class="search-container">
            <span class="search-label">🔍 Race Search:</span>
            <input type="text" id="raceInput" class="search-box" onkeyup="filterRaces()" placeholder="ID or Race No (e.g. 11, 2024...)">
        </div>
        
        <div style="text-align:center; margin-bottom:20px; color:#666;">
            <h1>🏇 AI Keiba Daily Report</h1>
        </div>
        """
        
        # ファイルの読み書きモード設定
        mode = 'a' if os.path.exists(filename) else 'w'
        
        with open(filename, mode, encoding='utf-8') as f:
            if mode == 'w':
                f.write(html_header)
            
            # --- 重要な変更点: data-race-id 属性を追加 ---
            f.write(f'<div class="race-card" data-race-id="{race_id}">')
            
            # レース番号の抽出（IDの末尾2桁）
            try: race_num = f"{int(race_id[-2:]):02}R"
            except: race_num = "Unknown R"

            f.write(f'<h2>📍 {race_num} (ID: {race_id}) <span style="font-size:0.6em; float:right;">{now_str}</span></h2>')
            
            place = df['site'].iloc[0] if 'site' in df.columns else '-'
            cond = f"{df['distance'].iloc[0]}m / {df['weather'].iloc[0]}" if 'distance' in df.columns else ''
            f.write(f'<div class="meta">🏟️ {place} | 🌤️ {cond}</div>')
            
            f.write('<table>')
            f.write('<thead><tr><th>Rank</th><th>No.</th><th>馬名</th><th>AI指数</th><th>Odds</th><th>判定</th></tr></thead>')
            f.write('<tbody>')
            
            for i, row in df.iterrows():
                score = row['score_z']
                odds = float(row['単勝'])
                
                row_class = ""
                status_text = "-"
                
                if score > 1.5:
                    if odds >= 30.0: 
                        row_class = "status-fire"; status_text = "🧪 爆穴"
                    elif odds >= 5.0: 
                        row_class = "status-fire"; status_text = "🔥🔥 妙味"
                    elif odds == 0:
                        row_class = "status-wait"; status_text = "⏳ オッズ待"
                    else: 
                        row_class = "status-warn"; status_text = "⚠️ 本命"
                elif score > 0.8:
                    row_class = "status-safe"; status_text = "△ 抑え"
                
                f.write(f'<tr class="{row_class}">')
                f.write(f'<td>{i+1}</td>')
                f.write(f'<td>{row["馬 番"]}</td>')
                f.write(f'<td><b>{row["馬名"]}</b></td>')
                f.write(f'<td>{score:.2f}</td>')
                f.write(f'<td>{odds}</td>')
                f.write(f'<td>{status_text}</td>')
                f.write('</tr>')
                
            f.write('</tbody></table></div>')
            
        print(f"   ✨ レポート出力: {filename}")
    # ==========================================
    # ★ 修正版: 収支分析とグラフ化 (テキストレポート付き)
    # ==========================================
    def analyze_log(self):
        import matplotlib.pyplot as plt
        
        if not os.path.exists(self.log_path):
            print("⚠️ ログファイルがありません。先に予想と答え合わせを行ってください。")
            return

        # データを読み込み
        df = pd.read_csv(self.log_path)
        
        # 結果が確定している（着順が入っている）データのみ抽出
        df = df.dropna(subset=['actual_rank'])
        
        if len(df) == 0:
            print("⚠️ 結果が確定したデータがありません。「3. 答え合わせ」を行ってください。")
            return

        print(f"\n📊 データを分析中... (全データ数: {len(df)} 件)")

        # --- 収支計算ロジック ---
        # 購入対象となるステータスキーワード
        buy_keywords = ['本命', '妙味', '爆穴', '🔥🔥', '⚠️', '🧪']
        
        # 購入フラグを立てる
        # ai_status列にキーワードが含まれていれば購入対象(100円)、そうでなければ0円
        df['is_buy'] = df['ai_status'].astype(str).apply(lambda x: any(k in x for k in buy_keywords))
        df['bet_amount'] = df['is_buy'].apply(lambda x: 100 if x else 0)
        
        # 払戻金（結果がない場合は0）
        df['return_amount'] = df['return_amount'].fillna(0)
        
        # 購入した行だけを抽出
        bet_df = df[df['is_buy']].copy()
        
        if len(bet_df) == 0:
            print("⚠️ 推奨馬（購入対象）の記録がまだありません。")
            return

        # 集計
        total_bet = bet_df['bet_amount'].sum()
        total_return = bet_df['return_amount'].sum()
        total_balance = total_return - total_bet
        recovery_rate = (total_return / total_bet * 100) if total_bet > 0 else 0
        
        win_count = len(bet_df[bet_df['actual_rank'] == 1])
        total_races = len(bet_df)
        win_rate = (win_count / total_races * 100) if total_races > 0 else 0

        # --- コンソールへの収支レポート出力 ---
        print("\n" + "="*40)
        print(" 💰 推奨馬購入シミュレーション (単勝100円)")
        print("="*40)
        print(f" 🎫 購入総数   : {total_races} 頭")
        print(f" 🎯 的中数     : {win_count} 頭 (的中率 {win_rate:.1f}%)")
        print("-" * 40)
        print(f" 💴 総投資額   : {int(total_bet):,} 円")
        print(f" 💴 総払戻額   : {int(total_return):,} 円")
        print(f" 📈 回収率     : {recovery_rate:.1f} %")
        print("-" * 40)
        if total_balance >= 0:
            print(f" 💹 最終収支   : +{int(total_balance):,} 円 🔵")
        else:
            print(f" 📉 最終収支   : {int(total_balance):,} 円 🔴")
        print("="*40)

        # --- 以下、グラフ作成（既存機能の維持・強化） ---
        try:
            # 時系列収支
            bet_df['profit'] = bet_df['return_amount'] - bet_df['bet_amount']
            bet_df['balance_history'] = bet_df['profit'].cumsum()
            
            plt.figure(figsize=(10, 5))
            plt.plot(range(len(bet_df)), bet_df['balance_history'], marker='o', linestyle='-', color='blue')
            plt.axhline(y=0, color='r', linestyle='--')
            plt.title(f'Balance History (Total: {int(total_balance)} Yen)')
            plt.xlabel('Number of Bets')
            plt.ylabel('Balance (Yen)')
            plt.grid(True)
            plt.savefig('analysis_balance.png')
            plt.close()
            print("   📸 収支推移グラフを保存しました: analysis_balance.png")
            
            # ステータス別分析
            def get_simple_status(txt):
                if '爆穴' in txt or '🧪' in txt: return '穴(Risk)'
                if '妙味' in txt or '🔥🔥' in txt: return '妙(Value)'
                if '本命' in txt or '⚠️' in txt: return '本(Solid)'
                return 'Other'
            
            bet_df['type'] = bet_df['ai_status'].apply(get_simple_status)
            status_summary = bet_df.groupby('type').agg({
                'bet_amount': 'count',
                'return_amount': 'sum'
            })
            status_summary['recov'] = (status_summary['return_amount'] / (status_summary['bet_amount'] * 100)) * 100
            
            print("\n📋 タイプ別成績:")
            print(tabulate(status_summary[['bet_amount', 'recov']], headers=['Type', 'Count', 'Recov%'], tablefmt='simple'))

        except Exception as e:
            print(f"⚠️ グラフ作成中にエラー: {e}")


if __name__ == "__main__":
    system = KeibaSystem()
    while True:
        print("\n=== 🏇 AI Keiba System 2026 (Pro) ===")
        print("1. 予想 (ID入力 -> 単発/全レース選択可)")
        print("2. 結果追加 (ID入力 -> 単発/全レース選択可)")
        print("3. ★ 予想の答え合わせ (収支記録)")
        print("4. ★ 収支・分析グラフ出力") # 追加
        print("5. 強制再学習 (Rigorous)")
        print("6. 精度検証 (Eval)")
        print("7. 落とし穴チェック (Audit)")
        print("8. 終了")
        
        c = input("選択: ")
        
        if c == '1':
            rid = input("Race ID (代表): ")
            sub_c = input("   [1] このレースのみ予想  [2] この日の全12レースを予想 : ")
            if sub_c == '2':
                system.process_day_all(rid, mode='predict')
            else:
                system.predict_race(rid)

        elif c == '2':
            rid = input("Race ID (代表): ")
            sub_c = input("   [1] このレースのみ追加  [2] この日の全12レースを追加 : ")
            if sub_c == '2':
                system.process_day_all(rid, mode='result')
            else:
                system.add_result(rid)
        
        elif c == '3':
            # ★ 新規追加
            rid = input("Race ID (答え合わせしたいレース): ")
            sub_c = input("   [1] 単発  [2] この日の全レースを一括処理 : ")
            if sub_c == '2':
                # process_day_all に settle モードを追加するか、ループで回す
                base_id = rid[:-2]
                for i in range(1, 13):
                    target = base_id + f"{i:02}"
                    system.settle_predictions(target)
                    time.sleep(1)
            else:
                system.settle_predictions(rid)
        elif c == '4':
            system.analyze_log() # 呼び出し

        elif c == '5': system.retrain_models()
        elif c == '6': system.evaluate_performance()
        elif c == '7': system.audit_model()
        elif c == '8': break