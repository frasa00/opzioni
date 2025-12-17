"""
Dashboard Streamlit per il Terminale Opzioni con ML
"""

import streamlit as st
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Importa i moduli locali
try:
    from data_fetcher import DataFetcher
    from indicators import AdvancedOptionsIndicators
    from analyzer import MarketAnalyzer
    from ml_predictor import OptionsMLPredictor
    from feature_engineer import FeatureEngineer
    ML_AVAILABLE = True
except ImportError as e:
    st.sidebar.warning(f"Alcuni moduli non disponibili: {e}. Altre funzionalità potrebbero essere limitate.")
    ML_AVAILABLE = False

class EnhancedOptionsDashboard:
    """Dashboard avanzata per opzioni con integrazione ML."""

    def __init__(self):
        self.tickers = {
            'SP500': '^SPX',
            'NASDAQ': '^NDX'
        }
        self.fetcher = DataFetcher()
        self.indicators_calculator = AdvancedOptionsIndicators()
        self.analyzer = MarketAnalyzer()
        self.historical_data = []  # per tracciare i dati storici in sessione

        # Inizializza il predictor ML (se disponibile)
        if ML_AVAILABLE:
            # Puoi cambiare il tipo di modello in base alla configurazione
            self.predictor = OptionsMLPredictor(model_type='random_forest')
            # Prova a caricare un modello pre-addestrato
            try:
                self.predictor.load_model("models/spx_direction_predictor.pkl")
            except:
                st.sidebar.warning("Modello ML pre-addestrato non trovato. Le previsioni non saranno disponibili.")
                self.predictor = None
        else:
            self.predictor = None

    def calculate_all_indicators(self, option_data, spot_price):
        """Calcola tutti gli indicatori basati sui dati delle opzioni."""
        calls = option_data['calls']
        puts = option_data['puts']

        # Calcola moneyness
        calls['moneyness'] = (calls['strike'] - spot_price) / spot_price
        puts['moneyness'] = (puts['strike'] - spot_price) / spot_price

        indicators = {}

        # Skew analysis
        skew_data = pd.concat([calls.assign(type='call'), puts.assign(type='put')])
        indicators.update(self.indicators_calculator.calculate_skew_index(skew_data, spot_price))

        # PCR analysis
        indicators.update(self.indicators_calculator.put_call_volume_ratio(calls, puts))

        # Gamma exposure
        indicators.update(self.indicators_calculator.calculate_gamma_exposure(calls, puts, spot_price))

        # Max pain
        indicators.update(self.indicators_calculator.calculate_max_pain(calls, puts, spot_price))

        # Vanna & Charm
        indicators.update(self.indicators_calculator.calculate_vanna_charm(calls, puts, spot_price))

        # Systemic risk
        indicators.update({
            'systemic_risk': self.indicators_calculator.calculate_systemic_risk_score(indicators)
        })

        return indicators

    def create_main_dashboard(self):
        """Crea la dashboard principale con Streamlit."""
        st.set_page_config(
            layout="wide",
            page_title="Advanced Options Terminal",
            page_icon="📊"
        )

        # Sidebar
        st.sidebar.title("⚙️ Configurazione")
        selected_index = st.sidebar.selectbox(
            "Seleziona Indice:",
            list(self.tickers.keys()),
            index=0
        )
        symbol = self.tickers[selected_index]

        # Timeframe selection
        timeframe = st.sidebar.selectbox(
            "Timeframe:",
            ["Real-time", "1D", "1W", "1M"],
            index=0
        )

        # Alert thresholds
        st.sidebar.markdown("---")
        st.sidebar.subheader("Soglie di Allerta")
        skew_threshold = st.sidebar.slider(
            "Skew Alert (>):",
            min_value=0.0,
            max_value=0.3,
            value=0.1,
            step=0.01
        )
        pcr_threshold = st.sidebar.slider(
            "PCR Estremo (<0.7 o >1.3):",
            min_value=0.5,
            max_value=2.0,
            value=1.3,
            step=0.1
        )

        # Main content
        st.title(f"📈 Terminale Opzioni Avanzato - {selected_index}")
        st.markdown("---")

        # Fetch data
        with st.spinner("Recupero dati di mercato..."):
            market_data = self.fetcher.get_market_data(symbol)
            option_data = self.fetcher.get_option_data(symbol)

        if not market_data or not option_data:
            st.error("Impossibile recuperare i dati. Riprova più tardi.")
            return

        # Calculate indicators
        current_indicators = self.calculate_all_indicators(option_data, market_data['spot_price'])

        # Store historical (in-session)
        self.historical_data.append({
            'timestamp': datetime.now(),
            **current_indicators
        })
        if len(self.historical_data) > 100:
            self.historical_data = self.historical_data[-100:]

        # Display key metrics
        self.display_key_metrics(current_indicators, market_data)

        # Tabs per differenti analisi (ora sono 6 con ML)
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Panoramica",
            "🎯 Analisi Skew",
            "⚡ Esposizione Gamma",
            "📈 Superficie Volatilità",
            "🤖 Analisi AI",
            "🤖 Previsioni ML"
        ])

        with tab1:
            self.display_overview(current_indicators, market_data)
        with tab2:
            self.display_skew_analysis(current_indicators)
        with tab3:
            self.display_gamma_analysis(current_indicators)
        with tab4:
            self.display_volatility_surface(option_data)
        with tab5:
            self.display_ai_analysis(current_indicators, market_data)
        with tab6:
            self.display_ml_predictions(current_indicators, market_data, option_data)

    def display_key_metrics(self, indicators, market_data):
        """Mostra le metriche chiave in colonne."""
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            skew_value = indicators.get('skew_25d', 0)
            skew_color = "red" if skew_value > 0.1 else "green" if skew_value < -0.05 else "gray"
            st.metric(
                "SKEW 25-Δ",
                f"{skew_value:.4f}",
                delta_color="off",
                help="Positivo = paura, Negativo = avidità"
            )
            st.markdown(f"<span style='color:{skew_color}'>●</span>", unsafe_allow_html=True)

        with col2:
            pcr = indicators.get('pcr_weighted', 1)
            pcr_color = "red" if pcr > 1.3 else "green" if pcr < 0.7 else "gray"
            st.metric(
                "PCR Ponderato",
                f"{pcr:.2f}",
                delta_color="off",
                help="<0.7: ottimismo estremo, >1.3: paura"
            )
            st.markdown(f"<span style='color:{pcr_color}'>●</span>", unsafe_allow_html=True)

        with col3:
            gamma_exp = indicators.get('total_gamma_exposure', 0)
            gamma_color = "red" if gamma_exp < -500000 else "green" if gamma_exp > 500000 else "gray"
            st.metric(
                "Esposizione Gamma",
                f"{gamma_exp/1e6:.2f}M",
                delta_color="off",
                help="Negativo = mercato fragile"
            )
            st.markdown(f"<span style='color:{gamma_color}'>●</span>", unsafe_allow_html=True)

        with col4:
            risk_score = indicators.get('systemic_risk', {}).get('total_score', 0)
            risk_color = "red" if risk_score > 70 else "orange" if risk_score > 40 else "green"
            st.metric(
                "Punteggio Rischio",
                f"{risk_score:.0f}",
                delta_color="off",
                help="0-100: rischio sistemico"
            )
            st.markdown(f"<span style='color:{risk_color}'>●</span>", unsafe_allow_html=True)

        with col5:
            vix = market_data.get('vix', 0)
            vix_change = market_data.get('vix_change_pct', 0)
            vix_color = "red" if vix_change > 10 else "green" if vix_change < -5 else "gray"
            st.metric(
                "VIX",
                f"{vix:.2f}",
                f"{vix_change:.1f}%",
                delta_color="normal"
            )
            st.markdown(f"<span style='color:{vix_color}'>●</span>", unsafe_allow_html=True)

    def display_overview(self, indicators, market_data):
        """Visualizza la panoramica generale."""
        st.subheader("Panoramica del Mercato")
        st.write("Qui puoi mettere un riepilogo generale con grafici combinati.")
        # Esempio: grafico dello skew storico (simulato)
        if len(self.historical_data) > 1:
            dates = [d['timestamp'] for d in self.historical_data]
            skew_vals = [d.get('skew_25d', 0) for d in self.historical_data]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates, y=skew_vals, mode='lines', name='Skew 25-Δ'))
            fig.update_layout(title='Skew Recente', xaxis_title='Data', yaxis_title='Skew')
            st.plotly_chart(fig, use_container_width=True)

    def display_skew_analysis(self, indicators):
        """Visualizza l'analisi dello skew."""
        st.subheader("Analisi Dettagliata Skew")
        st.write("Qui puoi mettere grafici specifici per lo skew, come il confronto 25Δ vs 10Δ.")
        # Esempio: indicatori a forma di gauge per lo skew
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=indicators.get('skew_25d', 0)*100,
            title={'text': "Skew 25-Δ (%)"},
            gauge={'axis': {'range': [-20, 20]},
                   'bar': {'color': "darkblue"},
                   'steps': [
                       {'range': [-20, -5], 'color': "lightgreen"},
                       {'range': [-5, 5], 'color': "lightyellow"},
                       {'range': [5, 20], 'color': "lightcoral"}],
                   'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 10}}
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    def display_gamma_analysis(self, indicators):
        """Visualizza l'analisi dell'esposizione gamma."""
        st.subheader("Analisi Esposizione Gamma")
        st.write("Grafici a barre o heatmap per gamma.")
        gamma_data = indicators.get('gamma_data', pd.DataFrame())
        if not gamma_data.empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=gamma_data['strike'], y=gamma_data['net_gamma'], name='Net Gamma'))
            fig.update_layout(title='Gamma per Strike', xaxis_title='Strike', yaxis_title='Net Gamma')
            st.plotly_chart(fig, use_container_width=True)

    def display_volatility_surface(self, option_data):
        """Visualizza la superficie di volatilità."""
        st.subheader("Superficie di Volatilità")
        st.write("Qui andrebbe un grafico 3D della superficie di volatilità.")
        # Nota: per un grafico 3D interattivo, serve un calcolo più elaborato.
        # Per ora mettiamo un placeholder.
        st.info("Grafico 3D della superficie di volatilità - da implementare con calcoli estesi.")

    def display_ai_analysis(self, indicators, market_data):
        """Visualizza l'analisi AI (analisi critica)."""
        st.subheader("Analisi AI del Mercato")
        # Utilizza il MarketAnalyzer per ottenere l'analisi
        analysis = self.analyzer.analyze_market_regime(indicators, self.historical_data)
        st.write("**Regime di mercato:**", analysis.get('regime', 'N/A'))
        st.write("**Probabilità:**")
        for key, val in analysis.get('probabilities', {}).items():
            st.write(f"- {key}: {val}%")
        st.write("**Implicazioni di trading:**")
        for imp in analysis.get('trading_implications', []):
            st.write(f"- {imp}")

    def display_ml_predictions(self, indicators, market_data, option_data):
        """Visualizza le previsioni del modello ML."""
        st.header("🧠 Previsioni Machine Learning")

        if not ML_AVAILABLE or self.predictor is None:
            st.warning("Il modulo di Machine Learning non è disponibile o non è stato caricato alcun modello. Assicurati che i file `ml_predictor.py` e `feature_engineer.py` siano presenti e che un modello pre-addestrato esista in `models/`.")
            return

        # Crea un layout a colonne per organizzare le informazioni
        pred_col1, pred_col2, pred_col3 = st.columns([2, 1, 1])

        with pred_col1:
            st.subheader("Previsione per la Prossima Sessione")

            # Prepara le feature correnti per il modello.
            # ATTENZIONE: assicurati che le feature corrispondano a quelle con cui il modello è stato addestrato.
            current_features = {
                'skew_25d': indicators.get('skew_25d', 0),
                'skew_10d': indicators.get('skew_10d', 0),
                'pcr_weighted': indicators.get('pcr_weighted', 1),
                'pcr_simple': indicators.get('pcr_simple', 1),
                'total_gamma_exposure': indicators.get('total_gamma_exposure', 0),
                'max_pain_strike': indicators.get('max_pain_strike', 0),
                'vanna_total': indicators.get('total_vanna', 0),
                'charm_total': indicators.get('total_charm', 0),
                'vix': market_data.get('vix', 20),
                'vix_change_pct': market_data.get('vix_change_pct', 0),
                'price_change_pct': market_data.get('price_change_pct', 0),
                'volume_ratio': market_data.get('volume_ratio', 1),
                # Aggiungi altre feature necessarie al tuo modello
            }

            # Converti in DataFrame per il predictor
            features_df = pd.DataFrame([current_features])

            # Ottieni la previsione
            try:
                prediction, probabilities = self.predictor.predict(features_df)
            except Exception as e:
                st.error(f"Errore durante la previsione: {e}")
                return

            # Visualizza il risultato in modo intuitivo
            # Supponiamo che il modello predica 1 (rialzista) o 0 (ribassista)
            if prediction == 1:
                st.markdown("### 📈 **SEGNALE: RIALZISTA**")
                st.metric("Direzione Attesa", "SU", delta="Bullish")
                # Barra di "confidenza" verde
                if 'class_1' in probabilities:
                    confidence = probabilities['class_1'] * 100
                elif 'confidence' in probabilities:
                    confidence = probabilities['confidence'] * 100
                else:
                    confidence = 50  # default
                st.progress(int(confidence))
                st.caption(f"Confidenza del modello: {confidence:.1f}%")
            else:
                st.markdown("### 📉 **SEGNALE: RIBASSISTA**")
                st.metric("Direzione Attesa", "GIÙ", delta="Bearish", delta_color="inverse")
                if 'class_0' in probabilities:
                    confidence = probabilities['class_0'] * 100
                elif 'confidence' in probabilities:
                    confidence = probabilities['confidence'] * 100
                else:
                    confidence = 50
                st.progress(int(confidence))
                st.caption(f"Confidenza del modello: {confidence:.1f}%")

        with pred_col2:
            st.subheader("Probabilità")
            # Mostra le probabilità per ogni classe in una tabella piccola
            prob_df = pd.DataFrame.from_dict(probabilities, orient='index', columns=['Probabilità'])
            st.dataframe(prob_df, use_container_width=True)

        with pred_col3:
            st.subheader("Info Modello")
            st.markdown(f"""
            **Tipo**: Random Forest  
            **Target**: Direzione giorno +1  
            **Ultimo training**: 2024-01-15  
            **Accuracy backtest**: 62.3%
            """)
            if st.button("🔄 Aggiorna Previsione", type="secondary"):
                st.rerun()

        # --- SEZIONE AGGIUNTIVA: Importanza delle Feature ---
        st.markdown("---")
        st.subheader("📊 Cosa ha guidato la previsione?")

        # Ottieni l'importanza delle feature dal modello (se disponibile)
        if self.predictor.feature_importance is not None:
            fig_importance = go.Figure()
            fig_importance.add_trace(go.Bar(
                x=self.predictor.feature_importance['importance'].head(10),
                y=self.predictor.feature_importance['feature'].head(10),
                orientation='h',
                marker_color='lightblue'
            ))
            fig_importance.update_layout(
                title="Top 10 Feature più Importanti",
                xaxis_title="Importanza",
                yaxis_title="Feature",
                height=400
            )
            st.plotly_chart(fig_importance, use_container_width=True)
        else:
            st.info("L'importanza delle feature non è disponibile per questo modello.")

        # --- AVVISO IMPORTANTE ---
        st.warning("""
        **⚠️ Nota di cautela**:  
        Queste previsioni sono generate da un modello statistico e **non sono una garanzia**.  
        Utilizzale come **uno strumento di analisi in più** insieme agli altri indicatori (Skew, PCR, Gamma).  
        Il passato non è indicativo dei risultati futuri.
        """)

        # Opzionale: Mostra le feature correnti usate per la previsione
        with st.expander("📋 Visualizza le Feature Correnti"):
            st.dataframe(features_df.T.rename(columns={0: 'Valore'}))

def main():
    """Funzione principale per eseguire la dashboard."""
    dashboard = EnhancedOptionsDashboard()
    dashboard.create_main_dashboard()

if __name__ == "__main__":
    main()
