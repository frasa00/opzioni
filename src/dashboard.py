"""
Dashboard Streamlit aggiornata con nuovi indicatori
"""

import streamlit as st
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import local modules
from data_fetcher import DataFetcher
from indicators import AdvancedOptionsIndicators
from analyzer import MarketAnalyzer

class EnhancedOptionsDashboard:
    """Dashboard avanzata per opzioni"""
    
    def __init__(self):
        self.tickers = {
            'SP500': '^SPX',
            'NASDAQ': '^NDX'
        }
        self.fetcher = DataFetcher()
        self.indicators = AdvancedOptionsIndicators()
        self.analyzer = MarketAnalyzer()
        self.historical_data = []
        
    def create_main_dashboard(self):
        """Crea dashboard principale"""
        st.set_page_config(
            layout="wide",
            page_title="Advanced Options Terminal",
            page_icon="📊"
        )
        
        # Sidebar
        st.sidebar.title("⚙️ Configuration")
        selected_index = st.sidebar.selectbox(
            "Select Index:",
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
        st.sidebar.subheader("Alert Thresholds")
        
        skew_threshold = st.sidebar.slider(
            "Skew Alert (>):",
            min_value=0.0,
            max_value=0.3,
            value=0.1,
            step=0.01
        )
        
        pcr_threshold = st.sidebar.slider(
            "PCR Extreme (<0.7 or >1.3):",
            min_value=0.5,
            max_value=2.0,
            value=1.3,
            step=0.1
        )
        
        # Main content
        st.title(f"📈 Advanced Options Terminal - {selected_index}")
        st.markdown("---")
        
        # Fetch data
        with st.spinner("Fetching market data..."):
            market_data = self.fetcher.get_market_data(symbol)
            option_data = self.fetcher.get_option_data(symbol)
        
        if not market_data or not option_data:
            st.error("Failed to fetch data. Please try again.")
            return
        
        # Calculate indicators
        current_indicators = self.calculate_all_indicators(
            option_data, 
            market_data['spot_price']
        )
        
        # Store historical
        self.historical_data.append({
            'timestamp': datetime.now(),
            **current_indicators
        })
        
        # Keep only last 100 records
        if len(self.historical_data) > 100:
            self.historical_data = self.historical_data[-100:]
        
        # Display in columns
        self.display_key_metrics(current_indicators, market_data)
        
        # Tabs for different analyses
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview", 
            "🎯 Skew Analysis", 
            "⚡ Gamma Exposure",
            "📈 Volatility Surface",
            "🤖 AI Analysis"
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
    
    def display_key_metrics(self, indicators, market_data):
        """Display key metrics in columns"""
        
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
                "PCR Weighted",
                f"{pcr:.2f}",
                delta_color="off",
                help="<0.7: ottimismo estremo, >1.3: paura"
            )
            st.markdown(f"<span style='color:{pcr_color}'>●</span>", unsafe_allow_html=True)
        
        with col3:
            gamma_exp = indicators.get('total_gamma_exposure', 0)
            gamma_color = "red" if gamma_exp < -500000 else "green" if gamma_exp > 500000 else "gray"
            st.metric(
                "Gamma Exposure",
                f"{gamma_exp/1e6:.2f}M",
                delta_color="off",
                help="Negativo = mercato fragile"
            )
            st.markdown(f"<span style='color:{gamma_color}'>●</span>", unsafe_allow_html=True)
        
        with col4:
            risk_score = indicators.get('systemic_risk', {}).get('total_score', 0)
            risk_color = "red" if risk_score > 70 else "orange" if risk_score > 40 else "green"
            st.metric(
                "Risk Score",
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
    
    def calculate_all_indicators(self, option_data, spot_price):
        """Calculate all indicators"""
        
        calls = option_data['calls']
        puts = option_data['puts']
        
        # Calculate moneyness
        calls['moneyness'] = (calls['strike'] - spot_price) / spot_price
        puts['moneyness'] = (puts['strike'] - spot_price) / spot_price
        
        indicators = {}
        
        # Skew analysis
        indicators.update(
            self.indicators.calculate_skew_index(
                pd.concat([calls.assign(type='call'), puts.assign(type='put')]),
                spot_price
            )
        )
        
        # PCR analysis
        indicators.update(
            self.indicators.put_call_volume_ratio(calls, puts)
        )
        
        # Gamma exposure
        indicators.update(
            self.indicators.calculate_gamma_exposure(calls, puts, spot_price)
        )
        
        # Max pain
        indicators.update(
            self.indicators.calculate_max_pain(calls, puts, spot_price)
        )
        
        # Vanna & Charm
        indicators.update(
            self.indicators.calculate_vanna_charm(calls, puts, spot_price)
        )
        
        # Systemic risk
        indicators.update(
            systemic_risk=self.indicators.calculate_systemic_risk_score(indicators)
        )
        
        return indicators
    
    def display_skew_analysis(self, indicators):
        """Display skew analysis"""
        
        st.subheader("📊 Skew Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Skew comparison chart
            fig = go.Figure()
            
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=indicators.get('skew_25d', 0) * 100,
                title={'text': "Skew 25-Δ"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [-20, 20]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [-20, -5], 'color': "lightgreen"},
                        {'range': [-5, 5], 'color': "lightyellow"},
                        {'range': [5, 20], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 10
                    }
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Tail risk indicator
            tail_risk = indicators.get('tail_risk_ratio', 0)
            
            fig2 = go.Figure(go.Indicator(
                mode="number+gauge",
                value=tail_risk,
                title={'text': "Tail Risk Ratio (10Δ/25Δ)"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'shape': "bullet",
                    'axis': {'range': [0, 3]},
                    'threshold': {
                        'line': {'color': "red", 'width': 2},
                        'thickness': 0.75,
                        'value': 1.5
                    },
                    'steps': [
                        {'range': [0, 1], 'color': "gray"},
                        {'range': [1, 2], 'color': "orange"},
                        {'range': [2, 3], 'color': "red"}
                    ],
                    'bar': {'color': "black"}
                }
            ))
            
            fig2.update_layout(height=300)
            st.plotly_chart(fig2, use_container_width=True)
        
        # Historical skew chart
        if len(self.historical_data) > 1:
            st.subheader("📅 Historical Skew Trend")
            
            dates = [d['timestamp'] for d in self.historical_data]
            skew_values = [d.get('skew_25d', 0) for d in self.historical_data]
            
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=dates, y=skew_values,
                mode='lines+markers',
                name='Skew 25-Δ',
                line=dict(color='royalblue', width=2),
                fill='tozeroy',
                fillcolor='rgba(65, 105, 225, 0.2)'
            ))
            fig3.add_hline(y=0, line_dash="dash", line_color="gray")
            fig3.update_layout(
                xaxis_title="Time",
                yaxis_title="Skew Value",
                height=400
            )
            st.plotly_chart(fig3, use_container_width=True)
    
    def display_gamma_analysis(self, indicators):
        """Display gamma exposure analysis"""
        
        st.subheader("⚡ Gamma Exposure Analysis")
        
        gamma_data = indicators.get('gamma_data', pd.DataFrame())
        
        if not gamma_data.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Gamma exposure by strike
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=gamma_data['strike'],
                    y=gamma_data['net_gamma'],
                    name='Net Gamma',
                    marker_color=np.where(gamma_data['net_gamma'] > 0, 'green', 'red')
                ))
                
                fig.update_layout(
                    title="Gamma by Strike",
                    xaxis_title="Strike",
                    yaxis_title="Net Gamma",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Gamma exposure heatmap
                fig2 = go.Figure(data=go.Heatmap(
                    z=gamma_data['net_gamma'].values.reshape(-1, 1),
                    x=gamma_data['strike'],
                    y=['Gamma'],
                    colorscale='RdBu',
                    zmid=0
                ))
                
                fig2.update_layout(
                    title="Gamma Exposure Heatmap",
                    xaxis_title="Strike",
                    height=400
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            # Gamma flip zone info
            st.info(f"""
            **Gamma Flip Zone**: {indicators.get('gamma_flip_zone', 0):.2f}
            
            **Positive Gamma Strikes**: {len(indicators.get('positive_gamma_strikes', []))}
            
            **Negative Gamma Strikes**: {len(indicators.get('negative_gamma_strikes', []))}
            
            **Total Gamma Exposure**: {indicators.get('total_gamma_exposure', 0):.2e}
            """)
    
    def display_volatility_surface(self, option_data):
        """Display volatility surface"""
        
        st.subheader("📈 Volatility Surface")
        
        calls = option_data['calls']
        puts = option_data['puts']
        
        # Create volatility surface
        surface = self.indicators.calculate_volatility_surface(calls, puts)
        surface_data = surface['surface_data']
        
        if not surface_data.empty:
            fig = go.Figure(data=[
                go.Mesh3d(
                    x=surface_data['strike'],
                    y=surface_data['days_to_expiry'],
                    z=surface_data['iv'],
                    intensity=surface_data['iv'],
                    colorscale='Viridis',
                    opacity=0.8
                )
            ])
            
            fig.update_layout(
                title="3D Volatility Surface",
                scene=dict(
                    xaxis_title='Strike',
                    yaxis_title='Days to Expiry',
                    zaxis_title='Implied Volatility'
                ),
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def display_ai_analysis(self, indicators, market_data):
        """Display AI-powered analysis"""
        
        st.subheader("🤖 AI Market Analysis")
        
        # Get analysis from analyzer
        analysis = self.analyzer.analyze_market_regime(
            indicators,
            self.historical_data
        )
        
        # Display regime
        regime = analysis.get('regime', 'NORMAL')
        regime_color = {
            'FEAR': 'red',
            'COMPLACENCY': 'orange',
            'FRAGILE': 'purple',
            'NORMAL': 'green'
        }.get(regime, 'gray')
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Market Regime",
                regime,
                delta_color="off"
            )
            st.markdown(f"<span style='color:{regime_color}; font-size: 24px'>●</span>", 
                       unsafe_allow_html=True)
        
        with col2:
            confidence = analysis.get('confidence', 0)
            st.metric(
                "Analysis Confidence",
                f"{confidence}%",
                delta_color="off"
            )
        
        with col3:
            anomalies = len(analysis.get('anomalies', []))
            st.metric(
                "Anomalies Detected",
                anomalies,
                delta_color="off"
            )
        
        # Probabilities
        st.subheader("📊 Probability Assessment")
        
        probs = analysis.get('probabilities', {})
        
        prob_cols = st.columns(len(probs))
        for idx, (key, value) in enumerate(probs.items()):
            with prob_cols[idx]:
                color = "green" if "bull" in key else "red" if "bear" in key else "orange"
                st.markdown(f"""
                <div style="text-align: center;">
                    <div style="font-size: 24px; color: {color}; font-weight: bold;">
                        {value}%
                    </div>
                    <div style="font-size: 12px; text-transform: uppercase;">
                        {key.replace('_', ' ')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Trading implications
        st.subheader("🎯 Trading Implications")
        
        implications = analysis.get('trading_implications', [])
        
        for imp in implications:
            if "🚨" in imp or "⚠️" in imp:
                st.warning(imp)
            elif "🎯" in imp or "📊" in imp:
                st.info(imp)
            else:
                st.write(imp)
        
        # Generate signals
        st.subheader("📡 Trading Signals")
        
        signals = self.analyzer.generate_trading_signals(
            analysis,
            market_data['spot_price']
        )
        
        for signal in signals:
            with st.expander(f"{signal['type']} - {signal['asset']} ({signal['confidence']}% confidence)"):
                st.write(f"**Reason**: {signal['reason']}")
                st.write(f"**Details**: {signal}")
        
        # Historical comparison
        if 'historical_pattern' in analysis:
            st.subheader("📚 Historical Pattern Match")
            
            pattern = analysis['historical_pattern']
            st.success(f"""
            **Pattern**: {pattern['period']}
            
            **Similarity**: {pattern['similarity']:.0%}
            
            **Historical Outcome**: {pattern['outcome']}
            """)

def main():
    """Main function"""
    dashboard = EnhancedOptionsDashboard()
    dashboard.create_main_dashboard()

if __name__ == "__main__":
    main()
