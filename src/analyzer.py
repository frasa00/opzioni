"""
Analisi critica dei dati di mercato con AI semplice
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest

class MarketAnalyzer:
    """Analisi avanzata del sentiment di mercato"""
    
    def __init__(self):
        self.historical_patterns = self.load_historical_patterns()
    
    def load_historical_patterns(self):
        """Pattern storici per confronto"""
        return {
            'flash_crash_2020': {
                'skew': 0.15,
                'pcr': 1.8,
                'vix_change': 45,
                'gamma': -1.2e6
            },
            'bull_market_2021': {
                'skew': -0.03,
                'pcr': 0.6,
                'vix_change': -5,
                'gamma': 0.8e6
            }
        }
    
    def analyze_market_regime(self, current_data, historical_data):
        """Determina il regime di mercato attuale"""
        
        analysis = {
            'regime': 'NORMAL',
            'confidence': 0,
            'anomalies': [],
            'probabilities': {},
            'trading_implications': []
        }
        
        # 1. Analisi SKEW
        current_skew = current_data.get('skew_25d', 0)
        skew_trend = self.calculate_trend(historical_data, 'skew_25d')
        
        if current_skew > 0.1:
            analysis['regime'] = 'FEAR'
            analysis['confidence'] += 25
            
            if skew_trend < 0:  # Skew in calo
                analysis['trading_implications'].append(
                    "🔻 Skew in calo durante paura: possibile buy signal"
                )
                analysis['probabilities']['reversal_up'] = 65
            else:
                analysis['probabilities']['continuation_down'] = 70
        
        elif current_skew < -0.08:
            analysis['regime'] = 'COMPLACENCY'
            analysis['confidence'] += 20
            analysis['trading_implications'].append(
                "⚠️ Skew negativo estremo: mercato troppo ottimista - rischio correzione"
            )
            analysis['probabilities']['correction'] = 60
        
        # 2. Analisi PCR vs Prezzo
        pcr = current_data.get('pcr_weighted', 1)
        price_change = current_data.get('price_change_pct', 0)
        
        if pcr < 0.7 and price_change > 0:
            # Buy call estrema in rally - pericoloso
            analysis['anomalies'].append('EXTREME_OPTIMISM')
            analysis['trading_implications'].append(
                "🎯 PCR basso in rally: possibili acquisti forzati - attenzione a squeezes"
            )
            analysis['probabilities']['short_squeeze'] = 55
        
        elif pcr > 1.3 and price_change < -1:
            analysis['anomalies'].append('PANIC_SELLING')
            analysis['trading_implications'].append(
                "📉 PCR alto in sell-off: paura estrema - possibile oversold"
            )
            analysis['probabilities']['bounce'] = 60
        
        # 3. Analisi Gamma Exposure
        gamma = current_data.get('total_gamma_exposure', 0)
        
        if gamma < -500000:  # Gamma negativo significativo
            analysis['regime'] = 'FRAGILE'
            analysis['confidence'] += 30
            analysis['trading_implications'].append(
                "⚠️ Gamma Exposure negativo: rischio accelerazione movimenti"
            )
            
            # Verifica se siamo vicini a gamma flip
            flip_zone = current_data.get('gamma_flip_zone', 0)
            spot = current_data.get('spot_price', 0)
            
            if abs(flip_zone - spot) / spot < 0.02:  # Entro 2%
                analysis['anomalies'].append('GAMMA_FLIP_IMMINENT')
                analysis['trading_implications'].append(
                    "🚨 VICINI A GAMMA FLIP: possibili movimenti violenti"
                )
                analysis['probabilities']['high_volatility'] = 75
        
        # 4. Analisi Volatilità
        vix_change = current_data.get('vix_change_pct', 0)
        
        if vix_change > 15 and price_change > -1:
            # VIX sale senza motivo apparente
            analysis['anomalies'].append('VOLATILITY_DIVERGENCE')
            analysis['trading_implications'].append(
                "📊 VIX in aumento senza sell-off: hedge in arrivo - attenzione"
            )
            analysis['probabilities']['down_move'] = 65
        
        # 5. Analisi Max Pain vs Spot
        max_pain = current_data.get('max_pain_strike', 0)
        spot = current_data.get('spot_price', 0)
        
        if abs(max_pain - spot) / spot > 0.03:  > 3%
            analysis['trading_implications'].append(
                f"🎯 Max Pain a {max_pain:.2f} (distanza: {((max_pain-spot)/spot*100):.1f}%) - attrazione magnetica"
            )
            if max_pain > spot:
                analysis['probabilities']['pull_up'] = 60
            else:
                analysis['probabilities']['pull_down'] = 60
        
        # 6. Confronto con pattern storici
        pattern_match = self.compare_with_historical(current_data)
        if pattern_match:
            analysis['historical_pattern'] = pattern_match
            analysis['trading_implications'].append(
                f"📚 Pattern simile a {pattern_match['period']}: {pattern_match['outcome']}"
            )
        
        # Calcola probabilità finale
        analysis = self.calculate_final_probabilities(analysis)
        
        return analysis
    
    def calculate_trend(self, historical_data, indicator, window=5):
        """Calcola trend dell'indicatore"""
        if len(historical_data) < window:
            return 0
        
        values = [d.get(indicator, 0) for d in historical_data[-window:]]
        if len(values) < 2:
            return 0
        
        # Regressione lineare semplice
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        return slope
    
    def compare_with_historical(self, current_data):
        """Confronta con pattern storici"""
        best_match = None
        best_score = 0
        
        for period, pattern in self.historical_patterns.items():
            score = 0
            matches = 0
            
            for key, historical_value in pattern.items():
                current_value = current_data.get(key, 0)
                
                if historical_value == 0:
                    continue
                
                # Calcola similarità
                similarity = 1 - abs(current_value - historical_value) / abs(historical_value)
                if similarity > 0.7:  > 70%
                    score += similarity
                    matches += 1
            
            if matches > 0:
                avg_score = score / matches
                if avg_score > best_score:
                    best_score = avg_score
                    best_match = {
                        'period': period,
                        'similarity': avg_score,
                        'outcome': self.get_pattern_outcome(period)
                    }
        
        return best_match if best_score > 0.6 else None
    
    def get_pattern_outcome(self, period):
        """Risultato storico del pattern"""
        outcomes = {
            'flash_crash_2020': 'Fortissimo sell-off seguito da rimbalzo',
            'bull_market_2021': 'Rally sostenuto con correzioni minori'
        }
        return outcomes.get(period, 'Sconosciuto')
    
    def calculate_final_probabilities(self, analysis):
        """Calcola probabilità finali basate su tutti i fattori"""
        
        base_probs = {
            'bullish': 40,
            'bearish': 30,
            'high_volatility': 30,
            'range_bound': 40
        }
        
        # Adjust based on regime
        if analysis['regime'] == 'FEAR':
            base_probs['bearish'] += 20
            base_probs['high_volatility'] += 25
            base_probs['bullish'] -= 15
        
        elif analysis['regime'] == 'COMPLACENCY':
            base_probs['bullish'] += 15
            base_probs['bearish'] -= 10
            base_probs['high_volatility'] -= 10
        
        elif analysis['regime'] == 'FRAGILE':
            base_probs['high_volatility'] += 35
            base_probs['range_bound'] -= 20
        
        # Adjust based on anomalies
        if 'EXTREME_OPTIMISM' in analysis.get('anomalies', []):
            base_probs['bearish'] += 15
            base_probs['high_volatility'] += 10
        
        if 'PANIC_SELLING' in analysis.get('anomalies', []):
            base_probs['bullish'] += 20  # Oversold bounce
        
        # Normalize to 100%
        total = sum(base_probs.values())
        analysis['probabilities'] = {k: int(v/total * 100) for k, v in base_probs.items()}
        
        return analysis
    
    def generate_trading_signals(self, analysis, current_price):
        """Genera segnali di trading specifici"""
        
        signals = []
        
        # Signal 1: Skew Reversal
        if (analysis.get('regime') == 'FEAR' and 
            analysis.get('probabilities', {}).get('reversal_up', 0) > 60):
            signals.append({
                'type': 'LONG',
                'asset': 'CALLS',
                'strike': 'ATM',
                'expiry': '1-2 weeks',
                'confidence': 65,
                'reason': 'Skew fear extreme + declining trend'
            })
        
        # Signal 2: Gamma Flip Play
        if 'GAMMA_FLIP_IMMINENT' in analysis.get('anomalies', []):
            signals.append({
                'type': 'STRADDLE',
                'asset': 'ATM Options',
                'expiry': 'Weekly',
                'confidence': 70,
                'reason': 'Imminent gamma flip - expect violent move'
            })
        
        # Signal 3: Max Pain Gravitation
        max_pain = analysis.get('max_pain_strike', 0)
        if abs(max_pain - current_price) / current_price > 0.03:
            direction = 'CALLS' if max_pain > current_price else 'PUTS'
            signals.append({
                'type': 'CREDIT_SPREAD',
                'direction': direction,
                'strikes': [current_price, max_pain],
                'confidence': 60,
                'reason': 'Strong max pain gravitational pull'
            })
        
        return signals
