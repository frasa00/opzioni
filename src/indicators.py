"""
Indicatori avanzati per opzioni - File principale per i calcoli
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
import pandas_ta as ta

class AdvancedOptionsIndicators:
    """Classe per indicatori avanzati di opzioni"""
    
    @staticmethod
    def calculate_skew_index(option_chain_df, spot_price):
        """
        Skew Index avanzato - misura la paura/avidità
        Restituisce:
        - Skew 25-delta
        - Skew 10-delta (tail risk)
        - Skew curvature
        """
        calls = option_chain_df[option_chain_df['type'] == 'call'].copy()
        puts = option_chain_df[option_chain_df['type'] == 'put'].copy()
        
        # Calcola delta per ogni opzione (approssimato)
        calls['delta'] = norm.cdf(calls['moneyness'])
        puts['delta'] = norm.cdf(puts['moneyness']) - 1
        
        # Skew 25-delta (standard)
        calls_25d = calls[(calls['delta'] >= 0.2) & (calls['delta'] <= 0.3)]
        puts_25d = puts[(puts['delta'] <= -0.2) & (puts['delta'] >= -0.3)]
        
        skew_25d = puts_25d['impliedVolatility'].mean() - calls_25d['impliedVolatility'].mean()
        
        # Skew 10-delta (tail risk)
        calls_10d = calls[(calls['delta'] >= 0.05) & (calls['delta'] <= 0.15)]
        puts_10d = puts[(puts['delta'] <= -0.05) & (puts['delta'] >= -0.15)]
        
        skew_10d = puts_10d['impliedVolatility'].mean() - calls_10d['impliedVolatility'].mean()
        
        # Term structure skew
        skew_curvature = skew_10d - skew_25d
        
        return {
            'skew_25d': skew_25d,
            'skew_10d': skew_10d,
            'skew_curvature': skew_curvature,
            'tail_risk_ratio': skew_10d / skew_25d if skew_25d != 0 else 0
        }
    
    @staticmethod
    def put_call_volume_ratio(calls, puts, method='simple'):
        """
        Calcola diversi tipi di PCR:
        - Simple: volume totale
        - Weighted: ponderato per moneyness
        - Smoothed: media mobile
        """
        total_put_vol = puts['volume'].sum()
        total_call_vol = calls['volume'].sum()
        
        # PCR semplice
        pcr_simple = total_put_vol / total_call_vol if total_call_vol > 0 else 0
        
        # PCR ponderato per moneyness (più vicino allo strike, più peso)
        puts['moneyness_weight'] = np.exp(-abs(puts['moneyness']))
        calls['moneyness_weight'] = np.exp(-abs(calls['moneyness']))
        
        weighted_put_vol = (puts['volume'] * puts['moneyness_weight']).sum()
        weighted_call_vol = (calls['volume'] * calls['moneyness_weight']).sum()
        pcr_weighted = weighted_put_vol / weighted_call_vol if weighted_call_vol > 0 else 0
        
        # PCR per strike bucket
        strikes = np.linspace(calls['strike'].min(), calls['strike'].max(), 10)
        pcr_by_strike = {}
        
        for i in range(len(strikes)-1):
            low, high = strikes[i], strikes[i+1]
            put_vol = puts[(puts['strike'] >= low) & (puts['strike'] < high)]['volume'].sum()
            call_vol = calls[(calls['strike'] >= low) & (calls['strike'] < high)]['volume'].sum()
            pcr_by_strike[f'{low:.0f}-{high:.0f}'] = put_vol/call_vol if call_vol > 0 else 0
        
        return {
            'pcr_simple': pcr_simple,
            'pcr_weighted': pcr_weighted,
            'pcr_by_strike': pcr_by_strike,
            'extremeness': abs(pcr_simple - 1)  # Quanto è estremo il PCR
        }
    
    @staticmethod
    def calculate_gamma_exposure(calls, puts, spot_price):
        """
        Gamma Exposure - misura della fragilità del mercato
        Calcola il gamma netto per livello di strike
        """
        # Gamma per opzione = gamma di Black-Scholes
        # Approx: gamma ≈ N'(d1) / (S * σ * √T)
        
        results = []
        all_strikes = sorted(set(calls['strike']).union(set(puts['strike'])))
        
        for strike in all_strikes:
            # Gamma per call
            call_gamma = 0
            if strike in calls['strike'].values:
                call_oi = calls[calls['strike'] == strike]['openInterest'].iloc[0]
                # Gamma approx
                call_gamma = call_oi * 100 * (1 / (spot_price * 0.2 * np.sqrt(30/365)))
            
            # Gamma per put (gamma è simmetrico per put e call)
            put_gamma = 0
            if strike in puts['strike'].values:
                put_oi = puts[puts['strike'] == strike]['openInterest'].iloc[0]
                put_gamma = put_oi * 100 * (1 / (spot_price * 0.2 * np.sqrt(30/365)))
            
            net_gamma = call_gamma - put_gamma  # Gamma netto
            gamma_exposure = net_gamma * (strike - spot_price)  # Esposizione gamma
            
            results.append({
                'strike': strike,
                'call_gamma': call_gamma,
                'put_gamma': put_gamma,
                'net_gamma': net_gamma,
                'gamma_exposure': gamma_exposure,
                'distance_pct': (strike - spot_price) / spot_price * 100
            })
        
        df = pd.DataFrame(results)
        
        # Total gamma exposure
        total_gamma_exposure = df['gamma_exposure'].sum()
        
        # Gamma flip zone (dove gamma cambia segno)
        positive_gamma = df[df['net_gamma'] > 0]
        negative_gamma = df[df['net_gamma'] < 0]
        
        return {
            'total_gamma_exposure': total_gamma_exposure,
            'positive_gamma_strikes': positive_gamma['strike'].tolist(),
            'negative_gamma_strikes': negative_gamma['strike'].tolist(),
            'gamma_flip_zone': spot_price * 1.05,  # Approssimato
            'gamma_data': df
        }
    
    @staticmethod
    def calculate_vanna_charm(calls, puts, spot_price):
        """
        Vanna (dV/dσ) e Charm (dV/dT) - sensibilità avanzate
        Vanna: come cambia il delta con la volatilità
        Charm: come cambia il delta con il tempo
        """
        # Calcoli approssimati
        vanna_total = 0
        charm_total = 0
        
        for _, row in calls.iterrows():
            # Approx vanna per call
            vanna = -row['openInterest'] * np.sqrt(row['days_to_expiry']/365) * norm.pdf(0.3)
            vanna_total += vanna
            
            # Approx charm per call (decay di delta)
            charm = -row['openInterest'] * norm.pdf(0.3) * (0.2 / (2 * np.sqrt(row['days_to_expiry']/365)))
            charm_total += charm
        
        for _, row in puts.iterrows():
            # Approx vanna per put
            vanna = row['openInterest'] * np.sqrt(row['days_to_expiry']/365) * norm.pdf(0.3)
            vanna_total += vanna
            
            # Approx charm per put
            charm = row['openInterest'] * norm.pdf(0.3) * (0.2 / (2 * np.sqrt(row['days_to_expiry']/365)))
            charm_total += charm
        
        return {
            'total_vanna': vanna_total,
            'total_charm': charm_total,
            'vanna_charm_ratio': abs(vanna_total / charm_total) if charm_total != 0 else 0
        }
    
    @staticmethod
    def calculate_max_pain(calls, puts, spot_price):
        """
        Max Pain Theory - prezzo che causa massima perdita ai trader di opzioni
        """
        strikes = sorted(set(calls['strike']).union(set(puts['strike'])))
        
        pain_points = []
        for strike in strikes:
            total_pain = 0
            
            # Calcola pain per ogni strike superiore (call writers perdono)
            for s in strikes:
                if s >= strike:
                    call_oi = calls[calls['strike'] == s]['openInterest'].sum()
                    total_pain += call_oi * (s - strike)
            
            # Calcola pain per ogni strike inferiore (put writers perdono)
            for s in strikes:
                if s <= strike:
                    put_oi = puts[puts['strike'] == s]['openInterest'].sum()
                    total_pain += put_oi * (strike - s)
            
            pain_points.append({
                'strike': strike,
                'total_pain': total_pain,
                'distance_from_spot': abs(strike - spot_price)
            })
        
        df = pd.DataFrame(pain_points)
        max_pain_strike = df.loc[df['total_pain'].idxmin(), 'strike']
        
        return {
            'max_pain_strike': max_pain_strike,
            'pain_at_spot': df[df['strike'] == spot_price]['total_pain'].iloc[0] if spot_price in df['strike'].values else 0,
            'pain_data': df,
            'gravitational_pull': (max_pain_strike - spot_price) / spot_price * 100
        }
    
    @staticmethod
    def calculate_volatility_surface(calls, puts):
        """
        Superficie di volatilità - IV per strike e expiry
        """
        surface_data = []
        
        # Per calls
        for _, row in calls.iterrows():
            surface_data.append({
                'strike': row['strike'],
                'expiry': row['expiry'],
                'iv': row['impliedVolatility'],
                'type': 'call',
                'moneyness': row.get('moneyness', 0)
            })
        
        # Per puts
        for _, row in puts.iterrows():
            surface_data.append({
                'strike': row['strike'],
                'expiry': row['expiry'],
                'iv': row['impliedVolatility'],
                'type': 'put',
                'moneyness': row.get('moneyness', 0)
            })
        
        df = pd.DataFrame(surface_data)
        
        # Calcola skew e smile
        smile_curvature = df.groupby('type')['iv'].std()
        
        return {
            'surface_data': df,
            'call_skew': smile_curvature.get('call', 0),
            'put_skew': smile_curvature.get('put', 0),
            'volatility_smile': df.groupby('strike')['iv'].mean().to_dict()
        }
    
    @staticmethod
    def calculate_systemic_risk_score(indicators_dict):
        """
        Punteggio di rischio sistemico basato su multipli indicatori
        """
        weights = {
            'skew_25d': 0.25,
            'pcr_weighted': 0.20,
            'total_gamma_exposure': 0.15,
            'vanna_charm_ratio': 0.10,
            'gravitational_pull': 0.15,
            'volatility_spread': 0.15
        }
        
        score = 0
        details = {}
        
        # Normalizza ogni indicatore a scala 0-100
        for indicator, weight in weights.items():
            value = indicators_dict.get(indicator, 0)
            
            if indicator == 'skew_25d':
                norm_value = min(abs(value) * 100, 100)  # Skew alto = rischio alto
                details['skew_score'] = norm_value
                score += norm_value * weight
            
            elif indicator == 'pcr_weighted':
                # PCR estremo (<0.7 o >1.3) = rischio alto
                if value < 0.7:
                    norm_value = ((0.7 - value) / 0.7) * 100
                elif value > 1.3:
                    norm_value = ((value - 1.3) / 1.3) * 100
                else:
                    norm_value = 0
                details['pcr_score'] = norm_value
                score += norm_value * weight
            
            elif indicator == 'total_gamma_exposure':
                norm_value = min(abs(value) / 1e6 * 10, 100)  # Scala appropriata
                details['gamma_score'] = norm_value
                score += norm_value * weight
        
        # Aggiungi rischio di flash crash se gamma negativo e skew alto
        if indicators_dict.get('total_gamma_exposure', 0) < 0 and indicators_dict.get('skew_25d', 0) > 0.05:
            score += 20
            details['flash_crash_risk'] = "HIGH"
        else:
            details['flash_crash_risk'] = "LOW"
        
        return {
            'total_score': min(score, 100),
            'risk_level': 'HIGH' if score > 70 else 'MEDIUM' if score > 40 else 'LOW',
            'score_details': details,
            'timestamp': pd.Timestamp.now()
        }
