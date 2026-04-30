import logging
import pandas as pd
from typing import List, Dict, Any, Tuple

from ..data.point_in_time import PointInTimeContext
from ..data.price_loader import PriceLoader
from ..data.fundamental_loader import FundamentalLoader

logger = logging.getLogger(__name__)

class HardScreener:
    """
    Implements Phase 4: The Cost Control Gate.
    Filters out weak companies programmatically using strict rules 
    before wasting expensive LLM API calls on them.
    """
    
    def __init__(self, price_loader: PriceLoader, fundamental_loader: FundamentalLoader):
        self.price_loader = price_loader
        self.fundamental_loader = fundamental_loader
        
        # Hard Rule Institutional Thresholds
        self.min_price = 10.0
        self.min_adv = 1_000_000 # 1 Million shares average daily volume
        self.require_positive_fcf = True
        self.require_positive_net_income = True
        self.require_above_200dma = True
        self.require_golden_cross = True # 50 DMA > 200 DMA

    def calculate_technical_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculates 200DMA and other technicals."""
        if len(df) < 200:
            return {"error": "Insufficient price history for 200DMA"}
            
        current_price = df.iloc[-1]['Close']
        dma_200 = df['Close'].rolling(window=200).mean().iloc[-1]
        dma_50 = df['Close'].rolling(window=50).mean().iloc[-1]
        adv_20 = df['Volume'].rolling(window=20).mean().iloc[-1]
        
        return {
            "current_price": float(current_price),
            "dma_200": float(dma_200),
            "dma_50": float(dma_50),
            "adv_20": float(adv_20),
            "is_above_200dma": current_price > dma_200,
            "is_golden_cross": dma_50 > dma_200
        }

    def evaluate_ticker(self, ticker: str, pit_context: PointInTimeContext) -> Dict[str, Any]:
        """
        Evaluates a single ticker against the hard rules.
        """
        logger.info(f"Screening {ticker}...")
        
        # 1. Fetch Data
        price_df = self.price_loader.fetch_daily_prices(ticker, pit_context)
        fundamentals = self.fundamental_loader.fetch_fundamentals(ticker, pit_context)
        
        # 2. Check Data Integrity
        if price_df.empty:
            return {"ticker": ticker, "passed": False, "reason": "No price data"}
        if "error" in fundamentals:
            return {"ticker": ticker, "passed": False, "reason": f"Fundamental Error: {fundamentals['error']}"}
            
        # 3. Technical Evaluation
        techs = self.calculate_technical_metrics(price_df)
        if "error" in techs:
            return {"ticker": ticker, "passed": False, "reason": techs["error"]}
            
        # 4. Apply Hard Institutional Rules
        if techs["current_price"] < self.min_price:
            return {"ticker": ticker, "passed": False, "reason": f"Price below ${self.min_price}"}
            
        if techs["adv_20"] < self.min_adv:
            return {"ticker": ticker, "passed": False, "reason": f"Illiquid (ADV < 1M)"}
            
        if self.require_above_200dma and not techs["is_above_200dma"]:
            return {"ticker": ticker, "passed": False, "reason": "Downtrend (Below 200DMA)"}
            
        if self.require_golden_cross and not techs["is_golden_cross"]:
            return {"ticker": ticker, "passed": False, "reason": "No Golden Cross (50DMA < 200DMA)"}
            
        if self.require_positive_fcf and fundamentals.get("free_cash_flow", 0) <= 0:
            return {"ticker": ticker, "passed": False, "reason": "Negative Free Cash Flow"}
            
        if self.require_positive_net_income and fundamentals.get("net_income", 0) <= 0:
            return {"ticker": ticker, "passed": False, "reason": "Negative Net Income"}
            
        # If it survives all checks
        return {
            "ticker": ticker, 
            "passed": True, 
            "metrics": {
                "price": techs["current_price"],
                "dma_200": techs["dma_200"],
                "adv": techs["adv_20"],
                "fcf": fundamentals["free_cash_flow"],
                "net_income": fundamentals["net_income"]
            }
        }

    def run_screen(self, universe: List[str], pit_context: PointInTimeContext) -> Tuple[List[str], pd.DataFrame]:
        """
        Runs the full universe through the hard screener.
        Returns a tuple: (List of passed tickers, DataFrame of full results).
        """
        passed_tickers = []
        detailed_results = []
        
        print(f"\n--- Initiating Hard Screener as of {pit_context.analysis_date} ---")
        print(f"Scanning {len(universe)} tickers...")
        
        for ticker in universe:
            result = self.evaluate_ticker(ticker, pit_context)
            if result["passed"]:
                print(f"[PASS] {ticker} (Price: ${result['metrics']['price']:.2f}, FCF: {result['metrics']['fcf']})")
                passed_tickers.append(ticker)
                detailed_results.append({
                    "Ticker": ticker,
                    "Status": "✅ PASS",
                    "Reason": "Met all criteria",
                    "Price": f"${result['metrics']['price']:.2f}",
                    "Vol (ADV)": f"{result['metrics']['adv']/1e6:.1f}M",
                    "200 DMA": f"${result['metrics']['dma_200']:.2f}",
                    "FCF": f"${result['metrics']['fcf']/1e6:.1f}M",
                    "Net Inc": f"${result['metrics']['net_income']/1e6:.1f}M"
                })
            else:
                print(f"[FAIL] {ticker}: {result['reason']}")
                detailed_results.append({
                    "Ticker": ticker,
                    "Status": "❌ FAIL",
                    "Reason": result['reason'],
                    "Price": "N/A",
                    "Vol (ADV)": "N/A",
                    "200 DMA": "N/A",
                    "FCF": "N/A",
                    "Net Inc": "N/A"
                })
                
        print(f"--- Screening Complete: {len(passed_tickers)} / {len(universe)} passed ---\n")
        df_results = pd.DataFrame(detailed_results)
        return passed_tickers, df_results
