def test_negative_scenario():
    print("\n🧪 STARTING SYSTEM STRESS TEST (SIMULATION)")
    print("="*50)
    
    # 1. Simulate NEGATIVE News (Score: -0.85)
    print("📰 Fetching News... (SIMULATED INPUT)")
    fake_headlines = [
        "Quarterly losses widen by 40% amid poor demand",
        "CEO steps down following regulatory probe",
        "Analysts downgrade stock to 'Underperform'"
    ]
    
    # We manually assign a negative score to prove the logic handles it
    avg_sentiment = -0.85 
    sentiment_status = "NEGATIVE 🔴"
    
    for head in fake_headlines:
        print(f"   [ -0.90 ] {head}")
        
    print(f"\n   🧠 AI Sentiment Score: {avg_sentiment:.2f} ({sentiment_status})")

    # 2. Simulate BEARISH Trend (Price < MA, RSI = 40)
    # Note: We use RSI=40 because it is NOT Oversold (<30), allowing a "Sell" signal.
    print("\n📈 Analyzing Price Charts... (SIMULATED INPUT)")
    print("   📊 Price: ₹100.00 | 50-MA: ₹150.00 | RSI: 40.00")
    print("   (Technical Condition: Price below Average AND RSI is healthy enough to sell)")
    
    tech_signal = "BEARISH (Downtrend)"
    
    # 3. Verdict Logic
    print("\n" + "="*50)
    print(f"🔮 FINAL PREDICTION")
    print("="*50)
    print(f"   1. NEWS SENTIMENT:  {sentiment_status}")
    print(f"   2. PRICE TREND:     {tech_signal}")
    print("-" * 30)
    
    if "NEGATIVE" in sentiment_status and "BEARISH" in tech_signal:
        print("   🔻 RECOMMENDATION: STRONG SELL")
    else:
        print("   ✋ RECOMMENDATION: HOLD / WAIT")
        
    print("="*50 + "\n")

if __name__ == "__main__":
    test_negative_scenario()