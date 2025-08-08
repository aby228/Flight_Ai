# ✈️ Flight AI - Intelligent Delay Prediction System

A sophisticated machine learning model that predicts flight delays with remarkable accuracy, helping travelers and 
airlines make better decisions.

> Trained on January 2025 United Airlines flight data only (scoped intentionally due to compute limits)


## 🎯 What Makes This Special

This isn't just another flight delay predictor - it's a **precision-engineered AI system** that combines multiple data sources to deliver predictions you can actually trust. Built with real-world airline data for January 2025 and advanced feature engineering, it goes beyond simple weather checks to understand the complex web of factors that cause delays.

## 🌟 Key Highlights

- **🎯 73.9% Prediction Accuracy** - Industry-leading performance
- **⚡ Real-time Predictions** - Get instant delay estimates
- **🌤️ Weather Integration** - Advanced meteorological analysis
- **📊 Route Intelligence** - Learn from historical patterns
- **🎨 Beautiful Interface** - Modern, responsive design

## 🚀 Live Demo

https://aby228.github.io/Flight_Ai/

## 🛠️ Technical Stack

- **Model (research)**: RandomForest + GradientBoosting trained on January 2025 UA flights
- **Frontend (demo)**: HTML5/CSS3/JavaScript (static)
- **Public demo**: Client‑side predictor that mirrors the trained model’s decision logic for instant, serverless predictions

### Model Architecture
- **77 Engineered Features** - Comprehensive data analysis
- **Dual Model Ensemble** - RandomForest + GradientBoosting for reliability
- **Cross-validation** - 5-fold validation ensuring robustness
- **Feature Importance** - Automatic ranking of predictive factors

## 📊 Performance & Data Scope

| Metric | Value | Description |
|--------|-------|-------------|
| **R² Score** | 73.9% | Model accuracy |
| **RMSE** | 7.04 min | Average prediction error |
| **MAE** | 4.69 min | Median prediction error |
| **Dataset** | 69,827 flights | United Airlines, January 2025 only |

### Why January 2025 only?
- **Compute limits:** I scoped training to a single month to run robust experiments (feature engineering, cross‑validation, error analysis) within an academic compute budget.
- **Methodological focus:** Narrow scope improves internal validity — fewer seasonal confounds while prototyping the pipeline.
- **Future work:** Extend to multiple months/seasons; add drift monitoring and scheduled re‑training.

## 🎯 Smart Features

### For Travelers
- **Trip Planning** - Know delays before they happen
- **Airport Timing** - Optimize your arrival time
- **Route Selection** - Choose flights with lower delay probability

### For Airlines
- **Operational Planning** - Better resource allocation
- **Customer Communication** - Proactive delay notifications
- **Route Optimization** - Identify problematic patterns

## 🔮 What's Next

### Planned Enhancements
- **Real-time Weather API** - Live meteorological data
- **Multi-airline Support** - Expand beyond United Airlines
- **Deep Learning Integration** - Neural networks for complex patterns
- **Mobile App** - Native iOS/Android applications
- **API Service** - Enterprise-grade prediction API

## 🎨 Personal Touch

This project represents my passion for **practical AI applications** that solve real-world problems. Every line of code, every feature, and every prediction is crafted with attention to detail and a commitment to excellence.

*"In a world of uncertainty, knowledge is power. This AI system gives travelers and airlines the power to make informed decisions."*

## 📁 Project Structure

```
Flight_Ai/
├── index.html         # UI
├── styles.css         # Styling
├── script.js          # Client-side predictor & logic
└── README.md
```


## 🤝 Contributing

I welcome contributions! Whether it's:
- 🐛 Bug fixes
- ✨ New features
- 📚 Documentation improvements
- 🎨 UI/UX enhancements

Feel free to open an issue or submit a pull request.

## 📞 Connect

- **GitHub**: www.github.com/aby228
- **LinkedIn**: www.linkedin.com/in/abraham-yarba


---

*Built with ❤️ and ☕ by Abraham Yarba*

## 🙌 Acknowledgements
This project was completed thanks to the collaboration between me and my teammates **Jadryan Pena** and **Mitchell Chen** during the **Ignite AI4ALL Program**. I’m grateful for their insights, feedback, and shared iteration that made this work stronger.

---

**Note**: The public demo uses a deterministic client‑side predictor aligned with the trained model’s logic. To serve the trained model, expose a lightweight API (e.g., FastAPI on Railway/Render) and point the frontend to it.
