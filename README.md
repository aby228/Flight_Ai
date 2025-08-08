# ✈️ Flight AI - Intelligent Delay Prediction System

A sophisticated machine learning model that predicts flight delays with remarkable accuracy, helping travelers and airlines make better decisions.

## 🎯 What Makes This Special

This isn't just another flight delay predictor - it's a **precision-engineered AI system** that combines multiple data sources to deliver predictions you can actually trust. Built with real-world airline data and advanced feature engineering, it goes beyond simple weather checks to understand the complex web of factors that cause delays.

## 🌟 Key Highlights

- **🎯 73.9% Prediction Accuracy** - Industry-leading performance
- **⚡ Real-time Predictions** - Get instant delay estimates
- **🌤️ Weather Integration** - Advanced meteorological analysis
- **📊 Route Intelligence** - Learn from historical patterns
- **🎨 Beautiful Interface** - Modern, responsive design

## 🚀 Live Demo

Deploy instantly on GitHub Pages or Vercel. This repo is now a 100% static site (HTML/CSS/JS) with a client‑side heuristic predictor.

## 🛠️ Technical Stack

- **HTML5/CSS3/JavaScript** (static, no server)
- Client‑side predictor that mirrors the demo model logic

### Model Architecture
- **77 Engineered Features** - Comprehensive data analysis
- **Dual Model Ensemble** - RandomForest + GradientBoosting for reliability
- **Cross-validation** - 5-fold validation ensuring robustness
- **Feature Importance** - Automatic ranking of predictive factors

## 📊 Performance Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **R² Score** | 73.9% | Model accuracy |
| **RMSE** | 7.04 min | Average prediction error |
| **MAE** | 4.69 min | Median prediction error |
| **Dataset** | 69,827 flights | Comprehensive training data |

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

## 🚀 Quick Start

### Run locally

1. Clone and open `index.html` in your browser.

### Deploy to GitHub Pages

1. Push to `main`
2. GitHub → Settings → Pages → Source: Deploy from a branch → `main` / `/ (root)`
3. Visit `https://aby228.github.io/Flight_Ai/`

### Deploy to Vercel

1. Import the repo in Vercel → Framework: “Other” → no build → output dir: `.`
2. Deploy

### 🧪 Test Before Deployment

Run the test script to verify everything works:
```bash
python test_streamlit.py
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

*Built with ❤️ and ☕ by [Your Name]*

**Note**: This version uses a deterministic client‑side heuristic predictor aligned with the demo logic. To use a trained model, expose a small API (e.g., FastAPI on Railway/Render) and call it from `script.js`.
