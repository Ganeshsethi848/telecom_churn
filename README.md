# 📡 Telecom Customer Churn Prediction AI

A deep learning-powered web application that predicts customer churn for telecom companies using PyTorch and Streamlit.

**🚀 [Live Demo](https://telecomchurn-kdoqe85qjq8rke47fmqrqe.streamlit.app/)** | **📊 [GitHub Repository](https://github.com/Ganeshsethi848/telecom_churn)**

---

## 🎯 Features

- **AI-Powered Predictions**: Uses a trained neural network (MLP) to predict customer churn probability
- **Interactive Dashboard**: User-friendly Streamlit interface with real-time predictions
- **Customer Insights**: Actionable recommendations based on churn risk assessment
- **Easy Deployment**: Cloud-ready with Streamlit Cloud integration
- **Mobile Responsive**: Works seamlessly on desktop and mobile devices

---

## 📋 Project Overview

This project helps telecom companies identify customers at risk of churning (leaving the service). By analyzing customer behavior patterns and service usage, the AI model provides:

- **Churn Risk Classification**: Binary prediction (Churn / Stay)
- **Probability Score**: Confidence level of the prediction
- **Business Recommendations**: Tailored strategies for customer retention

---

## 🔧 Installation

### Prerequisites
- Python 3.8+
- pip

### Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Ganeshsethi848/telecom_churn.git
   cd telecom_churn
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**
   ```bash
   streamlit run telecom_test.py
   ```

4. **Access the app**
   - Open your browser to `http://localhost:8501`

---

## 🎮 Usage

### Local Deployment
```bash
streamlit run telecom_test.py
```

### Cloud Deployment (Streamlit Cloud)
1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Select `telecom_test.py` as main file
5. Click Deploy

### Using the App
1. **Select Customer Information** from the sidebar:
   - Phone Service (Yes/No)
   - Internet Service Type (DSL/Fiber/None)
   - Service add-ons (Online Security, Tech Support, etc.)
   - Contract type (Month-to-month/1-year/2-year)
   - Payment method
   - Billing information (Monthly/Total Charges)
   - Tenure (months with service)

2. **Click "🔍 Predict Churn"** to get predictions

3. **View Results**:
   - Churn probability
   - Risk assessment
   - Recommended actions

---

## 🏗️ Model Architecture

The model is a **Multi-Layer Perceptron (MLP)** neural network:

```
Input Layer (15 features)
    ↓
Dense(15 → 32) + ReLU
    ↓
Dense(32 → 64) + ReLU
    ↓
Dense(64 → 128) + ReLU
    ↓
Dense(128 → 2) + Softmax
    ↓
Output (Churn probability)
```

**Framework**: PyTorch
**Input Features**: 15 customer attributes
**Output**: Binary classification (Churn/Stay) with probability scores

---

## 📁 Project Structure

```
telecom_churn/
├── telecom_test.py           # Main Streamlit application
├── telecom_data_model.pth    # Pre-trained PyTorch model
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── .gitignore
```

---

## 📊 Feature Engineering

The model uses the following 15 features:

1. **Tenure** - Months with service
2. **Phone Service** - Has phone service (0/1)
3. **Multiple Lines** - Multiple phone lines (0/1/2)
4. **Internet Service** - Type of internet service (0/1/2)
5. **Online Security** - Has online security (0/1/2)
6. **Online Backup** - Has online backup (0/1/2)
7. **Device Protection** - Has device protection (0/1/2)
8. **Tech Support** - Has tech support (0/1/2)
9. **Streaming TV** - Has streaming TV (0/1/2)
10. **Streaming Movies** - Has streaming movies (0/1/2)
11. **Contract Type** - Contract term (0/1/2)
12. **Paperless Billing** - Uses paperless billing (0/1)
13. **Payment Method** - Payment type (0/1/2/3)
14. **Monthly Charges** - Monthly billing amount
15. **Total Charges** - Total accumulated charges

---

## 🛠️ Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'torch'"
**Solution**: Install dependencies
```bash
pip install -r requirements.txt
```

### Issue: "FileNotFoundError: telecom_data_model.pth"
**Solution**: Ensure the model file is in the project root directory and committed to Git

### Issue: App not loading on Streamlit Cloud
**Solution**: 
- Verify `requirements.txt` is present
- Ensure relative paths (not absolute paths) are used
- Check that all dependencies are listed

### Issue: "RuntimeError: Expected all tensors to be on the same device"
**Solution**: The app uses `map_location="cpu"` - no GPU required

---

## 💡 Key Insights & Recommendations

### High Churn Risk Indicators:
- Month-to-month contracts
- High monthly charges relative to service level
- Lack of service add-ons
- Recent customers (low tenure)
- Electronic payment method

### Retention Strategies:
1. **Offer discounts** for longer-term contracts
2. **Bundle services** to increase perceived value
3. **Proactive support** for at-risk customers
4. **Loyalty programs** for long-term customers
5. **Automatic payment options** (reduces friction)

---

## 📈 Performance Metrics

- **Training Data**: Telecom customer dataset
- **Model Type**: Neural Network (MLP)
- **Input Features**: 15
- **Output Classes**: 2 (Churn/Stay)
- **Framework**: PyTorch

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -m 'Add improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## 📝 License

This project is open source and available under the MIT License.

---

## 📞 Support

For issues, questions, or suggestions:
- Open an [Issue](https://github.com/Ganeshsethi848/telecom_churn/issues)
- Check existing documentation
- Review troubleshooting section above

---

## 🎓 Learning Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [Machine Learning Basics](https://www.coursera.org/learn/machine-learning)
- [Neural Networks](https://www.deeplearningbook.org/)

---

**Built with ❤️ using PyTorch + Streamlit**

⭐ If you found this project helpful, please consider giving it a star on GitHub!
