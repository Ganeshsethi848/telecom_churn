import streamlit as st
import torch
import torch.nn as nn
import numpy as np

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Telecom Churn AI",
    page_icon="📡",
    layout="wide"
)

# -------------------------------
# Model Definition (MATCH TRAINING)
# -------------------------------
class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.model=nn.Sequential(nn.Linear(15,32),
                            nn.ReLU(),
                            nn.Linear(32,64),
                            nn.ReLU(),
                            nn.Linear(64,128),
                            nn.ReLU(),
                            nn.Linear(128,2))
        
    def forward(self,x):
            
        return self.model(x)

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    model = MLP()
    model.load_state_dict(torch.load("telecom_data_model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# -------------------------------
# Encoding Maps
# -------------------------------
binary_map = {'No': 0, 'Yes': 1}

multiple_lines_map = {
    'No phone service': 0,
    'No': 1,
    'Yes': 2
}

internet_service_map = {
    'DSL': 0,
    'Fiber optic': 1,
    'No': 2
}

no_internet_map = {
    'No': 0,
    'Yes': 1,
    'No internet service': 2
}

contract_map = {
    'Month-to-month': 0,
    'One year': 1,
    'Two year': 2
}

payment_map = {
    'Electronic check': 0,
    'Mailed check': 1,
    'Bank transfer (automatic)': 2,
    'Credit card (automatic)': 3
}

# -------------------------------
# Title
# -------------------------------
st.title("📡 AI Telecom Customer Churn Prediction")

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("📋 Customer Information")

PhoneService = st.sidebar.selectbox("Phone Service", ['No', 'Yes'])

MultipleLines = st.sidebar.selectbox(
    "Multiple Lines",
    ['No phone service', 'No', 'Yes']
)

InternetService = st.sidebar.selectbox(
    "Internet Service",
    ['DSL', 'Fiber optic', 'No']
)

OnlineSecurity = st.sidebar.selectbox(
    "Online Security",
    ['No', 'Yes', 'No internet service']
)

OnlineBackup = st.sidebar.selectbox(
    "Online Backup",
    ['No', 'Yes', 'No internet service']
)

DeviceProtection = st.sidebar.selectbox(
    "Device Protection",
    ['No', 'Yes', 'No internet service']
)

TechSupport = st.sidebar.selectbox(
    "Tech Support",
    ['No', 'Yes', 'No internet service']
)

StreamingTV = st.sidebar.selectbox(
    "Streaming TV",
    ['No', 'Yes', 'No internet service']
)

StreamingMovies = st.sidebar.selectbox(
    "Streaming Movies",
    ['No', 'Yes', 'No internet service']
)

Contract = st.sidebar.selectbox(
    "Contract",
    ['Month-to-month', 'One year', 'Two year']
)

PaperlessBilling = st.sidebar.selectbox(
    "Paperless Billing",
    ['Yes', 'No']
)

PaymentMethod = st.sidebar.selectbox(
    "Payment Method",
    [
        'Electronic check',
        'Mailed check',
        'Bank transfer (automatic)',
        'Credit card (automatic)'
    ]
)

tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
MonthlyCharges = st.sidebar.slider("Monthly Charges", 0.0, 200.0, 70.0)
TotalCharges = st.sidebar.slider("Total Charges", 0.0, 10000.0, 2000.0)

# -------------------------------
# Encode Function
# -------------------------------
def encode():
    features = np.array([
        tenure,
        binary_map[PhoneService],
        multiple_lines_map[MultipleLines],
        internet_service_map[InternetService],
        no_internet_map[OnlineSecurity],
        no_internet_map[OnlineBackup],
        no_internet_map[DeviceProtection],
        no_internet_map[TechSupport],
        no_internet_map[StreamingTV],
        no_internet_map[StreamingMovies],
        contract_map[Contract],
        binary_map[PaperlessBilling],
        payment_map[PaymentMethod],
        MonthlyCharges,
        TotalCharges
    ], dtype=np.float32)

    return torch.tensor(features).unsqueeze(0)

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔍 Predict Churn"):

    x = encode()

    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits,dim=1)
        pred= torch.argmax(prob,dim=1).item()
        print(pred)

    # -------------------------------
    # Output UI
    # -------------------------------
    st.subheader("📊 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        if pred == 1:
            st.error("⚠️ Customer Likely to Churn")
        else:
            st.success("✅ Customer Likely to Stay")

    with col2:
        st.metric("Churn Probability:", f"{prob[0][pred]:.2f}")

    # Progress bar
    st.progress(float(prob[0][pred]))

    # -------------------------------
    # Insights
    # -------------------------------
    st.markdown("### 🧠 AI Insight")

    if pred == 1:
        st.write("""
        - High churn risk detected  
        - Offer discounts or retention plans  
        - Improve service quality  
        """)
    else:
        st.write("""
        - Customer is stable  
        - Opportunity for upselling  
        - Maintain engagement  
        """)

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("⚡ Built with PyTorch + Streamlit")
