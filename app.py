import streamlit as st
import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit import RDLogger
import warnings

warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')

st.set_page_config(
    page_title="ADMET Toxicity Predictor",
    page_icon="🧪",
    layout="wide"
)

TASKS = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase',
    'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma',
    'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
]

EXAMPLES = {
    "Ethanol (safe)": "CCO",
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "Aflatoxin B1 (toxic)": "O=c1oc2c(OC)cc3c(c2c2c1[C@@H]1C=C[C@@H](O1)O2)OCO3",
}

if 'smiles' not in st.session_state:
    st.session_state.smiles = "CCO"


def set_example(smi):
    st.session_state.smiles = smi


@st.cache_resource
def load_model():
    return joblib.load('model/multitask_xgb.pkl')


def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
    return mol, np.array(fp).reshape(1, -1)


# ── UI ──────────────────────────────────────────────────────────
st.title("🧪 ADMET Toxicity Predictor")
st.markdown("Predict molecular toxicity across **12 Tox21 endpoints** using XGBoost + Morgan fingerprints.")
st.divider()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input molecule")
    smiles_input = st.text_input(
        "SMILES string",
        key="smiles",
        help="e.g. CCO (ethanol), CC(=O)Oc1ccccc1C(=O)O (aspirin)"
    )

    st.markdown("**Try an example:**")
    for name, smi in EXAMPLES.items():
        st.button(name, on_click=set_example, args=(smi,))

with col2:
    mol, fp = smiles_to_fp(smiles_input)
    if mol is None:
        st.error("❌ Invalid SMILES — please check your input.")
    else:
        st.subheader("Structure")
        img = Draw.MolToImage(mol, size=(300, 200))
        st.image(img, caption=smiles_input)

st.divider()

# ── Predictions ─────────────────────────────────────────────────
if smiles_input and mol is not None:
    try:
        models = load_model()

        st.subheader("Toxicity predictions")
        st.markdown("*Probability of toxicity per endpoint — 🟢 safe · 🟡 moderate · 🔴 high risk*")

        cols = st.columns(3)
        high_risk = []

        for i, task in enumerate(TASKS):
            prob = models[task].predict_proba(fp)[0, 1]

            if prob > 0.7:
                icon = "🔴"
                label = "HIGH RISK"
                high_risk.append(task)
            elif prob > 0.4:
                icon = "🟡"
                label = "MODERATE"
            else:
                icon = "🟢"
                label = "LOW RISK"

            with cols[i % 3]:
                st.metric(
                    label=f"{icon} {task}",
                    value=f"{prob:.1%}",
                    delta=label,
                    delta_color="off"
                )

        st.divider()

        if high_risk:
            st.error(f"⚠️ High toxicity risk flagged for: {', '.join(high_risk)}")
        else:
            st.success("✅ No high toxicity signals detected across 12 endpoints")

    except FileNotFoundError:
        st.warning("⚠️ Model not found. Run notebook 02 first to train and save the model.")

st.divider()
st.caption("""
Built on Tox21 dataset (7,831 compounds, 12 endpoints) ·
XGBoost with Morgan fingerprints (ECFP4, radius=2, 1024 bits) ·
Mean AUC: 0.790 · [GitHub](https://github.com/cecilesde/admet-toxicity-predictor)
""")
