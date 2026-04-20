# # import streamlit as st
# # import cv2
# # from src.inference.pipeline import run_pipeline

# # st.set_page_config(layout="wide")
# # st.title("Hybrid Oil Spill Detection System")

# # uploaded = st.file_uploader("Upload SAR Image", type=["png", "jpg", "jpeg"])

# # if uploaded:
# #     with open("temp.png", "wb") as f:
# #         f.write(uploaded.read())

# #     original, dl_pred, final_pred, decision = run_pipeline("temp.png")

# #     col1, col2, col3 = st.columns(3)

# #     with col1:
# #         st.image(
# #             cv2.cvtColor(original, cv2.COLOR_BGR2RGB),
# #             caption="Original SAR Image",
# #             use_container_width=True
# #         )

# #     with col2:
# #         st.image(
# #             cv2.cvtColor(dl_pred, cv2.COLOR_BGR2RGB),
# #             caption="DL Prediction (Potential Oil Regions)",
# #             use_container_width=True
# #         )

# #     with col3:
# #         st.image(
# #             cv2.cvtColor(final_pred, cv2.COLOR_BGR2RGB),
# #             caption="Final Prediction (ML Verified Oil)",
# #             use_container_width=True
# #         )

# #     st.markdown("---")

# #     if decision == "OIL DETECTED":
# #         st.success("🛢️ **FINAL DECISION: OIL IS PRESENT IN THE IMAGE**")
# #     else:
# #         st.info("🌊 **FINAL DECISION: NO OIL DETECTED IN THE IMAGE**")
# import streamlit as st
# import cv2
# import pandas as pd

# from src.inference.pipeline import run_pipeline
# from src.evaluation.performance_metrics import build_metrics_from_logs

# st.set_page_config(layout="wide")
# st.title("🛢️ Hybrid Oil Spill Detection System")

# uploaded = st.file_uploader("Upload SAR Image", type=["png", "jpg", "jpeg"])

# if uploaded:
#     with open("temp.png", "wb") as f:
#         f.write(uploaded.read())

#     original, dl_pred, final_pred, decision = run_pipeline("temp.png")

#     col1, col2, col3 = st.columns(3)

#     with col1:
#         st.image(cv2.cvtColor(original, cv2.COLOR_BGR2RGB),
#                  caption="Original SAR Image", use_container_width=True)

#     with col2:
#         st.image(cv2.cvtColor(dl_pred, cv2.COLOR_BGR2RGB),
#                  caption="DL Prediction (Potential Oil Regions)", use_container_width=True)

#     with col3:
#         st.image(cv2.cvtColor(final_pred, cv2.COLOR_BGR2RGB),
#                  caption="Final Prediction (ML Verified Oil)", use_container_width=True)

#     st.markdown("---")

#     if decision == "OIL DETECTED":
#         st.success("🛢️ **FINAL DECISION: OIL IS PRESENT IN THE IMAGE**")
#     else:
#         st.info("🌊 **FINAL DECISION: NO OIL DETECTED IN THE IMAGE**")

# st.markdown("---")
# st.subheader("📊 Model Performance Metrics")

# table, cm= build_metrics_from_logs()

# if table is not None:
#     st.markdown("### 🔢 Core Metrics")
#     st.table(table.round(4))

#     st.markdown("### 🧮 Confusion Matrix")
#     cm_df = pd.DataFrame(
#         cm,
#         index=["Actual No Oil", "Actual Oil"],
#         columns=["Predicted No Oil", "Predicted Oil"]
#     )
#     st.table(cm_df)

# else:
#     st.info("Run more labeled samples to generate performance metrics.")
import streamlit as st, cv2, pandas as pd
from src.inference.pipeline import run_pipeline

st.set_page_config(layout="wide")
st.title("Hybrid Oil Spill Detection System")

uploaded = st.file_uploader("Upload SAR Image", type=["png","jpg","jpeg"])

if uploaded:
    with open("temp.png","wb") as f:
        f.write(uploaded.read())

    original, dl_pred, final_pred, decision = run_pipeline("temp.png")

    c1,c2,c3 = st.columns(3)
    c1.image(cv2.cvtColor(original, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)
    c2.image(cv2.cvtColor(dl_pred, cv2.COLOR_BGR2RGB), caption="DL Regions", use_container_width=True)
    c3.image(cv2.cvtColor(final_pred, cv2.COLOR_BGR2RGB), caption="Final ML Output", use_container_width=True)

    st.markdown("---")
    st.success(f"🧪 **FINAL DECISION: {decision}**")

# ---------------- REAL MODEL PERFORMANCE ----------------

st.markdown("---")
st.subheader("📊 Trained Model Performance (Offline Evaluation)")

metrics = pd.read_csv("offline_model_metrics.csv")
cm = pd.read_csv("offline_confusion_matrix.csv").values

st.table(metrics.style.format({"Value": "{:.4f}"}))

cm_df = pd.DataFrame(cm, 
    columns=["Predicted No Oil","Predicted Oil"],
    index=["Actual No Oil","Actual Oil"]
)
st.subheader("🧮 Confusion Matrix")
st.table(cm_df)
