# ui/app.py – Giao diện nhiều trang (Trang chọn mô hình + Trang thao tác)
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from module import apriori_custom
from module.naive_bayes_custom import NaiveBayesClassifier
from module.rough_set_custom import RoughSet

st.set_page_config(page_title="Khai phá dữ liệu", layout="wide")

# --- Khởi tạo session state ---
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# --- Trang chọn mô hình ---
def show_home():
    st.markdown("<h1 style='text-align: center;'>📚 Chọn mô hình khai phá dữ liệu</h1>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📊 Apriori (Luật kết hợp)", use_container_width=True):
            st.session_state.page = 'apriori'

    with col2:
        if st.button("🧠 Naive Bayes (Phân lớp)", use_container_width=True):
            st.session_state.page = 'bayes'
    
    with col3:
        if st.button("📘 Tập thô (Rough Set)", use_container_width=True):
            st.session_state.page = 'rough'

# --- Trang Apriori ---
def show_apriori():
    st.button("⬅️ Quay lại menu", on_click=lambda: st.session_state.update(page='home'))
    st.header("📊 Luật kết hợp - Apriori hoặc Không tăng cường")
    uploaded_file = st.file_uploader("📂 Tải file CSV", type=["csv"], key="apriori_upload")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df)
        event_col = st.selectbox("🔹 Cột giao dịch:", df.columns)
        item_col = st.selectbox("🔸 Cột mặt hàng:", df.columns)

        st.radio("⚙️ Chọn thuật toán:", options=["Apriori", "Không tăng cường"], key="algo")

        min_sup = st.number_input("📏 Min Support (0–1):", min_value=0.0, max_value=1.0, value=0.4, step=0.01)
        min_conf = st.number_input("📐 Min Confidence (0–1):", min_value=0.0, max_value=1.0, value=0.6, step=0.01)

        if st.button("🚀 Chạy luật kết hợp"):
            df_temp = df[[event_col, item_col]].dropna()
            df_temp.columns = ['Invoice', 'Item']
            transactions = apriori_custom.create_transactions(df_temp)

            freq_items = apriori_custom.find_frequent_itemsets(transactions, min_sup)
            rules = apriori_custom.generate_rules(freq_items, transactions, min_conf)

            st.subheader("✅ Tập phổ biến:")
            if not freq_items.empty:
                freq_items['itemsets'] = freq_items['itemsets'].apply(lambda x: ', '.join(sorted(list(x))))
                st.dataframe(freq_items)
                csv_freq = freq_items.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Tải tập phổ biến", csv_freq, file_name="frequent_itemsets.csv", mime="text/csv")

                # 📊 Biểu đồ bar Support
                st.subheader("📊 Biểu đồ Support:")
                fig, ax = plt.subplots()
                top_freq = freq_items.sort_values('support', ascending=False).head(15)
                ax.barh(top_freq['itemsets'], top_freq['support'])
                ax.set_xlabel('Support')
                ax.invert_yaxis()
                st.pyplot(fig)
            else:
                st.warning("⚠️ Không có tập phổ biến nào.")

            st.subheader("📐 Luật kết hợp:")
            if not rules.empty:
                rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(sorted(list(x))))
                rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(sorted(list(x))))
                st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence']])

                csv_rules = rules.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Tải luật kết hợp", csv_rules, file_name="association_rules.csv", mime="text/csv")

                # 📈 Biểu đồ scatter Support vs Confidence
                st.subheader("📈 Biểu đồ Support vs Confidence:")
                fig2, ax2 = plt.subplots()
                ax2.scatter(rules['support'], rules['confidence'], alpha=0.6)
                ax2.set_xlabel('Support')
                ax2.set_ylabel('Confidence')
                ax2.set_title('Scatter plot: Support vs Confidence')
                st.pyplot(fig2)
            else:
                st.warning("⚠️ Không tìm thấy luật nào.")

# --- Trang Naive Bayes ---
def show_bayes():
    st.button("⬅️ Quay lại menu", on_click=lambda: st.session_state.update(page='home'))
    st.header("🧠 Naive Bayes - Phân lớp")
    uploaded_file = st.file_uploader("📂 Tải file CSV", type=["csv"], key="bayes_upload")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df)
        target_col = st.selectbox("🎯 Cột mục tiêu:", df.columns)
        input_values = {}
        feature_cols = [col for col in df.columns if col != target_col]

        for col in feature_cols:
            input_values[col] = st.selectbox(f"🔹 {col}", df[col].unique())

        if st.button("🚀 Dự đoán"):
            clf = NaiveBayesClassifier()
            clf.fit(df, target_col)
            predicted_class, log_scores = clf.predict(input_values)

            st.success(f"✅ Dự đoán: `{predicted_class}`")
            st.json({k: round(v, 4) for k, v in log_scores.items()})

# ---Trang Rough Set---

def show_rough_set():
    st.button("⬅️ Quay lại menu", on_click=lambda: st.session_state.update(page='home'))
    st.header("📘 Tập thô – Rough Set")

    uploaded_file = st.file_uploader("📂 Tải file CSV", type=["csv"], key="rough_upload")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df)

        decision_col = st.selectbox("🎯 Chọn cột quyết định (Decision Attribute):", df.columns)
        condition_cols = st.multiselect("🔹 Chọn các cột điều kiện (Condition Attributes):", [col for col in df.columns if col != decision_col])
        func = st.radio("⚙️ Chọn chức năng:", ["Xấp xỉ (Lower/Upper)", "Phụ thuộc thuộc tính", "Tìm reduct", "Sinh luật chính xác 100%"])

        if st.button("🚀 Thực hiện"):
            rs = RoughSet(df, condition_cols, decision_col)

            if func == "Xấp xỉ (Lower/Upper)":
                val = st.selectbox("🧪 Chọn giá trị quyết định:", df[decision_col].unique())
                lower = rs.lower_approx(val)
                upper = rs.upper_approx(val)

                st.write(f"📥 Lower approximation của `{val}` ({len(lower)} dòng):", sorted(lower))
                st.write(f"📤 Upper approximation của `{val}` ({len(upper)} dòng):", sorted(upper))

            elif func == "Phụ thuộc thuộc tính":
                degree = rs.dependency_degree()
                st.success(f"📊 Mức độ phụ thuộc: `{round(degree, 4)}`")

            elif func == "Tìm reduct":
                reduct = rs.find_reduct()
                st.success(f"🔍 Reduct tìm được: `{', '.join(reduct)}`")

            elif func == "Sinh luật chính xác 100%":
                rules = rs.generate_rules()
                if rules:
                    st.subheader(f"📜 {len(rules)} luật chính xác 100%:")
                    for i, r in enumerate(rules, 1):
                        cond = ' ∧ '.join([f"{k}={v}" for k, v in r['conditions'].items()])
                        st.write(f"**Luật {i}:** Nếu {cond} → {decision_col} = {r['decision']}")
                    # Cho phép tải
                    rule_df = pd.DataFrame([{
                        **r['conditions'], '=>': f"{decision_col}={r['decision']}"
                    } for r in rules])
                    csv = rule_df.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Tải các luật về CSV", csv, file_name="rough_rules.csv", mime='text/csv')
                else:
                    st.warning("⚠️ Không sinh được luật nào.")

# --- Hiển thị trang tương ứng ---
if st.session_state.page == 'home':
    show_home()
elif st.session_state.page == 'apriori':
    show_apriori()
elif st.session_state.page == 'bayes':
    show_bayes()
elif st.session_state.page == 'rough':
    show_rough_set()