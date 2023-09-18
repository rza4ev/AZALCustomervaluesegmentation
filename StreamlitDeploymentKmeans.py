import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from PIL import Image
import io

# Loading csv and pkl file then create a streamlit web service
filename = 'final_KMEANS.pkl'
loaded_model = pickle.load(open(filename, 'rb'))
df = pd.read_csv("F_KMEANS.csv")
st.set_option('deprecation.showPyplotGlobalUse', False)
st.markdown('<style>body{background-color: Blue;}</style>', unsafe_allow_html=True)
st.title("AZAL Customer Values Segmentation and Clustering")  # Başlık eklenen satır
# CREATING FORMS
with st.form("my_form"):
    Length = st.number_input('Length', min_value=0.0, max_value=10.0)
    Recency = st.number_input('Recency', min_value=0.0, max_value=10.0)
    Frequency = st.number_input('Frequency', min_value=0.0, max_value=10.0)
    MONETARY = st.number_input('MONETARY', min_value=0.0, max_value=10.0)
    data = [[Length, Recency, Frequency, MONETARY]]
    submitted = st.form_submit_button("Submit")
if submitted and all(value == 0.0 for value in [Length, Recency, Frequency, MONETARY]):
    st.write("Incorrect values. Please enter valid values.")
else:
    # VIZUALISATION PREDICTING CLUSTER
    if submitted:
        clust = loaded_model.predict(data)[0]
        st.write('Data Belongs to Cluster', clust)

        # Her bir küme için farklı açıklamaları ekrana yazdırın
        if clust == 0:
            st.write('General Customers')
            st.write("Average-Length Values: General customers typically have an average length of engagement with your business.")
            st.write("High Recency Values: They may have recently interacted with your business, but it's not a frequent occurrence.")
            st.write("Low Frequency Values: These customers don't make purchases or interact with your business very often.")
            st.write("Low Monetary Values: They tend to spend relatively less compared to other segments.")
        elif clust == 1:
            st.write('Loyal Customers')
            st.write("High Length Values: Loyal customers have a long history of engagement with your business, indicating their loyalty.")
            st.write("Low Recency Values: They interact with your business frequently, showing consistent engagement.")
            st.write("Low Frequency Values: Despite their loyalty, their purchase frequency might not be very high.")
            st.write("Average Monetary Values: They spend an average amount per transaction, reflecting their consistent support.")
        elif clust == 2:
            st.write('Potential Customers')
            st.write("All Values Are Average: Potential customers fall into the middle ground for all the characteristics.")
            st.write("They have an average length of engagement, recency, frequency, and spending.")
            st.write("This segment may represent customers who are still in the early stages of their relationship with your business and have the potential to become more loyal or valuable over time.")
        elif clust == 3:
            st.write('Important Customers')
            st.write("High Length Values: These customers have been engaged with your business for a long time.")
            st.write("Low Recency Values: They may not have interacted with your business recently, but their history of engagement is significant.")
            st.write("High Frequency Values: They make frequent purchases or interactions with your business.")
            st.write("High Monetary Values: Important customers are also high spenders, contributing significantly to your revenue.")

        df_c = df[df['K_Cluster'] == clust]

        # Her bir özellik için ayrı bir subplot oluşturun
        for c in df_c.drop(['K_Cluster'], axis=1):
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(data=df_c, x=c, kde=True, ax=ax)
            plt.title(f'Cluster {clust} - {c}')
            st.pyplot(fig)




