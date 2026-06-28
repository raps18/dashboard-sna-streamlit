import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ==========================
# KONFIGURASI HALAMAN
# ==========================

st.set_page_config(
    page_title="Dashboard Analisis Komentar MPL ID S17",
    page_icon="📊",
    layout="wide"
)

# ==========================
# CSS
# ==========================

st.markdown("""
<style>

.main{
    background-color:#F8FAFC;
}

[data-testid="stMetric"]{
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 0px 8px rgba(0,0,0,.1);
    text-align:center;
}

h1,h2,h3{
    color:#0F172A;
}

</style>
""", unsafe_allow_html=True)

# ==========================
# LOAD DATA
# ==========================

@st.cache_data
def load_data():

    raw = pd.read_csv("GrandFinal_MPLID_S17.csv")

    toxic = pd.read_csv("mpl_s17_toxic_labeled.csv")

    topic = pd.read_csv("youtube_topics_sample.csv")

    return raw,toxic,topic


raw_df,toxic_df,topic_df = load_data()

# ==========================
# MEMBUAT KOLOM KATEGORI
# ==========================

toxic_df["Kategori"] = toxic_df["label"].replace({
    0:"Non-Toxic",
    1:"Toxic"
})

# ==========================
# SIDEBAR
# ==========================

st.sidebar.title("📊 Dashboard")

menu = st.sidebar.radio(

    "Pilih Menu",

    [

        "🏠 Overview",

        "🔍 Analisis Komentar",

        "🤖 Hasil Klasifikasi",

        "📚 BERTopic",

        "🌐 Social Network Analysis"

    ]

)

# ==========================
# OVERVIEW
# ==========================

if menu=="🏠 Overview":

    st.title("📊 Dashboard Analisis Komentar MPL ID Season 17")

    st.write("Ringkasan hasil analisis komentar YouTube MPL ID Season 17.")

    total=len(toxic_df)

    toxic=len(toxic_df[toxic_df["Kategori"]=="Toxic"])

    non=len(toxic_df[toxic_df["Kategori"]=="Non-Toxic"])

    persen_toxic=round((toxic/total)*100,2)

    persen_non=round((non/total)*100,2)

    c1,c2,c3,c4=st.columns(4)

    c1.metric("Total Komentar",total)

    c2.metric("Toxic",toxic)

    c3.metric("Non Toxic",non)

    c4.metric("% Toxic",f"{persen_toxic}%")

    st.markdown("---")

    col1,col2=st.columns(2)

    with col1:

        st.subheader("Distribusi Toxic")

        data=toxic_df["Kategori"].value_counts().reset_index()

        data.columns=["Kategori","Jumlah"]

        fig=px.bar(

            data,

            x="Kategori",

            y="Jumlah",

            text="Jumlah",

            color="Kategori"

        )

        st.plotly_chart(fig,use_container_width=True)

    with col2:

        st.subheader("Persentase")

        fig2=px.pie(

            data,

            names="Kategori",

            values="Jumlah",

            hole=.45

        )

        st.plotly_chart(fig2,use_container_width=True)

    st.markdown("---")

    st.subheader("Preview Dataset")

    st.dataframe(raw_df.head(10),use_container_width=True)

    st.download_button(

        "📥 Download Dataset",

        raw_df.to_csv(index=False),

        "GrandFinal_MPLID_S17.csv",

        "text/csv"

    )

# =====================================================
# ANALISIS KOMENTAR
# =====================================================

elif menu == "🔍 Analisis Komentar":

    st.title("🔍 Analisis Komentar")

    st.write(
        "Menu ini digunakan untuk mencari komentar dan melihat apakah komentar tersebut termasuk Toxic atau Non-Toxic."
    )

    st.markdown("---")

    # ==========================
    # FILTER
    # ==========================

    col1, col2 = st.columns([1,2])

    with col1:

        pilihan = st.selectbox(

            "Filter Label",

            [

                "Semua",

                "Toxic",

                "Non-Toxic"

            ]

        )

    with col2:

        keyword = st.text_input(

            "Cari Komentar",

            placeholder="Masukkan kata kunci..."

        )

    # ==========================
    # COPY DATA
    # ==========================

    hasil = toxic_df.copy()

    # ==========================
    # FILTER LABEL
    # ==========================

    if pilihan != "Semua":

        hasil = hasil[
            hasil["Kategori"] == pilihan
        ]

    # ==========================
    # FILTER KEYWORD
    # ==========================

    if keyword != "":

        hasil = hasil[
            hasil["text"]
            .astype(str)
            .str.contains(
                keyword,
                case=False,
                na=False
            )
        ]

    st.markdown("---")

    # ==========================
    # KPI
    # ==========================

    total_hasil = len(hasil)

    toxic_hasil = len(
        hasil[
            hasil["Kategori"]=="Toxic"
        ]
    )

    non_hasil = len(
        hasil[
            hasil["Kategori"]=="Non-Toxic"
        ]
    )

    c1,c2,c3 = st.columns(3)

    c1.metric(
        "Jumlah Hasil",
        total_hasil
    )

    c2.metric(
        "Toxic",
        toxic_hasil
    )

    c3.metric(
        "Non Toxic",
        non_hasil
    )

    st.markdown("---")

    # ==========================
    # BAR CHART
    # ==========================

    chart = (
        hasil["Kategori"]
        .value_counts()
        .reset_index()
    )

    chart.columns = [

        "Kategori",

        "Jumlah"

    ]

    fig = px.bar(

        chart,

        x="Kategori",

        y="Jumlah",

        text="Jumlah",

        color="Kategori",

        title="Distribusi Hasil Pencarian"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    # ==========================
    # TABEL
    # ==========================

    tampil = hasil[

        [

            "text",

            "Kategori"

        ]

    ]

    st.dataframe(

        tampil,

        use_container_width=True,

        height=450

    )

    # ==========================
    # DOWNLOAD
    # ==========================

    csv = tampil.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(

        "📥 Download Hasil",

        csv,

        "hasil_pencarian.csv",

        "text/csv"

    )

# =====================================================
# HASIL KLASIFIKASI
# =====================================================

elif menu == "🤖 Hasil Klasifikasi":

    st.title("🤖 Hasil Klasifikasi")

    st.write(
        "Perbandingan performa model Naive Bayes dan Support Vector Machine pada klasifikasi komentar toxic."
    )

    st.markdown("---")

    # ====================================
    # KPI
    # ====================================

    c1, c2 = st.columns(2)

    with c1:

        st.info("Naive Bayes")

        st.metric(
            "Accuracy",
            "97.14%"
        )

        st.metric(
            "Precision",
            "97%"
        )

        st.metric(
            "Recall",
            "100%"
        )

        st.metric(
            "F1 Score",
            "99%"
        )

    with c2:

        st.success("Support Vector Machine")

        st.metric(
            "Accuracy",
            "97.62%"
        )

        st.metric(
            "Precision",
            "98%"
        )

        st.metric(
            "Recall",
            "100%"
        )

        st.metric(
            "F1 Score",
            "99%"
        )

    st.markdown("---")

    # ====================================
    # GRAFIK PERBANDINGAN ACCURACY
    # ====================================

    st.subheader("📈 Perbandingan Accuracy")

    accuracy_df = pd.DataFrame({

        "Model":[
            "Naive Bayes",
            "Support Vector Machine"
        ],

        "Accuracy":[
            97.14,
            97.62
        ]

    })

    fig = px.bar(

        accuracy_df,

        x="Model",

        y="Accuracy",

        text="Accuracy",

        color="Model",

        title="Perbandingan Accuracy Model"

    )

    fig.update_traces(texttemplate='%{text:.2f}%')

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    # ====================================
    # PERBANDINGAN METRIK
    # ====================================

    st.subheader("📊 Perbandingan Seluruh Metrik")

    metric_df = pd.DataFrame({

        "Metric":[
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ],

        "Naive Bayes":[
            97.14,
            97,
            100,
            99
        ],

        "Support Vector Machine":[
            97.62,
            98,
            100,
            99
        ]

    })

    st.dataframe(
        metric_df,
        use_container_width=True
    )

    st.markdown("---")

    # ====================================
    # CONFUSION MATRIX
    # ====================================

    st.subheader("🖼️ Confusion Matrix")

    col1, col2 = st.columns(2)

    with col1:

        st.write("Naive Bayes")

        st.image(
            "nb_confusion.png",
            use_container_width=True
        )

    with col2:

        st.write("Support Vector Machine")

        st.image(
            "svm_confusion.png",
            use_container_width=True
        )

    st.markdown("---")

    # ====================================
    # KESIMPULAN
    # ====================================

    st.subheader("🏆 Kesimpulan")

    st.success(
        """
        Berdasarkan hasil evaluasi model:

        ✅ Naive Bayes Accuracy : 97.14%

        ✅ Support Vector Machine Accuracy : 97.62%

        **Support Vector Machine memiliki performa terbaik**
        karena memperoleh nilai accuracy yang lebih tinggi.
        """
    )

# =====================================================
# BERTopic
# =====================================================

elif menu == "📚 BERTopic":

    st.title("📚 Analisis Topik (BERTopic)")

    st.write(
        "Visualisasi hasil klasterisasi komentar menggunakan BERTopic."
    )

    st.markdown("---")

    # ====================================
    # KPI
    # ====================================

    total_topic = topic_df["topic"].nunique()
    total_comment = len(topic_df)

    c1, c2 = st.columns(2)

    c1.metric(
        "Jumlah Topic",
        total_topic
    )

    c2.metric(
        "Jumlah Komentar",
        total_comment
    )

    st.markdown("---")

    # ====================================
    # DISTRIBUSI TOPIC
    # ====================================

    st.subheader("📊 Distribusi Topic")

    topic_count = (
        topic_df["topic"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    topic_count.columns = [
        "Topic",
        "Jumlah"
    ]

    fig = px.bar(
        topic_count,
        x="Topic",
        y="Jumlah",
        text="Jumlah",
        color="Topic",
        title="Distribusi Komentar Setiap Topic"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    # ====================================
    # PIE CHART
    # ====================================

    st.subheader("🥧 Persentase Topic")

    fig2 = px.pie(
        topic_count,
        names="Topic",
        values="Jumlah",
        hole=0.45
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    st.markdown("---")

    # ====================================
    # PILIH TOPIC
    # ====================================

    st.subheader("🎯 Pilih Topic")

    pilih = st.selectbox(

        "Topic",

        sorted(topic_df["topic"].unique())

    )

    hasil = topic_df[
        topic_df["topic"] == pilih
    ]

    st.success(
        f"Jumlah komentar pada Topic {pilih} : {len(hasil)}"
    )

    st.dataframe(

        hasil[
            [
                "text",
                "topic"
            ]
        ],

        use_container_width=True,

        height=350

    )

    st.markdown("---")

    # ====================================
    # BARIS PERTAMA
    # ====================================

    st.subheader("📝 Contoh Komentar")

    contoh = hasil["text"].head(5)

    for i, komentar in enumerate(contoh, start=1):

        st.write(f"**{i}.** {komentar}")

    st.markdown("---")

    # ====================================
    # DOWNLOAD
    # ====================================

    csv = hasil.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(

        "📥 Download Topic",

        csv,

        f"topic_{pilih}.csv",

        "text/csv"

    )

# =====================================================
# SOCIAL NETWORK ANALYSIS
# =====================================================

elif menu == "🌐 Social Network Analysis":

    st.title("🌐 Social Network Analysis")

    st.write(
        "Analisis hubungan antar kata pada komentar YouTube menggunakan Network Analysis."
    )

    st.markdown("---")

    import networkx as nx
    from itertools import combinations
    from collections import Counter

    # ======================================
    # MEMBANGUN GRAPH
    # ======================================

    G = nx.Graph()

    for kalimat in topic_df["normalisasi"].dropna():

        kata = str(kalimat).split()

        kata = list(set(kata))

        for a, b in combinations(kata, 2):

            if G.has_edge(a, b):
                G[a][b]["weight"] += 1
            else:
                G.add_edge(a, b, weight=1)

    # ======================================
    # KPI
    # ======================================

    node = G.number_of_nodes()

    edge = G.number_of_edges()

    density = nx.density(G)

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Jumlah Node",
        node
    )

    c2.metric(
        "Jumlah Edge",
        edge
    )

    c3.metric(
        "Density",
        round(density,4)
    )

    st.markdown("---")

    # ======================================
    # NETWORK GRAPH
    # ======================================

    st.subheader("Visualisasi Network")

    pos = nx.spring_layout(
        G,
        seed=42,
        k=0.4
    )

    edge_x=[]
    edge_y=[]

    for e in G.edges():

        x0,y0=pos[e[0]]

        x1,y1=pos[e[1]]

        edge_x.extend([x0,x1,None])

        edge_y.extend([y0,y1,None])

    edge_trace=go.Scatter(

        x=edge_x,

        y=edge_y,

        mode="lines",

        line=dict(width=0.5,color="gray"),

        hoverinfo="none"

    )

    node_x=[]

    node_y=[]

    text=[]

    size=[]

    degree=dict(G.degree())

    for n in G.nodes():

        x,y=pos[n]

        node_x.append(x)

        node_y.append(y)

        text.append(n)

        size.append(degree[n]*4+8)

    node_trace=go.Scatter(

        x=node_x,

        y=node_y,

        mode="markers+text",

        text=text,

        textposition="top center",

        hoverinfo="text",

        marker=dict(

            size=size,

            color=size,

            colorscale="Blues",

            showscale=True

        )

    )

    fig=go.Figure(

        data=[edge_trace,node_trace]

    )

    fig.update_layout(

        showlegend=False,

        margin=dict(l=0,r=0,t=0,b=0)

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    st.markdown("---")

    # ======================================
    # DEGREE CENTRALITY
    # ======================================

    st.subheader("Top 10 Degree Centrality")

    degree = nx.degree_centrality(G)

    degree_df = pd.DataFrame(

        degree.items(),

        columns=["Node","Degree"]

    )

    degree_df = degree_df.sort_values(

        "Degree",

        ascending=False

    ).head(10)

    st.dataframe(

        degree_df,

        use_container_width=True

    )

    st.bar_chart(

        degree_df.set_index("Node")

    )

    st.markdown("---")

    # ======================================
    # BETWEENNESS
    # ======================================

    st.subheader("Top 10 Betweenness Centrality")

    between = nx.betweenness_centrality(G)

    between_df = pd.DataFrame(

        between.items(),

        columns=["Node","Betweenness"]

    )

    between_df = between_df.sort_values(

        "Betweenness",

        ascending=False

    ).head(10)

    st.dataframe(

        between_df,

        use_container_width=True

    )

    st.bar_chart(

        between_df.set_index("Node")

    )

    st.markdown("---")

    # ======================================
    # CLOSENESS
    # ======================================

    st.subheader("Top 10 Closeness Centrality")

    close = nx.closeness_centrality(G)

    close_df = pd.DataFrame(

        close.items(),

        columns=["Node","Closeness"]

    )

    close_df = close_df.sort_values(

        "Closeness",

        ascending=False

    ).head(10)

    st.dataframe(

        close_df,

        use_container_width=True

    )

    st.bar_chart(

        close_df.set_index("Node")

    )

    st.markdown("---")

    # ======================================
    # DEGREE CENTRALITY
    # ======================================

    st.subheader("Top 10 Degree Centrality")

    degree = nx.degree_centrality(G)

    degree_df = pd.DataFrame(

        degree.items(),

        columns=["Node","Degree"]

    )

    degree_df = degree_df.sort_values(

        "Degree",

        ascending=False

    ).head(10)

    st.dataframe(

        degree_df,

        use_container_width=True

    )

    st.bar_chart(

        degree_df.set_index("Node")

    )

    st.markdown("---")

    # ======================================
    # BETWEENNESS
    # ======================================

    st.subheader("Top 10 Betweenness Centrality")

    between = nx.betweenness_centrality(G)

    between_df = pd.DataFrame(

        between.items(),

        columns=["Node","Betweenness"]

    )

    between_df = between_df.sort_values(

        "Betweenness",

        ascending=False

    ).head(10)

    st.dataframe(

        between_df,

        use_container_width=True

    )

    st.bar_chart(

        between_df.set_index("Node")

    )

    st.markdown("---")

    # ======================================
    # CLOSENESS
    # ======================================

    st.subheader("Top 10 Closeness Centrality")

    close = nx.closeness_centrality(G)

    close_df = pd.DataFrame(

        close.items(),

        columns=["Node","Closeness"]

    )

    close_df = close_df.sort_values(

        "Closeness",

        ascending=False

    ).head(10)

    st.dataframe(

        close_df,

        use_container_width=True

    )

    st.bar_chart(

        close_df.set_index("Node")

    )

    st.markdown("---")