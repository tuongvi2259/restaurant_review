import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from io import StringIO
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn. metrics import classification_report, roc_auc_score, roc_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
from wordcloud import WordCloud
import pickle
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


st.title("Data Science Project")
st.write("## Project 1 - Sentiment Analysis")
menu = ["Business Objective_Restaurant", "Results of Project","Setiment Analysis", "Restaurant Review"]
choice = st.sidebar.selectbox('Menu', menu)

# 1. Read data
Restaurant = pd.read_csv(r"1_Restaurants.csv")
Review = pd.read_csv(r"2_Reviews.csv")

if choice == 'Business Objective_Restaurant':  

    st.subheader("Business Objective")
    st.write("""
    **Mục tiêu/ vấn đề**: Xây dựng mô hình dự đoán giúp nhà hàng/ quán ăn có thể biết được những phản hồi nhanh chóng của khách hàng về sản phẩm hay dịch vụ của họ (tích cực, tiêu cực hay trung tính), điều này giúp cho nhà hàng hiểu được tình hình kinh doanh, hiểu được ý kiến của khách hàng từ đó giúp nhà hàng cải thiện hơn về chất lượng dịch vụ và sản phẩm.
    """)
    st.write("""
    **Dữ liệu được cung cấp** sẵn gồm có các tập tin:
        ***1_Restaurants.csv***,
        ***2_Reviews.csv*** chứa thông tin về nhà hàng/ quán ăn và review của khách hàng dành cho các món ăn của nhà hàng đó.
    """)
    st.write("""1_Restaurants.csv""")
    st.dataframe(Restaurant.head())
    st.write("""2_Reviews.csv""")
    st.dataframe(Review.head())
    st.image("Fig1.png")

    st.subheader("Setiment Analysis")
    st.write("""
    Trong Setiment Analysis, chúng tôi cung cấp cho nhà hàng/ quán ăn sự đánh giá cho các comment của khách hàng là tích cực/tiêu cực/trung tính
    """)

    st.subheader("Restaurant Review")
    st.write("""
    Trong Restaurant Review, chúng tôi cung cấp cho nhà hàng/ quán ăn một cái nhìn tổng quan về các đánh giá để nhà hàng/ quán ăn có thể đánh giá tình hình hoạt động, từ đó cải thiện chất lượng sản phẩm và các dịch vụ của họ
    """)

    st.subheader("Authors")
    st.write("""Hạ Thị Thiều Dao""")
    st.write("""Huỳnh Thiện Phúc""")
    st.write("""Văn Thị Tường Vi""")
elif choice == 'Results of Project':

    data = pd.read_csv(r"cleaned_data_Sentiment.csv")
    vectorizer = pickle.load(open('vectorizer.pkl','rb'))
    rf_classifier  = joblib.load(open('P1RFModel.joblib', 'rb'))

    st.subheader("Các thông tin thu thập được từ dữ liệu thô")
    st.write("""Biểu đồ phân bố Rating""")
    st.image("Rating.jpg")
    st.write("Biểu đồ phân phối số lượt mua hàng")
    st.image("Luotmua.jpg")
    st.write("Có tổng cộng 1174 nhà hàng")
    st.write("- Số lượt mua trung bình cho mỗi tháng nhà hàng là khoảng 25.52")
    st.write("- 25% nhà hàng có số lượt mua ít hơn hoặc bằng 2 lượt")
    st.write("- Số lượt mua nhiều nhất cho một nhà hàng là 100 lượt")
    st.write("Biểu đồ số lượt comment theo Quận")
    st.image("Quan.jpg")
    st.write("Quận 1, 3, 4 và 5 có số lượt mua cao hơn (quận gần trung tâm)=> Địa lý có ảnh hưởng đến lượt mua")
    
    st.subheader("Performance of Project")
    st.write("""Performance of Model:""")
    
    X_text = data['processed_comment']
    X_counts = data[['positive_word_count', 'negative_word_count']]
    y = data['Sentiment']

    # Dùng TF-IDF, Convert text data sang numerical
    vectorizer = TfidfVectorizer()
    X_text = vectorizer.fit_transform(X_text)

    # Concatenate text features with counts
    X = hstack((X_text, X_counts))
    # Perform resampling
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y)

    # Split the resampled data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

        # Predict the test set
    y_pred_rf = rf_classifier.predict(X_test)

    #4. Evaluate model
    score_train = rf_classifier.score(X_train,y_train)
    score_test = rf_classifier.score(X_test,y_test)
    cm = confusion_matrix(y_test,y_pred_rf, labels=['positive', 'neutral', 'negative'])

    cr = classification_report(y_test, y_pred_rf)

    st.write("""Model được xây dựng để phân tích tình cảm, cảm xúc của các commen mà khách hàng đánh giá""")
    st.write("""Model đạt được hiệu quả phân loại ở tập train là 0.99 và tập test là 0.95""")
    st.code("Score train:"+ str(round(score_train,2)) + " vs Score test:" + str(round(score_test,2)))
    st.write("""Model đạt đã phân loại đúng 3646 comment tích cực (Positive), 3634 comment trung tính (Neutral) và 3940 comment tiêu cực (Negative)""")
    st.write("""###### Confusion matrix:""")
    st.code(cm)
    st.write("""Model đạt hiệu quả phân loại đối với comment tích cực là 94%, trung tính là 95% và tiêu cực là 97%""")
    st.write("""###### Classification report:""")
    st.code(cr)


elif choice == 'Setiment Analysis':
    data = pd.read_csv(r"cleaned_data_Sentiment.csv")
    vectorizer = pickle.load(open('vectorizer.pkl','rb'))
    rf_classifier  = joblib.load(open('P1RFModel.joblib', 'rb'))
    
    # Load positive and negative words
    positive_words = [
    "thích", "tốt", "xuất sắc", "tuyệt vời", "tuyệt hảo", "đẹp", "ổn", "ngon",
    "hài lòng", "ưng ý", "hoàn hảo", "chất lượng", "thú vị", "nhanh",
    "tiện lợi", "dễ sử dụng", "hiệu quả", "ấn tượng",
    "nổi bật", "tận hưởng", "tốn ít thời gian", "thân thiện", "hấp dẫn",
    "gợi cảm", "tươi mới", "lạ mắt", "cao cấp", "độc đáo",
    "hợp khẩu vị", "rất tốt", "rất thích", "tận tâm", "đáng tin cậy", "đẳng cấp",
    "hấp dẫn", "an tâm", "không thể cưỡng lại", "thỏa mãn", "thúc đẩy",
    "cảm động", "phục vụ tốt", "làm hài lòng", "gây ấn tượng", "nổi trội",
    "sáng tạo", "quý báu", "phù hợp", "tận tâm",
    "hiếm có", "cải thiện", "hoà nhã", "chăm chỉ", "cẩn thận",
    "vui vẻ", "sáng sủa", "hào hứng", "đam mê", "vừa vặn", "đáng tiền"
    # Từ mới từ file trích xuất
    ,"ưng", "thoải mái", "được giá", "thơm", "mềm mịn", "thanh lịch", "thông thoáng","đậm đà","nóng hổi", "tỉ mỉ","nóng giòn"
    ,"sạch thoáng", "sang trọng","ấm cúng"
    ]
    negative_words = [
    "kém", "tệ", "đau", "xấu", "dở", "ức",
    "buồn", "rối", "thô", "lâu", "chán",
    "tối", "chán", "ít", "mờ", "mỏng",
    "lỏng lẻo", "khó", "cùi", "yếu",
    "kém chất lượng", "không thích", "không thú vị", "không ổn",
    "không hợp", "không đáng tin cậy", "không chuyên nghiệp",
    "không phản hồi", "không an toàn", "không phù hợp", "không thân thiện", "không linh hoạt", "không đáng giá",
    "không ấn tượng", "không tốt", "chậm", "khó khăn", "phức tạp",
    "khó hiểu", "khó chịu", "gây khó dễ", "rườm rà", "khó truy cập",
    "thất bại", "tồi tệ", "khó xử", "không thể chấp nhận", "tồi tệ","không rõ ràng",
    "không chắc chắn", "rối rắm", "không tiện lợi", "không đáng tiền", "chưa đẹp", "không đẹp"
    # Từ mới từ file trích xuất
    ,"lõng bõng", "tùm lum", "tí xíu","ngấy", "kì kì","nhão nhão","chậm chạp", "không trả lời", "không ngon"
    ,"nhạt nhẽo", "mặn chát", "ỉu", "quá tải","tanh"
    ]

    # Function to count positive/negative words
    def count_words(document, word_list):
        document_lower = document.lower()
        word_count = 0
        word_list_found = []
        for word in word_list:
            if word in document_lower:
                word_count += document_lower.count(word)
                word_list_found.append(word)
        return word_count, word_list_found
    
    # Function to predict sentiment for a single comment
    def predict_single_comment(comment):
        
        # Preprocess the comment
        processed_comment = vectorizer.transform([comment])

        # Count positive/negative words for the comment
        positive_word_count, _ = count_words(comment, positive_words)
        negative_word_count, _ = count_words(comment, negative_words)

        # Combine text features with numeric features
        combined_features = hstack((processed_comment, [[positive_word_count, negative_word_count]]))

        # Predict sentiment for the comment
        predicted_sentiment = rf_classifier.predict(combined_features)

        return predicted_sentiment[0]


    # Cho người dùng chọn nhập dữ liệu hoặc upload file
    type = st.radio("Chọn cách nhập dữ liệu", options=["Nhập dữ liệu vào text area", "Nhập nhiều dòng dữ liệu trực tiếp", "Upload file"])
   # Nếu người dùng chọn nhập dữ liệu vào text area
    if type == "Nhập dữ liệu vào text area":
        st.subheader("Nhập dữ liệu vào text area")
        content = st.text_area("Nhập ý kiến:")
        results=predict_single_comment(content)
    # Từ df này, người dùng có thể thực hiện các xử lý dữ liệu khác nhau
        submitted= st.button("Dự đoán cảm xúc")
        if submitted:
            st.write("Hiển thị kết quả dự đoán cảm xúc...")
            st.write(results)

# Nếu người dùng chọn nhập nhiều dòng dữ liệu trực tiếp vào một table
    elif type == "Nhập nhiều dòng dữ liệu trực tiếp":
            st.subheader("Nhập nhiều dòng dữ liệu trực tiếp")        
            data = []
            # Loop to collect user inputs
            for i in range(5):
                opinion = st.text_area(f"Nhập ý kiến {i+1}:")
                # Append each user input to the list
                data.append(opinion)
            
            # Create a DataFrame from the list of data
            df = pd.DataFrame({"Ý kiến": data})
            st.dataframe(df)
            results = []
        # Iterate over each content in the DataFrame
            for content in df["Ý kiến"]:
        # Predict sentiment for each content
                result = predict_single_comment(content) if content.strip() else None
        # Append the result to the results list
                results.append(result)
        # Add results to the DataFrame as a new column
            df["Kết quả"] = results
            df.fillna("None")
        # Từ df này, người dùng có thể thực hiện các xử lý dữ liệu khác nhau
            submitted= st.button("Dự đoán cảm xúc")
            if submitted:
                st.write("Hiển thị kết quả dự đoán cảm xúc...")
                st.write(df)
    # Nếu người dùng chọn upload file
    elif type == "Upload file":
        st.subheader("Upload file")
        st.write("""
        File upload là file ".csv" và có cột "Comment" chứa nội dung comment""")
        
        # Upload file
        st.markdown('<style>body { font-family: "Arial Unicode MS"; }</style>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Chọn file dữ liệu", type=["csv"])
        if uploaded_file is not None:
            # Đọc file dữ liệu
            df = pd.read_csv(uploaded_file, encoding='utf-8')
            # Kiểm tra xem dấu ngăn cách là dấu phẩy hay dấu chấm phẩy
            if ";" in df.to_string():
                # Thay thế dấu chấm phẩy bằng dấu phẩy
                df = df.replace(';', ',')
            # In ra danh sách tên cột
            st.write("Danh sách tên cột trong DataFrame:")
            st.write(df.columns)
            results = []
            # Iterate over each content in the DataFrame
            for content in df["Comment"]:
                # Predict sentiment for each content
                result = predict_single_comment(content) if content.strip() else None
                # Append the result to the results list
                results.append(result)
            
            # Add results to the DataFrame as a new column
            df["Kết quả"] = results
            df.fillna("None", inplace=True)  # Fill NaN values with "None"
            # Từ df này, người dùng có thể thực hiện các xử lý dữ liệu khác nhau
            submitted = st.button("Dự đoán cảm xúc")
            if submitted:
                st.write("Hiển thị kết quả dự đoán cảm xúc...")
                st.write(df)

elif choice == 'Restaurant Review':
    data = pd.read_csv(r"cleaned_data_Sentiment.csv")
    vectorizer = pickle.load(open('vectorizer.pkl','rb'))
    rf_classifier  = joblib.load(open('P1RFModel.joblib', 'rb'))
    nltk.download('punkt')
    with open("vietnamese-stopwords.txt", "r", encoding="utf-8") as file:
        file_contents = file.read()
    # Process the file contents as needed
        # For example, you can split the text into lines and display them
        stop_words = file_contents.split("\n")
    nltk.download('punkt')

    def visualize_restaurant_info(restaurant_id):
        # Find comment with the input ID
        restaurant_data = data[data['IDRestaurant'] == int(restaurant_id)]

        # Exit if no data
        if restaurant_data.empty:
            st.write(f"Không có dữ liệu cho nhà hàng có ID {restaurant_id}.")
        else:
            # Display basic information about the restaurant
            st.write("Thông tin cơ bản về nhà hàng:")
            st.write(restaurant_data[['Restaurant', 'Address', 'District', 'Time_y', 'Price']].iloc[0])

            # Plot rating distribution
            fig, ax = plt.subplots()
            ax.hist(restaurant_data['Rating'], bins=10, color='skyblue', edgecolor='black')
            ax.set_xlabel('Rating')
            ax.set_ylabel('Số lượng đánh giá')
            st.pyplot(fig)

            # Find comment with the inputed ID
            restaurant_comments = restaurant_data['processed_comment']

            # Vectorize the comment
            restaurant_comments_tfidf = vectorizer.transform(restaurant_comments)

            # Get numeric feature
            restaurant_numeric = restaurant_data[['positive_word_count', 'negative_word_count']]

            # Combine text and numeric features
            restaurant_combined = hstack((restaurant_comments_tfidf, restaurant_numeric))

            # Predict sentiment
            sentiment_prediction = rf_classifier.predict(restaurant_combined)


            # Create a DataFrame to hold comments and their sentiment predictions
            comments_df = pd.DataFrame({
                'Comment': restaurant_comments,
                'Sentiment_Prediction': sentiment_prediction
            })

            # Display sentiment prediction
            st.write("Dự đoán cảm xúc:")
            st.write(comments_df)

            # Calculate mean rating
            mean_rating = restaurant_data['Rating'].mean()
            st.write("Average rating:", mean_rating)

            # Determine overall sentiment for the restaurant
            overall_sentiment = "negative" if mean_rating < 5 else ("neutral" if 5 <= mean_rating < 7 else "positive")
            st.write("Overall sentiment for the restaurant:", overall_sentiment)

            # Get comments classified as negative sentiment
            negative_comments = restaurant_data[restaurant_data['Sentiment'] == 'negative']['processed_comment']
            negative_text = ' '.join(negative_comments)

            tokens = word_tokenize(negative_text)
            negative_text = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]

            # Get comments classified as positive sentiment
            positive_comments = restaurant_data[restaurant_data['Sentiment'] == 'positive']['processed_comment']
            positive_text = ' '.join(positive_comments)

            tokens = word_tokenize(positive_text)
            positive_text = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]

            # Generate word clouds for negative and positive comments
            if negative_text:
                negative_text_combined = ' '.join(negative_text)
                negative_wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=30).generate(negative_text_combined)
                st.write("Word Cloud for Negative Comments:")
                st.image(negative_wordcloud.to_array(), caption='Negative Sentiment Word Cloud', use_column_width=True)
            else:
                st.write("Không có bình luận tiêu cực.")

            if positive_text:
                positive_text_combined = ' '.join(positive_text)
                positive_wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=30).generate(positive_text_combined)
                st.write("Word Cloud for Positive Comments:")
                st.image(positive_wordcloud.to_array(), caption='Positive Sentiment Word Cloud', use_column_width=True)
            else:
                st.write("Không có bình luận tích cực.")
    

    # Streamlit UI
    st.title("Restaurant Information Visualization")
    type = st.radio("Chọn cách nhập dữ liệu", options=["Nhập dữ liệu vào text area", "Chọn ID nhà hàng"])
    if type == "Nhập dữ liệu vào text area":
        restaurant_id = st.text_input("Nhập ID của nhà hàng (ID từ 1 -> 1612):", "")
        if st.button("Visualize"):
            if restaurant_id.strip():
                visualize_restaurant_info(restaurant_id)
            else:
                st.write("Vui lòng nhập một ID nhà hàng hợp lệ.")
    elif type == "Chọn ID nhà hàng":
        lst = data['IDRestaurant'].unique()
        selected_option = st.selectbox('Chọn một tùy chọn:', lst)
        # Hiển thị giá trị đã chọn
        st.write('Bạn đã chọn:', selected_option)
        if st.button("Visualize"):
            visualize_restaurant_info(selected_option)
            
