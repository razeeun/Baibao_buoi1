import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pickle
import re

# Hàm chuyển đổi total_sqft thành số thực
def convert_sqft_to_num(x):
    try:
        # Xử lý khoảng số, ví dụ: "2100 - 2850"
        if '-' in str(x):
            tokens = x.split('-')
            return np.mean([float(tokens[0]), float(tokens[1])])
        # Xử lý các đơn vị như "34.46Sq. Meter", "4125Perch"
        if isinstance(x, str):
            # Trích xuất số bằng biểu thức chính quy
            num = re.findall(r'^\d*\.?\d*', x)
            if num and num[0]:
                value = float(num[0])
                # Chuyển đổi đơn vị nếu cần
                if 'Sq. Meter' in x:
                    value *= 10.764  # 1 Sq. Meter = 10.764 Sqft
                elif 'Perch' in x:
                    value *= 272.25  # 1 Perch = 272.25 Sqft
                return value
        return float(x)
    except:
        return np.nan  # Trả về NaN nếu không thể chuyển đổi

# Đọc dữ liệu từ file CSV
df = pd.read_csv(r"C:\Users\Admin\OneDrive\Documents\Hoc_May\dulieu\VUTHANHTUNG_TTNT2\buoi1\Bengaluru_House_Data.csv")

# Tiền xử lý dữ liệu
# Xử lý giá trị thiếu
df['bath'] = df['bath'].fillna(df['bath'].median())
df['balcony'] = df['balcony'].fillna(df['balcony'].median())
df['total_sqft'] = df['total_sqft'].apply(convert_sqft_to_num)
df['total_sqft'] = df['total_sqft'].fillna(df['total_sqft'].median())  # Điền giá trị thiếu bằng trung vị

# Tạo đặc trưng mới: Bhk (số phòng ngủ)
df['bhk'] = df['size'].apply(lambda x: int(x.split(' ')[0]) if pd.notnull(x) else np.nan)
df['bhk'] = df['bhk'].fillna(df['bhk'].median())

# Tính giá mỗi foot vuông
df['price_per_sqft'] = df['price'] * 100000 / df['total_sqft']

# Giảm chiều: nhóm các vị trí có dưới 10 danh sách thành 'other'
location_stats = df.groupby('location')['location'].agg('count').sort_values(ascending=False)
locations_less_than_10 = location_stats[location_stats <= 10]
df['location'] = df['location'].apply(lambda x: 'other' if x in locations_less_than_10 else x)

# Loại bỏ ngoại lệ
# Quy tắc kinh doanh: diện tích tối thiểu 300 sqft/phòng ngủ
df = df[~(df['total_sqft'] / df['bhk'] < 300)]

# Loại bỏ ngoại lệ dựa trên thống kê giá mỗi foot vuông theo vị trí
def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf['price_per_sqft'])
        st = np.std(subdf['price_per_sqft'])
        reduced_df = subdf[(subdf['price_per_sqft'] > (m - st)) & (subdf['price_per_sqft'] <= (m + st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out

df = remove_pps_outliers(df)

# Loại bỏ ngoại lệ Bhk: giá 3 BHK thấp hơn 2 BHK cùng diện tích và vị trí
def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df['price_per_sqft']),
                'std': np.std(bhk_df['price_per_sqft']),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk - 1)
            if stats and stats['count'] > 5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df['price_per_sqft'] < stats['mean']].index.values)
    return df.drop(exclude_indices, axis='index')

df = remove_bhk_outliers(df)

# Mã hóa one-hot cho biến phân loại
df = pd.get_dummies(df, columns=['location', 'area_type'], drop_first=True)

# Chuẩn hóa các đặc trưng số
scaler = StandardScaler()
numerical_features = ['total_sqft', 'bath', 'balcony', 'bhk']
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Phân tích dữ liệu khám phá (EDA): Bản đồ nhiệt tương quan
plt.figure(figsize=(10, 8))
correlation_matrix = df[['total_sqft', 'bath', 'balcony', 'bhk', 'price']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X = df.drop(['price', 'price_per_sqft', 'size', 'society', 'availability'], axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình Hồi quy Tuyến tính
model = LinearRegression()
model.fit(X_train, y_train)

# Đánh giá mô hình
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f'Train R^2 Score: {train_score}')
print(f'Test R^2 Score: {test_score}')

# Lưu mô hình và scaler
with open('house_price_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Hàm dự đoán giá nhà
def predict_price(location, area_type, total_sqft, bath, balcony, bhk):
    # Tạo DataFrame để giữ tên cột
    input_data = pd.DataFrame([[total_sqft, bath, balcony, bhk]], columns=numerical_features)
    
    # Chuẩn hóa các đặc trưng số
    input_data_scaled = scaler.transform(input_data)
    
    # Tạo DataFrame đầy đủ với tất cả các cột của X
    input_df = pd.DataFrame(np.zeros((1, len(X.columns))), columns=X.columns)
    
    # Gán giá trị đã chuẩn hóa cho các cột đặc trưng số
    for i, col in enumerate(numerical_features):
        input_df[col] = input_data_scaled[0][i]
    
    # Gán giá trị 1 cho cột location và area_type tương ứng
    loc_col = f'location_{location}' if location != 'other' else 'location_other'
    area_col = f'area_type_{area_type}' if area_type in ['Built-up  Area', 'Carpet  Area', 'Plot  Area'] else None
    
    if loc_col in X.columns:
        input_df[loc_col] = 1
    if area_col and area_col in X.columns:
        input_df[area_col] = 1
    
    # Dự đoán giá
    return model.predict(input_df)[0]

# Ví dụ dự đoán
print(predict_price('Electronic City Phase II', 'Super built-up  Area', 1056, 2, 1, 2))