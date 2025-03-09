import pandas as pd
#Cau 1:
df = pd.read_csv(r"D:\ThucHanh_TTNT\Scripts\gapminder.tsv", sep="\t")
#hien thi 5 dong
print("\nCâu 1:")
print(df.head())
#Cau 2:
df = pd.read_csv(r"D:\ThucHanh_TTNT\Scripts\gapminder.tsv", sep="\t")
# lay so row va so column
so_dong, so_cot = df.shape
# in kq
print("\nCâu 2:")
print(f"Số dòng: {so_dong}")
print(f"Số cột: {so_cot}")
#Cau 3:
df = pd.read_csv(r"D:\ThucHanh_TTNT\Scripts\gapminder.tsv", sep="\t")
# In tên các cột
print("\nCâu 3:")
print("Tên các cột trong Gapminder:")
print(df.columns.tolist())
#Cau 4:
df = pd.read_csv(r"D:\ThucHanh_TTNT\Scripts\gapminder.tsv", sep="\t")
# in kieu du lieu cua ten cot
print("\nCâu 4:")
print(type(df.columns))
#Cau 5:
df = pd.read_csv(r"D:\ThucHanh_TTNT\Scripts\gapminder.tsv", sep="\t")
# Lưu cột "country" vào biến riêng
country_column = df["country"]
# hien thi 5 dong dau tien
print("\nCâu 5:")
print(country_column.head(5))
#Cau 6:
df = pd.read_csv(r"D:\ThucHanh_TTNT\Scripts\gapminder.tsv", sep="\t")
country_column = df["country"]
# 5 dong cuoi cung
print("\nCâu 6:")
print(country_column.tail(5))
#Cau 7:
df = pd.read_csv(r"D:\ThucHanh_TTNT\Scripts\gapminder.tsv", sep="\t")
# cac cột "country", "continent", "year"
selected_columns = df[["country", "continent", "year"]]
# 5 dong dau tien
print("First 5 observations:")
print(selected_columns.head(5))
# 5 dong cuoi cung
print("\nCâu 7:")
print("\nLast 5 observations:")
print(selected_columns.tail(5))
#Cau 8: How to get the first row of tsv file? How to get the 100th row.
df = pd.read_csv(r"D:\ThucHanh_TTNT\Scripts\gapminder.tsv", sep="\t")
# lay hang` dau` tien
first_row = df.iloc[0]
print("\nCâu 8:")
print("First row:\n", first_row)
# lay hang 100, chi lay so 99
hundredth_row = df.iloc[99]  # Vì index trong Python bắt đầu từ 0
print("\n100th row:\n", hundredth_row)
#Cau 9:
df = pd.read_csv(r"D:\ThucHanh_TTNT\Scripts\gapminder.tsv", sep="\t")
# lay cot dau tien = integer
first_column = df.iloc[:, 0]
print("\nCâu 9:")
print("First column:\n", first_column.head())
# lay cot dau tien va cot cuoi cung
first_and_last_columns = df.iloc[:, [0, -1]]
print("\nFirst and last columns:\n", first_and_last_columns.head())
#Cau 10: How to get the last row with .loc? Try with index -1? Correct?
df = pd.read_csv(r"D:\ThucHanh_TTNT\Scripts\gapminder.tsv", sep="\t")
# truy xuat hang cuoi cung voi 0 va -1
print("\nCâu 10:")
try:
    last_row = df.loc[-1]
    print("Last row using .loc[-1]:\n", last_row)
except KeyError:
    print("Using .loc[-1] is incorrect because .loc uses labels, not integer indexes.")
#index cuối cùng của DataFrame
correct_last_row = df.loc[df.index[-1]]
print("\nLast row using correct .loc:\n", correct_last_row)
#Cau 11:
#Cach 1:
df = pd.read_csv(r"D:\ThucHanh_TTNT\Scripts\gapminder.tsv", sep="\t")
# chon cac hang` = so nguyen
selected_rows = df.iloc[[0, 99, 999]]  # bat dau tu 0
print("\nCâu 11 cach 1: ")
print("Selected rows using iloc:\n", selected_rows)
#Cach 2:
# index tương ứng với các hàng cần chọn
selected_rows_loc = df.loc[df.index[[0, 99, 999]]]
print("\nCâu 11 cach 2: ")
print("\nSelected rows using loc:\n", selected_rows_loc)
#Cau 12:
#Cach 1 dung` .loc:
df = pd.read_csv(r"D:\ThucHanh_TTNT\Scripts\gapminder.tsv", sep="\t")
# quoc gia thu 43 ( index bắt đầu từ 0)
country_43_iloc = df.iloc[42]["country"]  # index cua hang 43 la 42
print("\nCâu 12 cach 1: ")
print("Country at 43rd row using iloc:", country_43_iloc)
#Cach 2 dung` .iloc:
# lay quoc gia 43 thuc te
country_43_loc = df.loc[df.index[42], "country"]
print("\nCâu 12 cach 2: ")
print("Country at 43rd row using loc:", country_43_loc)
#Cau 13:
#Cach 1 dung .iloc (chi so nguyen):
df = pd.read_csv(r"D:\ThucHanh_TTNT\Scripts\gapminder.tsv", sep="\t")
# chon cac hang 1, 100, 1000 va cac cot 1, 4, 6 va chi bat dau tu 0
selected_rows = df.iloc[[0, 99, 999], [0, 3, 5]]
print("\nCâu 13 cach 1: ")
print("Selected rows using iloc:\n", selected_rows)
#Cach 2 dung .loc (dua theo cot):
selected_rows_loc = df.loc[df.index[[0, 99, 999]], [df.columns[0], df.columns[3], df.columns[5]]]
print("\nCâu 13 cach 2: ")
print("\nSelected rows using loc:\n", selected_rows_loc)
#Cau 14:
df = pd.read_csv(r"D:\ThucHanh_TTNT\Scripts\gapminder.tsv", sep="\t")
first_10_rows = df.head(10)
print("\nCâu 14:")
print(first_10_rows)
#Cau 15:
df = pd.read_csv(r"D:\ThucHanh_TTNT\Scripts\gapminder.tsv", sep="\t")
# tuoi tho trung binh qua tung nam
avg_life_expectancy_per_year = df.groupby("year")["lifeExp"].mean()
print("\nCâu 15:")
print(avg_life_expectancy_per_year)
#Cau 16:
df = pd.read_csv(r"D:\ThucHanh_TTNT\Scripts\gapminder.tsv", sep="\t")
# danh sach nam duy hat
years = df["year"].unique()
# tinh tuoi tb = subsetting
print("\nCâu 16:")
for year in years:
    avg_life_exp = df[df["year"] == year]["lifeExp"].mean()
    print(f"Năm {year}: Tuổi thọ trung bình = {avg_life_exp:.2f}")
#Cau 17:
s = pd.Series(["banana", 42], index=[0, 1])
print("\nCâu 17:")
print(s)
#Cau 18:
# ‘Person’ for ‘Wes MCKinney’ and index ‘Who’ for ‘Creator of Pandas’
s = pd.Series(["Wes McKinney", "Creator of Pandas"], index=["Person", "Who"])
print("\nCâu 18:")
print(s)
#Cau 19:
data = {
    "Occupation": ["Chemist", "Statistician"],
    "Born": ["1920-07-25", "1876-06-13"],
    "Died": ["1958-04-16", "1937-10-16"],
    "Age": [37, 61]
}
df = pd.DataFrame(data, index=["Franklin", "Gosset"])
print("\nCâu 19:")
print(df)


