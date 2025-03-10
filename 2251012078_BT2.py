import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
file_path = r"D:\Buoi2_TTNT\.venv\Scripts\gapminder.tsv"

#Cau 1:
df = pd.read_csv(file_path, sep="\t")
print("\nCâu 1:")
print(df.head())

#Cau 2:
df = pd.read_csv(file_path, sep="\t")
so_dong, so_cot = df.shape
print("\nCâu 2:")
print(f"Số dòng: {so_dong}")
print(f"Số cột: {so_cot}")

#Cau 3:
df = pd.read_csv(file_path, sep="\t")
print("\nCâu 3:")
print("Tên các cột trong Gapminder:")
print(df.columns.tolist())

#Cau 4:
df = pd.read_csv(file_path, sep="\t")
print("\nCâu 4:")
print(type(df.columns))

#Cau 5:
df = pd.read_csv(file_path, sep="\t")
country_column = df["country"]
print("\nCâu 5:")
print(country_column.head(5))

#Cau 6:
df = pd.read_csv(file_path, sep="\t")
country_column = df["country"]
print("\nCâu 6:")
print(country_column.tail(5))

#Cau 7:
df = pd.read_csv(file_path, sep="\t")
selected_columns = df[["country", "continent", "year"]]
print("First 5 observations:")
print(selected_columns.head(5))
print("\nCâu 7:")
print("\nLast 5 observations:")
print(selected_columns.tail(5))

#Cau 8:
df = pd.read_csv(file_path, sep="\t")
first_row = df.iloc[0]
print("\nCâu 8:")
print("First row:\n", first_row)
hundredth_row = df.iloc[99]
print("\n100th row:\n", hundredth_row)

#Cau 9:
df = pd.read_csv(file_path, sep="\t")
first_column = df.iloc[:, 0]
print("\nCâu 9:")
print("First column:\n", first_column.head())
first_and_last_columns = df.iloc[:, [0, -1]]
print("\nFirst and last columns:\n", first_and_last_columns.head())

#Cau 10:
df = pd.read_csv(file_path, sep="\t")
print("\nCâu 10:")
try:
    last_row = df.loc[-1]
    print("Last row using .loc[-1]:\n", last_row)
except KeyError:
    print("Using .loc[-1] is incorrect because .loc uses labels, not integer indexes.")
correct_last_row = df.loc[df.index[-1]]
print("\nLast row using correct .loc:\n", correct_last_row)

#Cau 11:
df = pd.read_csv(file_path, sep="\t")
selected_rows = df.iloc[[0, 99, 999]]
print("\nCâu 11 cach 1: ")
print("Selected rows using iloc:\n", selected_rows)
selected_rows_loc = df.loc[df.index[[0, 99, 999]]]
print("\nCâu 11 cach 2: ")
print("\nSelected rows using loc:\n", selected_rows_loc)

#Cau 12:
df = pd.read_csv(file_path, sep="\t")
country_43_iloc = df.iloc[42]["country"]
print("\nCâu 12 cach 1: ")
print("Country at 43rd row using iloc:", country_43_iloc)
country_43_loc = df.loc[df.index[42], "country"]
print("\nCâu 12 cach 2: ")
print("Country at 43rd row using loc:", country_43_loc)

#Cau 13:
df = pd.read_csv(file_path, sep="\t")
selected_rows = df.iloc[[0, 99, 999], [0, 3, 5]]
print("\nCâu 13 cach 1: ")
print("Selected rows using iloc:\n", selected_rows)
selected_rows_loc = df.loc[df.index[[0, 99, 999]], [df.columns[0], df.columns[3], df.columns[5]]]
print("\nCâu 13 cach 2: ")
print("\nSelected rows using loc:\n", selected_rows_loc)

#Cau 14:
df = pd.read_csv(file_path, sep="\t")
first_10_rows = df.head(10)
print("\nCâu 14:")
print(first_10_rows)

#Cau 15:
df = pd.read_csv(file_path, sep="\t")
avg_life_expectancy_per_year = df.groupby("year")["lifeExp"].mean()
print("\nCâu 15:")
print(avg_life_expectancy_per_year)

#Cau 16:
df = pd.read_csv(file_path, sep="\t")
years = df["year"].unique()
print("\nCâu 16:")
for year in years:
    avg_life_exp = df[df["year"] == year]["lifeExp"].mean()
    print(f"Năm {year}: Tuổi thọ trung bình = {avg_life_exp:.2f}")

#Cau 17:
s = pd.Series(["banana", 42], index=[0, 1])
print("\nCâu 17:")
print(s)

#Cau 18:
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
print ("\n\n\n\n-------------------------LÝ THUYẾT-------------------------\n\n\n\n")
#1
data = [4879,12104,12756,6792,142984,120536,51118,49528]
pd.Series(data)
diameters =pd.Series(data)
print(diameters)
#2
Index=["Mercury", "Venus", "Earth", "Mars", "Jupyter", "Saturn", "Uranus", "Neptune"]
pd.Series(data,Index)
diameters = pd.Series(data,Index)
print (diameters)
#3
print(diameters["Earth"])
#4
print(diameters["Mercury":"Mars"])
#5
print(diameters[["Neptune","Earth","Jupyter",]])
#6
diameters["Pluto"] = 2370
print (diameters)
print (Index)
#7
dataP={"diameter":[4879,12104,12756,6792,142984,120536,51118,49528,2370],
"avg_temp":[167,464,15,-65,-110, -140, -195, -200, -225],
"gravity":[3.7, 8.9, 9.8, 3.7, 23.1, 9.0, 8.7, 11.0, 0.7]}
Planets=pd.DataFrame(dataP)
print(Planets)
#8
print(Planets.head(3))
#9
print(Planets.tail(2))
#10
print(Planets.columns)
#11
Planets.index = ["Mercury", "Venus", "Earth", "Mars", "Jupyter", "Saturn", "Uranus", "Neptune","Pluto"]
print(Planets)
#12
print(Planets["gravity"])
#13
print(Planets[["gravity","diameter"]])
#14
print(Planets.loc["Earth","gravity"])
#15
print(Planets.loc["Earth",["gravity","diameter"]])
#16
print(Planets.loc["Earth":"Saturn",["gravity","diameter"]])
#17
print(Planets[Planets["diameter"]>1000])
#18
print(Planets[Planets["diameter"]>100000])
#19
print(Planets[(Planets["avg_temp"] > 0 ) & (Planets["gravity"] > 5 )])
#20
print(Planets.sort_values("diameter"))
#21
print(Planets.sort_values("diameter",ascending= False))
#22
print(Planets.sort_values("gravity",ascending= False))
#23
print(Planets.loc["Mercury"].sort_values())
#Seaborns
#1
import matplotlib.pyplot as plt
import seaborn as sns
tips = sns.load_dataset("tips")
sns.set_style("whitegrid")
g = sns.lmplot(x="tip", y="total_bill", data=tips,aspect=2)
g = (g.set_axis_labels("Tip","Total bill(USD)").set(xlim=(0,10),ylim=(0,100)))
plt.title("title")
plt.show()

#2
print(sns.get_dataset_names())
#3
print(sns.load_dataset("tips").head())
#4
sns.scatterplot(x="total_bill",y="tip",data=tips)
plt.title("Cau4")
plt.show()
#5
sns.scatterplot(x="total_bill",y="tip",data=tips)
sns.set_theme(style="darkgrid", font_scale=1.2)
plt.title("Cau5")
plt.show()
#6
sns.lmplot(x="tip", y="total_bill", data=tips, aspect=2, hue="day")
plt.title("Cau6")
plt.show()
#7

sns.lmplot(x="tip", y="total_bill", data=tips, aspect=2, scatter_kws={"s": tips["size"] * 10})
plt.title("Cau7")
plt.show()
#8
sns.lmplot(x="tip", y="total_bill", data=tips, aspect=2, col="time")
plt.title("Cau8")
plt.show()
#9

sns.lmplot(x="tip", y="total_bill", data=tips, aspect=2, col="time", row="sex")
plt.title("Cau9")
plt.show()