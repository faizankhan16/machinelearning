
# import libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split

# 1. Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)


# 2. Summarize the dataset
# shape
print(dataset.shape)
# head
print(dataset.head(20))
# descriptions
print(dataset.describe())
# class distribution
print(dataset.groupby('class').size())


# 3. Visualize the dataset
# univariate plots
#Box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()
#Histogram
dataset.hist()
pyplot.show()

# multivariate plots
scatter_matrix(dataset)
pyplot.show()


# 5. Evaluate some algorithms
#Create a test dataset
array = dataset.values
X = array[:,:-1]
y = array[:,-1]
X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

