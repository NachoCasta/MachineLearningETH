from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model

with open("train.csv") as f:
    data = []
    labels = []
    f.readline()
    for line in f:
        line = line.strip().split(",")
        y = int(float(line[1])*10**12)
        X = [int(float(n)*10**12) for n in line[2:]]
        labels.append(y)
        data.append(X)
    
with open("test.csv") as f:
    test = []
    f.readline()
    for line in f:
        line = line.strip().split(",")
        X = [int(float(n)*10**12) for n in line[1:]]
        test.append(X)



##clas = KNeighborsClassifier()
clas = linear_model.LinearRegression()
clas.fit(data, labels)

with open("predictions.csv", "w") as f:
    f.write("Id,y\n")
    for i, (X, p) in enumerate(zip(test, clas.predict(test))):
        y = sum(X)/10
        y /= 10**12
        p /= 10**12
        f.write("{},{}\n".format(i+10000, y))
    ##    print(abs(y-p))
    ##    print("Real:", y)
    ##    print("Pred:", p)
    ##    print("---------------")
        
    
