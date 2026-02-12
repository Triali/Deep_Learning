import numpy as np
import math
import matplotlib.pyplot as plt

# -------- activation functions -------
def relu(z):
    return np.maximum(0,z)

def relu_back(xbar, z):
    return np.where(z>0,xbar,0)
    # return xbar if z > 0 else 0

def lrelu(z, a=.01):
    return np.where(z > 0, z, a*z)

def lrelu_back(xbar, z,a=.01):
    return np.where(z > 0, xbar, a*xbar)
    # return xbar if z > 0 else 0

identity = lambda z: z

identity_back = lambda xbar, z: xbar

# ---------- initialization -----------
def initialization(nin, nout, seed = None):
    if seed is not None:
        np.random.seed(seed)
    rng = np.random.default_rng()

    # Generate a 3x4 array with mean (loc) 10 and std dev (scale) 2
    mean = 0
    std_dev = 2/(nin+nout)
    shape = (nout, nin)

    W = rng.normal(loc=mean, scale=std_dev, size=shape)
    b = rng.normal(loc=mean, scale=std_dev, size=(nout,1))
    return W, b
# -------------------------------------


# -------- loss functions -----------
def mse(yhat, y):
    return np.mean((yhat - y)**2)

def mse_back(yhat, y):
    N = yhat.shape[1]
    return 2/N*(yhat-y)
# -----------------------------------

# ------------- Layer ------------
class Layer:

    def __init__(self, nin, nout, activation=identity,seed = None,beta=.9):
        self.W, self.b = initialization(nin, nout,seed)
        self.activation = activation
        self.activation_back = None
        self.beta = beta
        if activation == relu:
            self.activation_back = relu_back
        if activation == lrelu:
            self.activation_back = lrelu_back
        if activation == identity:
            self.activation_back = identity_back

        self.x = None
        self.wbar = None
        self.bbar = None
        self.zbar = None
        self.z = None

    def forward(self, X, train=True):
        # print(f'W shape: {self.W.shape}')
        # print(f'X shape: {X.shape}')
        # print("+")
        # print(f'b shape: {self.b.shape}')
        Z = self.W @ X + self.b
        Xnew = self.activation(Z)
        # print("=")
        # print(f'Z shape: {Z.shape}')
        # print("\n")
        # save cache
        if train:
            self.x = X
            self.z = Z


        return Xnew

    def backward(self, Xnewbar):
        self.zbar = self.activation_back(Xnewbar, self.z)
        self.wbar = self.zbar@self.x.T
        self.bbar = np.sum(self.zbar, axis=1, keepdims=True)
        Xbar = self.W.T@self.zbar
        N = self.x.shape[1]
        return Xbar

    def update(self, alpha):
        self.W = self.W - alpha*self.wbar
        self.b = self.b - alpha*self.bbar

class Network:

    def __init__(self, layers, loss):
        self.layers = layers
        self.loss = loss

        if loss == mse:
            self.loss_back = mse_back
        self.yhat = None
        self.L = None
        self.y = None

    def pred(self,X,train=True):
        i=0

        for layer in self.layers:
            # print(f'Layer {i}')
            i+=1
            X = layer.forward(X, train)
        return X




    def forward(self, X, y, train=True):
        yhat = self.pred(X, train)
        L = self.loss(yhat, y)

        if train:
            self.yhat = yhat
            self.L = L
            self.y = y
        return L, yhat

    def backward(self):
        yhatbar = self.loss_back(self.yhat, self.y)
        Xbar = yhatbar
        for layer in reversed(self.layers):
            Xbar = layer.backward(Xbar)
        return Xbar

    def update(self, alpha):
        for layer in self.layers:
            layer.update(alpha)

# ---------- data preparation ----------------
# Initialize lists for the numeric data and the string data
numeric_data = []

# Read the text file
with open('auto-mpg.data', 'r') as file:
    for line in file:
        # Split the line into columns
        columns = line.strip().split()

        # Check if any of the first 8 columns contain '?'
        if '?' in columns[:8]:
            continue  # Skip this line if there's a missing value

        # Convert the first 8 columns to floats and append to numeric_data
        numeric_data.append([float(value) for value in columns[:8]])

# Convert numeric_data to a numpy array for easier manipulation
numeric_array = np.array(numeric_data)

# Shuffle the numeric array and the corresponding string array
nrows = numeric_array.shape[0]
indices = np.arange(nrows)
np.random.shuffle(indices)
shuffled_numeric_array = numeric_array[indices]

# Split into training (80%) and test (20%) sets
split_index = int(0.8 * nrows)

train_numeric = shuffled_numeric_array[:split_index]
test_numeric = shuffled_numeric_array[split_index:]

# separate inputs/outputs
Xtrain = train_numeric[:, 1:]
ytrain = train_numeric[:, 0]

Xtest = test_numeric[:, 1:]
ytest = test_numeric[:, 0]

# normalize
Xmean = np.mean(Xtrain, axis=0)
Xstd = np.std(Xtrain, axis=0)
ymean = np.mean(ytrain)
ystd = np.std(ytrain)

Xtrain = (Xtrain - Xmean) / Xstd
Xtest = (Xtest - Xmean) / Xstd
ytrain = (ytrain - ymean) / ystd
ytest = (ytest - ymean) / ystd

# reshape arrays (opposite order of pytorch, here we have nx x ns).
# I found that to be more conveient with the way I did the math operations, but feel free to setup
# however you like.
Xtrain = Xtrain.T
Xtest = Xtest.T
ytrain = np.reshape(ytrain, (1, len(ytrain)))
ytest = np.reshape(ytest, (1, len(ytest)))

# ------------------------------------------------------------
model = Network([Layer(7,30,activation=lrelu),Layer(30,20,activation=lrelu),Layer(20,20,activation=lrelu),Layer(20,1)],loss = mse)
alpha = .001
batch_size = 32
train_losses = []
test_losses = []
epochs = 3000
n_samples = Xtrain.shape[1]
for i in range(epochs):
    permutation = np.random.permutation(n_samples)
    epoch_loss = 0
    for i in range(0,n_samples,batch_size):
        idx = permutation[i:i+batch_size]
        Xb = Xtrain[:,idx]
        yb = ytrain[:,idx]
        train_loss,_ = model.forward(Xb, yb, train=True)
        epoch_loss += train_loss
        model.backward()
        model.update(alpha)

    train_losses.append(epoch_loss / (n_samples // batch_size))
    test_loss,_ = model.forward(Xtest, ytest, train=False)
    test_losses.append(test_loss)



# --- inference ----
_, yhat = model.forward(Xtest, ytest, train=False)

# unnormalize
yhat = (yhat * ystd) + ymean
ytest = (ytest * ystd) + ymean# --- inference ----


plt.figure()
plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, epochs + 1), test_losses, label='Testing Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Testing Losses')
plt.legend()
plt.savefig("train_v_test.png")


# plt.figure()
# plt.plot(ytest.T, yhat.T, "o")
# plt.plot([10, 45], [10, 45], "--")

print("avg error (mpg) =", np.mean(np.abs(yhat - ytest)))

plt.show()