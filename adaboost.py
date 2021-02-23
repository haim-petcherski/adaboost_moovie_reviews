#################################
# Your name: Haim Petcherski
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib.
from matplotlib import pyplot as plt
import numpy as np
from process_data import parse_data

np.random.seed(7)


def run_adaboost(X_train, y_train, T):
    """
    Returns:

        hypotheses :
            A list of T tuples describing the hypotheses chosen by the algorithm.
            Each tuple has 3 elements (h_pred, h_index, h_theta), where h_pred is
            the returned value (+1 or -1) if the count at index h_index is <= h_theta.

        alpha_vals :
            A list of T float values, which are the alpha values obtained in every
            iteration of the algorithm.
    """

    hypotheses = []
    alpha_vals = []
    thetas_per_field = []
    for field in range(len(y_train)):
        projection = [data[field] for data in X_train]
        thetas_per_field.append(calculate_thetas(projection))


    n = len(y_train)
    D = [(1 / n) for i in range(n)]
    for i in range(T):
        w_learner = update_WL(X_train, y_train, D, thetas_per_field)
        D = updateDist(D, w_learner[-1])
        h_t = w_learner[:3]
        hypotheses.append(h_t)
        alpha_vals.append((1 / 2) * np.log(((1 - w_learner[-2]) / w_learner[-2])))
    return hypotheses, alpha_vals


##############################################
# You can add more methods here, if needed.

def updateDist(D_t, e_vector):
    deVec = [(tup[0] * tup[1]) for tup in zip(D_t, e_vector)]
    Zt = np.sum(deVec)
    return list(map(lambda x: (x / Zt), deVec))




def calculate_thetas(projection):
    list_thetas = list(set(list(projection)))
    list_thetas.sort()
    thetas = [list_thetas[0] - 1]
    for i in range(1, len(list_thetas)):
        thetas.append(((list_thetas[i - 1] + list_thetas[i]) / 2))
    thetas.append(1 + list_thetas[-1])
    return thetas

def update_WL(X_train, y_train, D, thetas_per_field):
    favorite_h_t = (0, 0, 0, 2, [])  # for (h_pred, h_index, h_theta, error, e_vector)
    data = list(X_train)
    for field in range(len(y_train)):
        projection = [v[field] for v in data]
        thetas = thetas_per_field[field]
        for theta in thetas:
            negative_predict = calc_weak_error(-1, y_train, D, projection, theta)
            positive_predict = calc_weak_error(1, y_train, D, projection, theta)
            if positive_predict[0] < favorite_h_t[3]:
                favorite_h_t = (1, field, theta, positive_predict[0], positive_predict[1])
            if negative_predict[0] < favorite_h_t[3]:
                favorite_h_t = (-1, field, theta, negative_predict[0], negative_predict[1])
    return favorite_h_t


def calc_weak_error(predict, y_train, D, projection, theta):
    predictions = list(map(lambda x: predict if (x <= theta) else (-1 * predict), projection))
    truth_times_prediction = [tup[0] * tup[1] for tup in zip(predictions, y_train)]
    epsilon_t = 0
    for i, res in enumerate(truth_times_prediction):
        if res == -1:
            epsilon_t += D[i]
    w_t = (1 / 2) * np.log(((1 - epsilon_t) / epsilon_t))
    e_vector = [np.exp(((-1) * w_t * result)) for result in truth_times_prediction]
    return epsilon_t, e_vector




def error_calculation(X, y, hypotheses, alpha_vals):
    sum_alpha_mul_hypo = np.zeros(len(X))
    total_error = []
    for m in range(len(hypotheses)):
        error = 0
        for i in range(len(X)):
            if X[i][hypotheses[m][1]] > hypotheses[m][2]:
                sum_alpha_mul_hypo[i] += -hypotheses[m][0] * alpha_vals[m]
            else:
                sum_alpha_mul_hypo[i] += hypotheses[m][0]*alpha_vals[m]
            if sum_alpha_mul_hypo[i]*y[i]>=0: #sign(prediction*label)=1
                continue
            else:
                error += 1
        total_error.append(error/len(X))
    return total_error



def loss_calculation(y, X, hypotheses, alpha_vals):
    sum_loss = np.zeros(len(X))
    total_error = np.zeros(len(hypotheses))
    for m in range(len(hypotheses)):
        for i in range(len(X)):
            if X[i][hypotheses[m][1]] > hypotheses[m][2]:
                sum_loss[i] += (-hypotheses[m][0]) * alpha_vals[m] * (-1) * y[i]
            else:
                sum_loss[i] += hypotheses[m][0]*alpha_vals[m] * (-1) * y[i]

        expo_loss = np.exp(sum_loss)

        total_error[m]=(1/len(X))*np.sum(expo_loss, axis=0)
    return total_error






##############################################


def main():
    data = parse_data()
    if not data:
        return
    (X_train, y_train, X_test, y_test, vocab) = data
    T = 80
    hypotheses, alpha_vals = run_adaboost(X_train, y_train, T)



    #section a:
    test_set_error = error_calculation(X_test, y_test, hypotheses, alpha_vals)
    train_set_error = error_calculation(X_train,y_train, hypotheses,alpha_vals)
    plt.plot(np.arange(1,1+T), test_set_error, 'b', label='test set error')
    plt.plot(np.arange(1,1+T), train_set_error, 'k', label='train set error')
    plt.xlabel('t')
    plt.ylabel('error')
    plt.legend()
    #plt.show()

    # section b:

    for i in range(10):
        print(hypotheses[i])

    #section c:
    test_set_error = loss_calculation(y_test, X_test, hypotheses, alpha_vals)
    train_set_error = loss_calculation(y_train, X_train, hypotheses,alpha_vals)
    plt.plot(np.arange(1,1+T), test_set_error, 'b', label='test set error')
    plt.plot(np.arange(1,1+T), train_set_error, 'k', label='train set error')
    plt.xlabel('t')
    plt.ylabel('error')
    plt.legend()
    #plt.show()
