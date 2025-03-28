"""Functions for training, evaluating and using a neural network."""

import numpy as np

from neuron import derivative_of_sigmoid
from neuron import evaluate_neuron


def evaluate_network(x, w, b, Wt, bt):
    """Calculate the output of a neural network with two input nodes and
    two hidden nodes. The sigmoid function is used as activation-function.

    Args:
        x:  input (numpy array of size [batch_size, 2])
        w:  weights of output layer (numpy array of size [2])
        b:  bias of output layer (numpy array of size [1])
        Wt: weights of hidden layer (numpy array of size [2, 2])
        bt: bias of hidden layer (numpy array of size [2])

    Returns:
        y:   output of the output layer (numpy arrays of size [batch_size])
        a_y: pre-activation of the output layer (numpy arrays
             of size [batch_size])
        h:   output of the hidden layer (numpy arrays of size
             [batch_size, 2])
        a_h: pre-activation of the hidden layer (numpy arrays of
             size [batch_size, 2])
    """

    assert x.ndim == 2
    assert x.shape[1] == 2
    assert w.shape == (2,)
    assert b.shape == (1,) or b.shape == ()
    assert Wt.shape == (2, 2)
    assert bt.shape == (2,)

    yt_1,at_1 = evaluate_neuron(x,Wt[:,0],bt[0]) #first hidden neuron
    yt_2,at_2 = evaluate_neuron(x,Wt[:,1],bt[1]) #2nd
    h = np.column_stack((yt_1,yt_2))
    a_h = np.column_stack((at_1,at_2))
    
    y, a_y = evaluate_neuron(h,w,b)


    return y, a_y, h, a_h


def loss_function(x, t, w, b, Wt, bt):
    """Calculate the loss of the neural network.

    Args:
        x:  input (numpy array of size [batch_size, 2])
        t:  target, desired output (numpy array of size [batch_size])
        w:  weights of output layer (numpy array of size [2])
        b:  bias of output layer (numpy array of size [1])
        Wt: weights of hidden layer (numpy array of size [2, 2])
        bt: bias of hidden layer (numpy array of size [2])
    Returns:
        l:  loss (scalar)
    """

    assert x.ndim == 2
    assert x.shape[1] == 2
    assert t.ndim == 1
    assert t.shape[0] == x.shape[0]
    assert w.shape == (2,)
    assert b.shape == (1,) or b.shape == ()
    assert Wt.shape == (2, 2)
    assert bt.shape == (2,)

    y_hat, _, _, _ = evaluate_network(x,w,b,Wt,bt)
    diff = y_hat - t
    loss = 1/(2*x.shape[0]) * np.dot(diff,diff)
    return loss


def update_weights(x, t, w, b, Wt, bt, lr):
    """Update the weights and the biases by applying stochastic gradient descent.

    Args:
        x:  input (numpy array of size [batch_size, 2])
        t:  target, desired output (numpy array of size [batch_size])
        w:  weights (numpy array of size [2])
        b:  bias (numpy array of size [1])
        Wt: weights of hidden layer (numpy array of size [2, 2])
        bt: bias of hidden layer (numpy array of size [2])
        lr: learning rate

    Returns:
        w_new, b_new, Wt_new, bt_new: updated weights and biases
    """

    assert x.ndim == 2
    assert x.shape[1] == 2
    assert t.ndim == 1
    assert t.shape[0] == x.shape[0]
    assert w.shape == (2,)
    assert b.shape == (1,) or b.shape == ()
    assert Wt.shape == (2, 2)
    assert bt.shape == (2,)

    y, a_y, h, a_h = evaluate_network(x,w,b,Wt,bt)
    
    gw = 1/x.shape[0] * np.dot((y-t) * derivative_of_sigmoid(a_y),h)
    gb = 1/x.shape[0] * np.dot((y-t), derivative_of_sigmoid(a_y))

    # 1/x.shape[0] * np.dot((y-t) * derivative_of_sigmoid(a_y),w*derivative_of_sigmoid(a_h)*x)
    d_Wt = (x.T @ ((((t - y) * y * (1 - y))[:, None]) * w[None, :] * h * (1 - h))) / x.shape[0]
    gwb = 1/x.shape[0] * np.dot((y-t) * derivative_of_sigmoid(a_y), w*derivative_of_sigmoid(a_h))

    w_new = w - lr * gw
    b_new = b - lr * gb
    Wt_new = Wt + lr * d_Wt
    bt_new = bt -lr * gwb

    return w_new, b_new, Wt_new, bt_new


def evaluate_prediction(x, t, w, b, Wt, bt):
    """Evaluate the prediction (predicted class) of the network.

    Args:
        x:  input (numpy array of size [batch_size, 2])
        t:  target, desired output (numpy array of size [batch_size])
        w:  weights (numpy array of size [2])
        b:  bias (numpy array of size [1])
        Wt: weights of hidden layer (numpy array of size [2, 2])
        bt: bias of hidden layer (numpy array of size [2])

    Returns:
        prediction: predicted output (numpy array of size [batch_size])
        accuracy: proportion of correct predictions
    """

    assert x.ndim == 2
    assert x.shape[1] == 2
    assert t.ndim == 1
    assert t.shape[0] == x.shape[0]
    assert w.shape == (2,)
    assert b.shape == (1,) or b.shape == ()
    assert Wt.shape == (2, 2)
    assert bt.shape == (2,)

    y, _,_,_ = evaluate_network(x,w,b,Wt,bt)
    prediction = (y >= 0.5).astype(bool)
    accuracy = np.mean(prediction == t)
    
    return prediction, accuracy


# Tests for the defined functions
if __name__ == "__main__":

    print("Start unit test for module network.py.")

    # test values
    x = np.array(
        [[1.56, 2.58], [-4.64, 2.43], [3.49, -1.08], [4.34, 1.55], [1.79, -3.29]]
    )
    w = np.array([2.06, -4.68])
    b = np.array(2.23)
    Wt = np.array([[2.55, 1.80], [-2.24, 1.55]])
    bt = np.array([1.46, 2.09])
    t = np.array([1, 0, 0, 0, 1])
    a_h_target = np.array(
        [
            [-0.3412, 8.8970],
            [-15.8152, -2.4955],
            [12.7787, 6.6980],
            [9.0550, 12.3045],
            [13.3941, 0.2125],
        ]
    )
    h_target = np.array(
        [
            [0.415518012198846, 0.999863220003701],
            [1.35377300886619e-07, 0.0761742494745588],
            [0.999997179800188, 0.998768143152960],
            [0.999883208095262, 0.999995468712902],
            [0.999998475892438, 0.552925988132906],
        ]
    )
    a_y_target = np.array(
        [
            -1.59339276448770,
            1.87350479133630,
            -0.384240719567465,
            -0.390219384900142,
            1.70230323587642,
        ]
    )
    y_target = np.array(
        [
            0.168907094034165,
            0.866863290425436,
            0.405104495610523,
            0.403664489370525,
            0.845835311412465,
        ]
    )
    loss_target = 0.179298880618386
    da_y = np.array(
        [
            0.140377487619099,
            0.115411326138222,
            0.240994843246667,
            0.240719469391758,
            0.130397937380243,
        ]
    )
    lr = 0.75
    w_new_target = np.array([2.04106902058297, -4.69117981431689])
    b_new_target = np.array(2.20628884802289)
    Wt_new_target = np.array(
        [[2.56364264768371, 1.77110117610734], [-2.21741694559693, 1.57336762652625]]
    )
    bt_new_target = np.array([1.46875162265565, 2.09152728628318])
    prediction_target = np.array([0, 1, 0, 0, 1])
    accuracy_target = 0.6

    # test function evaluate_network
    y, a_y, h, a_h = evaluate_network(x, w, b, Wt, bt)
    assert np.all(np.abs(a_h_target - a_h) < 10e-15)
    assert np.all(np.abs(h_target - h) < 10e-15)
    assert np.all(np.abs(a_y_target - a_y) < 10e-15)
    assert np.all(np.abs(y_target - y) < 10e-15)

    # test function loss_function
    loss = loss_function(x, t, w, b, Wt, bt)
    assert np.abs(loss_target - loss) < 10e-15

    # test function update_weights
    w_new, b_new, Wt_new, bt_new = update_weights(x, t, w, b, Wt, bt, lr)
    assert np.all(np.abs(w_new_target - w_new) < 10e-15)
    assert np.all(np.abs(b_new_target - b_new) < 10e-15)
    assert np.all(np.abs(bt_new_target - bt_new) < 10e-15)
    assert np.all(np.abs(Wt_new_target - Wt_new) < 10e-15)

    # test function evaluate_prediction
    prediction, accuracy = evaluate_prediction(x, t, w, b, Wt, bt)
    assert np.all(np.abs(prediction_target - prediction) < 10e-15)
    assert np.abs(accuracy_target - accuracy) < 10e-15

    print("Unit test was successful.")
