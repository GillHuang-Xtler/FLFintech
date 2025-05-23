import torch
import copy
import logging
import os
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from src.model import LSTM
from src.common import stocks_data, load_data_from_stocks

class FederatedLearningBase(object):

    def __init__(self, args):
        self.args = args

        self.global_net = None

        self.df_list = []
        self.scaler_list = []

        self.test_set_list = [] # [[client 1 x data, client 1 y data], ..., []]
        self.train_set_list = [] # [[client 1 x data, client 1 y data], ..., []]

        self.device = torch.device("cuda:0" if self.args.cuda else "cpu")
        self.look_back = 60

        self.clients = []

        self.set_logging()

    def set_logging(self):
        if not os.path.exists(self.args.save_path):
            os.makedirs(self.args.save_path)

        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[
                logging.FileHandler('{}/fl.log'.format(self.args.save_path)),
                logging.StreamHandler()
            ]
        )
        logging.info(self.args)

    def load_data(self):
        # loading data for the clients
        symbols = self.args.stock_symbols

        for c in range(self.args.num_clients):
            sym = symbols[c]

            df = stocks_data([sym])
            # print(df)
            df = df[['Close']]
            # df = df[['Volume']]
            # print(df)
            df = df.fillna(method='ffill')

            scaler = MinMaxScaler(feature_range=(-1, 1))
            df['Close'] = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

            self.df_list.append(df)
            self.scaler_list.append(scaler)

            look_back = self.look_back  # choose sequence length
            x_train, y_train, x_test, y_test = load_data_from_stocks(df, look_back)

            self.train_set_list.append([x_train, y_train])
            self.test_set_list.append([x_test, y_test])

            # print('x_train.shape = ', x_train.shape)
            # print('y_train.shape = ', y_train.shape)
            # print('x_test.shape = ', x_test.shape)
            # print('y_test.shape = ', y_test.shape)


    def set_nets(self, path=None):
        def net_arch():
            input_dim = 1
            hidden_dim = 32
            num_layers = 2
            output_dim = 1
            return LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers, device=self.device)

        self.global_net = net_arch().to(self.device)

        if path is not None:
            state_dict = torch.load(path, map_location=self.device)
            self.global_net.load_state_dict(state_dict)

        for k in range(self.args.num_clients):
            self.clients.append(net_arch().to(self.device))

        self.update_client_parameters(self.global_net.state_dict())

    def update_client_parameters(self, new_params):
        for k in range(self.args.num_clients):
            self.clients[k].load_state_dict(copy.deepcopy(new_params), strict=True)

    def aggregate(self):
        new_params = {}
        net_param_name_list = self.clients[0].state_dict().keys()
        num_clients = self.args.num_clients
        for name in net_param_name_list:
            new_params[name] = sum([self.clients[k].state_dict()[name].data for k in range(num_clients)]) / num_clients

        self.global_net.load_state_dict(copy.deepcopy(new_params), strict=True)
        return new_params

    def federated_train(self):

        for r in range(self.args.global_rounds):

            for k in range(self.args.num_clients):
                self.local_update(k, r)

            new_params = self.aggregate()
            self.update_client_parameters(new_params)

        torch.save(self.global_net.state_dict(), '{}/final_global_model.pth'.format(self.args.save_path))

    def local_update(self, idx, global_round):
        local_net = self.clients[idx]
        local_net.train()

        loss_fn = torch.nn.MSELoss()
        optimiser = torch.optim.Adam(local_net.parameters(), lr=0.01)

        x_train = self.train_set_list[idx][0]
        y_train = self.train_set_list[idx][1]

        x_train = x_train.to(self.device)
        y_train = y_train.to(self.device)

        for c in range(10):
            # Forward pass
            y_train_pred = local_net(x_train)

            loss = loss_fn(y_train_pred, y_train)

            # print("debug *********")
            # print(x_train.shape)
            # print(y_train_pred.shape)
            # print(y_train.shape)

            # Zero out gradient, else they will accumulate between epochs
            optimiser.zero_grad()

            # Backward pass
            loss.backward()

            # Update parameters
            optimiser.step()

        if global_round % 10 == 0:
            print("Global Round:", global_round, "Client idx:", idx, "MSE:", loss.item())

    def predict(self, dataset_idx=0):
        x_test = self.test_set_list[dataset_idx][0].to(self.device)
        y_test = self.test_set_list[dataset_idx][1].to(self.device)
        x_train = self.train_set_list[dataset_idx][0].to(self.device)
        y_train = self.train_set_list[dataset_idx][1].to(self.device)

        scaler = self.scaler_list[dataset_idx]

        y_test_pred = self.global_net(x_test)
        y_train_pred = self.global_net(x_train)

        # invert predictions
        y_train_pred = scaler.inverse_transform(y_train_pred.detach().cpu().numpy())
        y_train = scaler.inverse_transform(y_train.detach().cpu().numpy())
        y_test_pred = scaler.inverse_transform(y_test_pred.detach().cpu().numpy())
        y_test = scaler.inverse_transform(y_test.detach().cpu().numpy())

        # calculate root mean squared error
        trainScore = math.sqrt(mean_squared_error(y_train[:, 0], y_train_pred[:, 0]))
        print('Train Score: %.2f RMSE' % (trainScore))
        testScore = math.sqrt(mean_squared_error(y_test[:, 0], y_test_pred[:, 0]))
        print('Test Score: %.2f RMSE' % (testScore))
        self.draw_fig(dataset_idx, y_test, y_test_pred)

    def draw_fig(self, dataset_idx, y_test, y_test_pred):
        from pylab import plt
        plt.style.use('seaborn')
        # Visualising the results
        figure, axes = plt.subplots(figsize=(15, 6))
        axes.xaxis_date()

        df = self.df_list[dataset_idx]

        axes.plot(df[len(df) - len(y_test):].index.tolist(), y_test, color='red', label='Real {} Stock Price'.format(self.args.stock_symbols[dataset_idx]))
        axes.plot(df[len(df) - len(y_test):].index.tolist(), y_test_pred, color='blue',
                  label='Predicted {} Stock Price'.format(self.args.stock_symbols[dataset_idx]))

        # axes.xticks(np.arange(0,394,50))
        plt.title('{} Stock Price Prediction'.format(self.args.stock_symbols[dataset_idx]))
        plt.xlabel('Time')
        plt.ylabel('{} Stock Price'.format(self.args.stock_symbols[dataset_idx]))
        plt.legend()
        plt.savefig('./{}/flbase_{}_pred.png'.format(self.args.save_path, self.args.stock_symbols[dataset_idx]))
        plt.show()

