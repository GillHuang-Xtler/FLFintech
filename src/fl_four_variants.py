from src.fl_base import *

class FLTwoVariants(FederatedLearningBase):

    def __init__(self, args):
        super(FLTwoVariants, self).__init__(args)

    def load_data(self):
        # loading data for the clients
        symbols = self.args.stock_symbols

        for c in range(self.args.num_clients):
            sym = symbols[c]

            df = stocks_data([sym])
            # print(df)
            df = df[['Close', 'Volume']]
            # df = df[['Volume']]
            # print(df)
            df = df.fillna(method='ffill')

            scaler = MinMaxScaler(feature_range=(-1, 1))
            df['Close'] = scaler.fit_transform(df['Close'].values.reshape(-1, 1))
            scaler2 = MinMaxScaler(feature_range=(-1, 1))
            df['Volume'] = scaler2.fit_transform(df['Volume'].values.reshape(-1, 1))
            scaler3 = MinMaxScaler(feature_range=(-1, 1))
            df['MA50'] = scaler3.fit_transform(df['Volume'].values.reshape(-1, 1))
            scaler4 = MinMaxScaler(feature_range=(-1, 1))
            df['RSI'] = scaler4.fit_transform(df['Volume'].values.reshape(-1, 1))

            self.df_list.append(df)
            self.scaler_list.append([scaler, scaler2])

            look_back = 60  # choose sequence length
            x_train, y_train, x_test, y_test = load_data_from_stocks(df, look_back)

            self.train_set_list.append([x_train, y_train])
            self.test_set_list.append([x_test, y_test])

            print('x_train.shape = ', x_train.shape)
            print('y_train.shape = ', y_train.shape)
            print('x_test.shape = ', x_test.shape)
            print('y_test.shape = ', y_test.shape)

    def set_nets(self, path=None):
        def net_arch():
            input_dim = 2
            hidden_dim = 32
            num_layers = 2
            output_dim = 2
            return LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers, device=self.device)

        self.global_net = net_arch().to(self.device)

        if path is not None:
            state_dict = torch.load(path, map_location=self.device)
            self.global_net.load_state_dict(state_dict)

        for k in range(self.args.num_clients):
            self.clients.append(net_arch().to(self.device))

        self.update_client_parameters(self.global_net.state_dict())

    def predict(self, dataset_idx=0):
        x_test = self.test_set_list[dataset_idx][0].to(self.device)
        y_test = self.test_set_list[dataset_idx][1].to(self.device)
        x_train = self.train_set_list[dataset_idx][0].to(self.device)
        y_train = self.train_set_list[dataset_idx][1].to(self.device)

        scaler, scaler2 = self.scaler_list[dataset_idx]

        y_test_pred = self.global_net(x_test)
        y_train_pred = self.global_net(x_train)

        # invert predictions
        y_train_pred = y_train_pred.detach().cpu().numpy()
        y_train_pred[:,[0]] = scaler.inverse_transform(y_train_pred[:,[0]])
        y_train_pred[:, [1]] = scaler2.inverse_transform(y_train_pred[:, [1]])

        y_train = y_train.detach().cpu().numpy()
        y_train[:, [0]] = scaler.inverse_transform(y_train[:, [0]])
        y_train[:, [1]] = scaler2.inverse_transform(y_train[:, [1]])

        y_test_pred = y_test_pred.detach().cpu().numpy()
        y_test_pred[:, [0]] = scaler.inverse_transform(y_test_pred[:, [0]])
        y_test_pred[:, [1]] = scaler2.inverse_transform(y_test_pred[:, [1]])

        y_test = y_test.detach().cpu().numpy()
        y_test[:, [0]] = scaler.inverse_transform(y_test[:, [0]])
        y_test[:, [1]] = scaler2.inverse_transform(y_test[:, [1]])

        # calculate root mean squared error
        trainScore = math.sqrt(mean_squared_error(y_train[:, 0], y_train_pred[:, 0]))
        print('Train Score: %.2f RMSE' % (trainScore))
        testScore = math.sqrt(mean_squared_error(y_test[:, 0], y_test_pred[:, 0]))
        print('Test Score: %.2f RMSE' % (testScore))
        self.draw_fig(dataset_idx, y_test, y_test_pred)
        self.draw_fig2(dataset_idx, y_test, y_test_pred)

    def draw_fig(self, dataset_idx, y_test, y_test_pred):
        from pylab import plt
        plt.style.use('seaborn')
        # Visualising the results
        figure, axes = plt.subplots(figsize=(15, 6))
        axes.xaxis_date()

        df = self.df_list[dataset_idx]

        axes.plot(df[len(df) - len(y_test):].index.tolist(), y_test[:,0], color='red', label='Real {} Stock Price'.format(self.args.stock_symbols[dataset_idx]))
        axes.plot(df[len(df) - len(y_test):].index.tolist(), y_test_pred[:,0], color='blue',
                  label='Predicted {} Stock Price'.format(self.args.stock_symbols[dataset_idx]))

        plt.title('{} Stock Price Prediction'.format(self.args.stock_symbols[dataset_idx]))
        plt.xlabel('Time')
        plt.ylabel('{} Stock Price'.format(self.args.stock_symbols[dataset_idx]))
        plt.legend()
        plt.savefig('./{}/fltwovariants_{}_pred.png'.format(self.args.save_path, self.args.stock_symbols[dataset_idx]))
        plt.show()

    def draw_fig2(self, dataset_idx, y_test, y_test_pred):
        from pylab import plt
        plt.style.use('seaborn')
        # Visualising the results
        figure, axes = plt.subplots(figsize=(15, 6))
        axes.xaxis_date()

        df = self.df_list[dataset_idx]

        axes.plot(df[len(df) - len(y_test):].index.tolist(), y_test[:,1], color='red', label='Real {} Stock Volume'.format(self.args.stock_symbols[dataset_idx]))
        axes.plot(df[len(df) - len(y_test):].index.tolist(), y_test_pred[:,1], color='blue',
                  label='Predicted {} Stock Volume'.format(self.args.stock_symbols[dataset_idx]))

        plt.title('{} Stock Volume Prediction'.format(self.args.stock_symbols[dataset_idx]))
        plt.xlabel('Time')
        plt.ylabel('{} Stock Volume'.format(self.args.stock_symbols[dataset_idx]))
        plt.legend()
        plt.savefig('./{}/fltwovariants_{}_pred_volume.png'.format(self.args.save_path, self.args.stock_symbols[dataset_idx]))
        plt.show()