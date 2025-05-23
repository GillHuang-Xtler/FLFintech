from src.fl_base import *

class GradientInversion(FederatedLearningBase):

    def __init__(self, args):
        super(GradientInversion, self).__init__(args)

        self.inversion_client_idx = None

    def get_gradient_from_client(self, idx=0):
        self.inversion_client_idx = idx

        local_net = self.clients[idx]
        local_net.train()

        loss_fn = torch.nn.MSELoss()
        optimiser = torch.optim.Adam(local_net.parameters(), lr=0.01)

        x_train = self.train_set_list[idx][0]
        y_train = self.train_set_list[idx][1]

        x_train = x_train.to(self.device)
        y_train = y_train.to(self.device)

        y_train_pred = local_net(x_train)
        y = loss_fn(y_train_pred, y_train)

        dy_dx_train = torch.autograd.grad(y, local_net.parameters())
        original_grad = list((_.detach().clone() for _ in dy_dx_train))

        return original_grad

    def draw_fig(self, dataset_idx, y_test, y_test_pred, iter=0):
        from pylab import plt
        plt.style.use('seaborn')
        # Visualising the results
        figure, axes = plt.subplots(figsize=(15, 6))
        axes.xaxis_date()

        df = self.df_list[dataset_idx]

        axes.plot(df[len(df) - len(y_test):].index.tolist(), y_test, color='red', label='Real {} Stock Price'.format(self.args.stock_symbols[dataset_idx]))
        axes.plot(df[len(df) - len(y_test):].index.tolist(), y_test_pred, color='blue',
                  label='Inverted {} Stock Price'.format(self.args.stock_symbols[dataset_idx]))

        # axes.xticks(np.arange(0,394,50))
        plt.title('{} Stock Price Inversion'.format(self.args.stock_symbols[dataset_idx]))
        plt.xlabel('Time')
        plt.ylabel('iter {}'.format(self.args.stock_symbols[dataset_idx]))
        plt.legend()
        plt.savefig('./{}/flbase_{}_iter_{}.png'.format(self.args.save_path, self.args.stock_symbols[dataset_idx], iter))
        # plt.show()

    def inversion(self):
        loss_fn = torch.nn.MSELoss()

        original_gradient = self.get_gradient_from_client()
        gt_data_x = self.train_set_list[self.inversion_client_idx][0]
        gt_data_y = self.train_set_list[self.inversion_client_idx][1]

        gt_data_y_init = self.train_set_list[5][1] # choose any other stock data as the initialization

        gt_data_x = gt_data_x.to(self.device)
        gt_data_y = gt_data_y.to(self.device)

        print(gt_data_x.shape, gt_data_y.shape)

        # print(gradient)

        # dummy_data_x = torch.randn(gt_data_x.size()).to(self.device).requires_grad_(True)
        # dummy_data_y = torch.randn(gt_data_y.size()).to(self.device).requires_grad_(True)

        # dummy_data = (torch.randn(gt_data_y.shape[0] + self.look_back - 1)).to(self.device).requires_grad_(True)
        # dummy_data = (2*torch.rand(gt_data_y.shape[0]+self.look_back-1)-1).to(self.device).requires_grad_(True)

        dummy_data = (2 * torch.rand(gt_data_y.shape[0] + self.look_back - 1) - 1).to(self.device)
        dummy_data[self.look_back-1:] = gt_data_y_init.reshape(gt_data_y_init.shape[0])
        dummy_data = dummy_data + (torch.rand(dummy_data.size())*0.1).to(self.device)
        dummy_data = dummy_data.requires_grad_(True)

        dummy_data_temp = dummy_data.unfold(0, self.look_back, 1)
        dummy_data_x = dummy_data_temp[:,:-1]
        dummy_data_x = dummy_data_x.reshape(dummy_data_x.shape[0], dummy_data_x.shape[1], 1)

        dummy_data_y = dummy_data_temp[:,-1]
        dummy_data_y = dummy_data_y.reshape(dummy_data_y.shape[0], 1)

        print("dummy data shape", dummy_data_x.shape, dummy_data_y.shape)

        optimizer = torch.optim.RMSprop([dummy_data], lr=0.001)

        for iter in range(1000):
            def closure():
                optimizer.zero_grad()

                with torch.backends.cudnn.flags(enabled=False):
                    dummy_pred = self.global_net(dummy_data_x)

                # dummy_loss = loss_fn(dummy_pred, dummy_data_y)
                dummy_loss = loss_fn(dummy_pred, gt_data_y)

                dummy_dy_dx = torch.autograd.grad(dummy_loss, self.global_net.parameters(), create_graph=True)

                dummy_dy_dx = list((_ for _ in dummy_dy_dx))

                grad_diff = 0
                for c in range(len(dummy_dy_dx)):
                    gx = dummy_dy_dx[c]
                    gy = original_gradient[c]

                    # grad_diff += ((gx - gy) ** 2).sum()
                    grad_diff += torch.abs(gx - gy).sum()

                grad_diff.backward()
                return grad_diff

            current_loss = optimizer.step(closure)
            with torch.no_grad():
                dummy_data.clamp_(-1,1)

            if iter % 100 == 0:
                print("iter: {}, loss: {}".format(iter, current_loss))
                print("MSE:", ((gt_data_y - dummy_data_y) ** 2).sum())

                self.draw_fig(0, gt_data_y.detach().cpu().numpy(), dummy_data_y.detach().cpu().numpy(), iter=1000)


    def inversion2(self):
        loss_fn = torch.nn.MSELoss()

        original_gradient = self.get_gradient_from_client()
        gt_data_x = self.train_set_list[self.inversion_client_idx][0]
        gt_data_y = self.train_set_list[self.inversion_client_idx][1]

        gt_data_x = gt_data_x.to(self.device)
        gt_data_y = gt_data_y.to(self.device)

        print(gt_data_x.shape, gt_data_y.shape)

        # print(gradient)

        # dummy_data_x = torch.randn(gt_data_x.size()).to(self.device).requires_grad_(True)
        # dummy_data_y = torch.randn(gt_data_y.size()).to(self.device).requires_grad_(True)

        dummy_data_y = (2*torch.rand(gt_data_y.size())-1).to(self.device).requires_grad_(True)

        dummy_data_x = gt_data_x
        # dummy_data_y = gt_data_y

        # optimizer_x = torch.optim.Adam([dummy_data_x], lr=0.01)
        # optimizer_y = torch.optim.Adam([dummy_data_y], lr=0.01)
        optimizer_y = torch.optim.RMSprop([dummy_data_y], lr=0.001)

        for iter in range(1000):
            def closure():
                optimizer_y.zero_grad()

                with torch.backends.cudnn.flags(enabled=False):
                    dummy_pred = self.global_net(dummy_data_x)

                dummy_loss = loss_fn(dummy_pred, dummy_data_y)

                dummy_dy_dx = torch.autograd.grad(dummy_loss, self.global_net.parameters(), create_graph=True)

                dummy_dy_dx = list((_ for _ in dummy_dy_dx))

                grad_diff = 0
                for c in range(len(dummy_dy_dx)):
                    gx = dummy_dy_dx[c]
                    gy = original_gradient[c]

                    # grad_diff += ((gx - gy) ** 2).sum()
                    grad_diff += torch.abs(gx - gy).sum()

                grad_diff.backward()
                return grad_diff

            current_loss = optimizer_y.step(closure)

            if iter % 10 == 0:
                print("iter: {}, loss: {}".format(iter, current_loss))
                print("MSE:", ((gt_data_y - dummy_data_y) ** 2).sum())
