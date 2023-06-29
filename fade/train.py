import math
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from fade.util import *
from torch.profiler import profile, record_function, ProfilerActivity


class Trainer:
    def __init__(self, settings, dataset, model, results, cuda=True):
        # parameter enables to suppress CUDA a priori
        self.settings = settings
        self.dataset = dataset
        self.model = model
        self.results = results
        device = 'cuda' if torch.cuda.is_available() and cuda else 'cpu'
        self.device = torch.device(device)
        self.results.log_device(execution_device=device)
        if self.settings[Constants.darts_optimization_method] == Constants.darts_optimization_method_trivial:
            self.loader_train, self.batch_count_train_all = self._get_data_loader(batch_size=self.settings[Constants.optimization_batch_size],
                                                        number_workers=4, ratio=0.9, distinct=True)
        else:
            self.loader_train_weights, self.batch_count_train_weights = self._get_data_loader(batch_size=self.settings[Constants.optimization_batch_size],
                                                     number_workers=4, ratio=0.9, distinct=True)
            self.loader_train_alphas, self.batch_count_train_alphas = self._get_data_loader(batch_size=self.settings[Constants.optimization_batch_size],
                                                    number_workers=4, ratio=0.9, distinct=False)
        self.loader_test, _ = self._get_data_loader(batch_size=self.settings[Constants.optimization_batch_size],
                                                         number_workers=4, ratio=0.1, distinct=True)
        self.model.to(self.device)
        #self.visualization.plot_model(self.model, self.dataset.get_sample()[0].to(self.device))
        self.loss_function = nn.CrossEntropyLoss()
        if self.settings[Constants.optimization_optimizer] == Constants.OPTIMIZER_ADAM:
            self.optimizer_weights = Adam(model.get_parameters(alphas=False), lr=1)
            self.optimizer_alphas = Adam(model.get_parameters(alphas=True), lr=1)
        elif self.settings[Constants.optimization_optimizer] == Constants.OPTIMIZER_SGD:
            self.optimizer_weights = SGD(model.get_parameters(alphas=False), lr=1, momentum=0.8)
            self.optimizer_alphas = SGD(model.get_parameters(alphas=True), lr=1, momentum=0.8)
        else:
            raise NotImplementedError(f"Optimizer {self.settings[Constants.optimization_optimizer]} not supported")
        if self.settings[Constants.darts_deactivate_darts] == 1:
            self.optimizer_all = Adam(model.get_parameters(alphas=False), lr=1)
        else:
            self.optimizer_all = Adam(model.get_parameters(alphas=False) + model.get_parameters(alphas=True), lr=1)
        self.scheduler_all = LinearLR(self.optimizer_all, total_iters=min(5, self.settings[Constants.optimization_number_epochs]),
                                      start_factor=self.settings[Constants.optimization_learning_rate_start], end_factor=self.settings[Constants.optimization_learning_rate_end])
        self.scheduler_weights = LinearLR(self.optimizer_weights, total_iters=min(5, self.settings[Constants.optimization_number_epochs]),
                                          start_factor=self.settings[Constants.optimization_learning_rate_start], end_factor=self.settings[Constants.optimization_learning_rate_end])
        self.scheduler_alphas = LinearLR(self.optimizer_alphas, total_iters=min(5, self.settings[Constants.optimization_number_epochs]),
                                         start_factor=self.settings[Constants.darts_learning_rate_start], end_factor=self.settings[Constants.darts_learning_rate_end])

    def _get_data_loader(self, batch_size, number_workers, ratio, distinct=False):
        dataset = self.dataset.get_subset(ratio, distinct)
        return (InfiniteIterator(DataLoader(dataset, batch_size=batch_size, num_workers=number_workers, shuffle=True)), math.ceil(len(dataset) / batch_size))

    def runtime_logger (self, runtime):
        self.log_runtime = runtime

    def run(self):
        for i in range(1, self.settings[Constants.optimization_number_epochs] + 1):
            self._train_epoch(i, RUNTIME_LOGGER=self.runtime_logger)
            runtime = self.log_runtime
            self.results.log_runtime(runtime)
            loss, accuracy = self._evaluate(i)
            self.results.log_epoch(i)
            self.results.log_lr_weights(self.scheduler_weights.get_last_lr())
            self.results.log_lr_alphas(self.scheduler_alphas.get_last_lr())
            self.results.log_accuracy(accuracy)
            self.results.log_loss(loss)
            for g in self.model.get_graphNN_ids():
                self.results.log_alpha(g, self.model.get_alpha(g))

    def _train_batch (self, loader, optimizer):
        i, o = next(loader)
        i, o = i.to(self.device), o.to(self.device)
        loss = self.loss(i, o)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    @speed_decorator
    def _train_epoch(self, epoch, **kwargs):
        self.model.train()
        if self.settings[Constants.darts_optimization_method] == Constants.darts_optimization_method_trivial:
            number_iterations = self.batch_count_train_all
        else:
            number_iterations = max(self.batch_count_train_weights, self.batch_count_train_alphas)
        for i in range(number_iterations):
            if self.settings[Constants.darts_optimization_method] == Constants.darts_optimization_method_trivial:
                self._train_batch(self.loader_train, self.optimizer_all)
            elif self.settings[Constants.darts_optimization_method] == Constants.darts_optimization_method_first_order:
                self._train_batch(self.loader_train_alphas, self.optimizer_alphas)
                self._train_batch(self.loader_train_weights, self.optimizer_weights)
            elif self.settings[Constants.darts_optimization_method] == Constants.darts_optimization_method_second_order:
                self._train_batch_second_order()
            else:
                raise NotImplementedError()
            if i % 10 == 0:
                print(f"\r{round(100*i/number_iterations):02d}%", end="")
        print("\r     ")
        if self.settings[Constants.darts_optimization_method] == Constants.darts_optimization_method_trivial:
            self.scheduler_all.step()
        else:
            self.scheduler_weights.step()
            self.scheduler_alphas.step()

    def _train_batch_second_order(self):
        p_convs = self.model.get_parameters(alphas=False)
        p_alpha = self.model.get_parameters(alphas=True)
        epsilon=0.001

        def _get_weight_():
            return [p.data.clone() for p in p_convs]

        def _set_weight_(value):
            for (v, p) in zip(value, p_convs):
                p.data.copy_(v)

        def _set_alpha_grad_(grads):
            for (p, g) in zip(p_alpha, grads):
                try:
                    p.grad.copy_(g)
                except:
                    p.grad = g.clone()

        train_input, train_target = next(self.loader_train_weights)
        train_input, train_target = train_input.to(self.device), train_target.to(self.device)
        valid_input, valid_target = next(self.loader_train_alphas)
        valid_input, valid_target = valid_input.to(self.device), valid_target.to(self.device)

        w = _get_weight_()
        train_logits = self.model(train_input)
        train_loss = self.loss(train_input, train_target)
        g_ = torch.autograd.grad(train_loss, p_convs)
        #try:
        #    g_w = [(g + args.momentum * w_opt.state[p]).data.clone() for (g, p) in zip(g_, p_convs)]
        #except:
        g_w = [g.data.clone() for g in g_]

        #w_t = [v-g*args.lr for (v,g) in zip(w, g_w)]
        w_t = [v-g*epsilon for (v,g) in zip(w, g_w)]

        _set_weight_(w_t)
        valid_logits = self.model(valid_input)
        valid_loss = self.loss(valid_input, valid_target)
        g_w_t = [g_.data.clone() for g_ in torch.autograd.grad(valid_loss, p_convs, retain_graph=True)]
        g_a_l = [g_.data.clone() for g_ in torch.autograd.grad(valid_loss, p_alpha)]

        R = 0.01 / math.sqrt(sum((w_*w_).sum() for w_ in w_t))

        w_n = [w_ - R * g_w_t_ for (w_, g_w_t_) in zip(w, g_w_t)]
        w_p = [w_ + R * g_w_t_ for (w_, g_w_t_) in zip(w, g_w_t)]

        _set_weight_(w_n)
        train_logits = self.model(train_input)
        train_loss = self.loss(train_input, train_target)
        g_a_n = [g_a.data.clone() for g_a in torch.autograd.grad(train_loss, p_alpha)]

        _set_weight_(w_p)
        train_logits = self.model(train_input)
        train_loss = self.loss(train_input, train_target)
        g_a_p = [g_a.data.clone() for g_a in torch.autograd.grad(train_loss, p_alpha)]

        _set_weight_(w)

        g_a_r = [(gr-gl)/(2*R) for (gr, gl) in zip(g_a_p, g_a_n) ]
        g_a = [gl - epsilon*gr for (gl, gr) in zip(g_a_l, g_a_r)]
        _set_alpha_grad_(g_a)
        self.optimizer_alphas.step()

        train_logits = self.model(train_input)
        train_loss = self.loss(train_input, train_target)
        self.optimizer_weights.zero_grad()
        train_loss.backward()
        self.optimizer_weights.step()
    
    def _evaluate(self, epoch):
        self.model.eval()
        with torch.no_grad():
            loss = 0
            accuracy = 0
            batchCount = 1
            for i, o in self.loader_test:
                i, o = i.to(self.device), o.to(self.device)
                loss = ((batchCount - 1) * loss + self.loss(i, o)) / batchCount
                accuracy = ((batchCount - 1) * accuracy + self.accuracy(i, o)) / batchCount
        self.model.train()
        return (get_numeric(loss), get_numeric(accuracy))

    def loss(self, i, o):
        if isinstance(self.loss_function, nn.CrossEntropyLoss):
            _, o = o.max(dim=-1) #get indices from one-hot encoding
        return self.loss_function(self.model(i), o)

    def accuracy (self, i, o):
        return torch.true_divide((self.model(i).max(dim=-1)[1] == o.max(dim=-1)[1]).sum(), i.size()[0])