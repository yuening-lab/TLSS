class cola_gnn(nn.Module):  
    def __init__(self, args, data): 
        super().__init__()
        self.x_h = 1 
        self.f_h = data.m   
        self.m = data.m  
        self.d = data.d 
        self.w = args.window
        self.h = args.horizon
        self.adj = data.adj
        self.o_adj = data.orig_adj
        if args.cuda:
            self.adj = sparse_mx_to_torch_sparse_tensor(normalize_adj2(data.orig_adj.cpu().numpy())).to_dense().cuda()
        else:
            self.adj = sparse_mx_to_torch_sparse_tensor(normalize_adj2(data.orig_adj.cpu().numpy())).to_dense()
        self.dropout = args.dropout
        self.n_hidden = args.n_hidden
        half_hid = int(self.n_hidden/2)
        self.V = Parameter(torch.Tensor(half_hid))
        self.bv = Parameter(torch.Tensor(1))
        self.W1 = Parameter(torch.Tensor(half_hid, self.n_hidden))
        self.b1 = Parameter(torch.Tensor(half_hid))
        self.W2 = Parameter(torch.Tensor(half_hid, self.n_hidden))
        self.act = F.elu 
        self.Wb = Parameter(torch.Tensor(self.m,self.m))
        self.wb = Parameter(torch.Tensor(1))
        self.k = args.k
        self.conv = nn.Conv1d(1, self.k, self.w)
        long_kernal = self.w//2
        self.conv_long = nn.Conv1d(self.x_h, self.k, long_kernal, dilation=2)
        long_out = self.w-2*(long_kernal-1)
        self.n_spatial = 10  
        self.conv1 = GraphConvLayer((1+long_out)*self.k, self.n_hidden) # self.k
        self.conv2 = GraphConvLayer(self.n_hidden, self.n_spatial)
 
        if args.rnn_model == 'LSTM':
            self.rnn = nn.LSTM( input_size=self.x_h, hidden_size=self.n_hidden, num_layers=args.n_layer, dropout=args.dropout, batch_first=True, bidirectional=args.bi)
        elif args.rnn_model == 'GRU':
            self.rnn = nn.GRU( input_size=self.x_h, hidden_size=self.n_hidden, num_layers=args.n_layer, dropout=args.dropout, batch_first=True, bidirectional=args.bi)
        elif args.rnn_model == 'RNN':
            self.rnn = nn.RNN( input_size=self.x_h, hidden_size=self.n_hidden, num_layers=args.n_layer, dropout=args.dropout, batch_first=True, bidirectional=args.bi)
        else:
            raise LookupError (' only support LSTM, GRU and RNN')

        hidden_size = (int(args.bi) + 1) * self.n_hidden
        self.out = nn.Linear(hidden_size + self.n_spatial, 1)  

        self.residual_window = 0
        self.ratio = 1.0
        if (self.residual_window > 0):
            self.residual_window = min(self.residual_window, args.window)
            self.residual = nn.Linear(self.residual_window, 1) 
        self.init_weights()
     
    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data) # best
            else:
                stdv = 1. / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)

    def forward(self, x, feat=None):
        '''
        Args:  x: (batch, time_step, m)  
            feat: [batch, window, dim, m]
        Returns: (batch, m)
        ''' 
        b, w, m = x.size()
        orig_x = x 
        x = x.permute(0, 2, 1).contiguous().view(-1, x.size(1), 1) 
        r_out, hc = self.rnn(x, None)
        last_hid = r_out[:,-1,:]
        last_hid = last_hid.view(-1,self.m, self.n_hidden)
        out_temporal = last_hid  # [b, m, 20]
        hid_rpt_m = last_hid.repeat(1,self.m,1).view(b,self.m,self.m,self.n_hidden) # b,m,m,w continuous m
        hid_rpt_w = last_hid.repeat(1,1,self.m).view(b,self.m,self.m,self.n_hidden) # b,m,m,w continuous w one window data
        a_mx = self.act( hid_rpt_m @ self.W1.t()  + hid_rpt_w @ self.W2.t() + self.b1 ) @ self.V + self.bv # row, all states influence one state 
        a_mx = F.normalize(a_mx, p=2, dim=1, eps=1e-12, out=None)
        r_l = []
        r_long_l = []
        h_mids = orig_x
        for i in range(self.m):
            h_tmp = h_mids[:,:,i:i+1].permute(0,2,1).contiguous() 
            r = self.conv(h_tmp) # [32, 10/k, 1]
            r_long = self.conv_long(h_tmp)
            r_l.append(r)
            r_long_l.append(r_long)
        r_l = torch.stack(r_l,dim=1)
        r_long_l = torch.stack(r_long_l,dim=1)
        r_l = torch.cat((r_l,r_long_l),-1)
        r_l = r_l.view(r_l.size(0),r_l.size(1),-1)
        r_l = torch.relu(r_l)
        adjs = self.adj.repeat(b,1)
        adjs = adjs.view(b,self.m, self.m)
        c = torch.sigmoid(a_mx @ self.Wb + self.wb)
        a_mx = adjs * c + a_mx * (1-c) 
        adj = a_mx 
        x = r_l 
        x = F.relu(self.conv1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        out_spatial = F.relu(self.conv2(x, adj))
        #print("out_spatial shape", out_spatial.shape)
        out = torch.cat((out_spatial, out_temporal),dim=-1)#(32,49,10)+(32,49,20)
        #print("out shape",out.shape)
        out = self.out(out)
        out = torch.squeeze(out)

        if (self.residual_window > 0):
            z = orig_x[:, -self.residual_window:, :]; #Step backward # [batch, res_window, m]
            z = z.permute(0,2,1).contiguous().view(-1, self.residual_window); #[batch*m, res_window]
            z = self.residual(z); #[batch*m, 1]
            z = z.view(-1,self.m); #[batch, m]
            out = out * self.ratio + z; #[batch, m]

        return out, None


class ARMA(nn.Module): 
    def __init__(self, args, data):
        super(ARMA, self).__init__()
        self.m = data.m
        self.w = args.window
        self.n = 2 # larger worse
        self.w = 2*self.w - self.n + 1 
        self.weight = Parameter(torch.Tensor(self.w, self.m)) # 20 * 49
        self.weight1 = Parameter(torch.Tensor(self.w, self.m))
        self.bias = Parameter(torch.zeros(self.m)) # 49
        nn.init.xavier_normal(self.weight)
        nn.init.xavier_normal(self.weight1)

        args.output_fun = None;
        self.output = None
        if (args.output_fun == 'sigmoid'):
            self.output = F.sigmoid;
        if (args.output_fun == 'tanh'):
            self.output = F.tanh;

    def forward(self, x, x1):
        x_o = x
        x = x.permute(0,2,1).contiguous()
        n = self.n
        cumsum = torch.cumsum(x,dim=-1)
        cumsum[:,:,n:] = cumsum[:,:,n:] - cumsum[:,:,:-n]
        x = cumsum[:,:,n - 1:] / n
        x = x.permute(0,2,1).contiguous()
        x = torch.cat((x_o,x), dim=1)

        x_o1 = x1
        x1 = x1.permute(0,2,1).contiguous()
        n = self.n
        cumsum = torch.cumsum(x1,dim=-1)
        cumsum[:,:,n:] = cumsum[:,:,n:] - cumsum[:,:,:-n]
        x1 = cumsum[:,:,n - 1:] / n
        x1 = x1.permute(0,2,1).contiguous()
        x1 = torch.cat((x_o1,x1), dim=1)
        x = torch.sum(x * self.weight, dim=1) + torch.sum(x1 * self.weight1, dim=1) + self.bias
        if (self.output != None):
            x = self.output(x)
        return x, None

class ARMA_news(nn.Module): 
    def __init__(self, args, data):
        super(ARMA_news, self).__init__()
        self.m = data.m
        self.w = args.window
        self.n = 2 # larger worse
        self.w = 2*self.w - self.n + 1 
        self.weight = Parameter(torch.Tensor(self.w, self.m)) # 20 * 49
        self.weight1 = Parameter(torch.Tensor(self.w, self.m))
        self.weight2 = Parameter(torch.Tensor(self.w, self.m))
        self.weight3 = Parameter(torch.Tensor(self.w, self.m*self.m))
        self.bias = Parameter(torch.zeros(self.m)) # 49
        self.fc = nn.Linear(self.m*self.m, self.m)
        nn.init.xavier_normal_(self.weight)
        nn.init.xavier_normal_(self.weight1)
        nn.init.xavier_normal_(self.weight2)
        nn.init.xavier_normal_(self.weight3)

        args.output_fun = None;
        self.output = None
        if (args.output_fun == 'sigmoid'):
            self.output = F.sigmoid;
        if (args.output_fun == 'tanh'):
            self.output = F.tanh;

    def forward(self, x, x1, x2, x3):
        x_o = x
        x = x.permute(0,2,1).contiguous()
        n = self.n
        cumsum = torch.cumsum(x,dim=-1)
        cumsum[:,:,n:] = cumsum[:,:,n:] - cumsum[:,:,:-n]
        x = cumsum[:,:,n - 1:] / n
        x = x.permute(0,2,1).contiguous()
        x = torch.cat((x_o,x), dim=1)

        x_o1 = x1
        x1 = x1.permute(0,2,1).contiguous()
        n = self.n
        cumsum = torch.cumsum(x1,dim=-1)
        cumsum[:,:,n:] = cumsum[:,:,n:] - cumsum[:,:,:-n]
        x1 = cumsum[:,:,n - 1:] / n
        x1 = x1.permute(0,2,1).contiguous()
        x1 = torch.cat((x_o1,x1), dim=1)

        x_o2 = x2
        x2 = x2.permute(0,2,1).contiguous()
        n = self.n
        cumsum = torch.cumsum(x2,dim=-1)
        cumsum[:,:,n:] = cumsum[:,:,n:] - cumsum[:,:,:-n]
        x2 = cumsum[:,:,n - 1:] / n
        x2 = x2.permute(0,2,1).contiguous()
        x2 = torch.cat((x_o2,x2), dim=1)

        x_o3 = x3
        x3 = x3.permute(0,2,1).contiguous()
        n = self.n
        cumsum = torch.cumsum(x3,dim=-1)
        cumsum[:,:,n:] = cumsum[:,:,n:] - cumsum[:,:,:-n]
        x3 = cumsum[:,:,n - 1:] / n
        x3 = x3.permute(0,2,1).contiguous()
        x3 = torch.cat((x_o3,x3), dim=1)

        x3 = self.fc(torch.sum(x3 * self.weight3, dim=1))
        x = torch.sum(x * self.weight, dim=1) + torch.sum(x1 * self.weight1, dim=1) + torch.sum(x2 * self.weight2, dim=1) + x3 + self.bias
        if (self.output != None):
            x = self.output(x)
        return x, None

class AR(nn.Module):
    def __init__(self, args, data):
        super(AR, self).__init__()
        self.m = data.m
        self.w = args.window
        self.weight = Parameter(torch.Tensor(self.w, self.m)) # 20 * 49self.m
        self.weight1 = Parameter(torch.Tensor(self.w, self.m))
        self.bias = Parameter(torch.zeros(self.m)) # 49
        nn.init.xavier_normal(self.weight)
        nn.init.xavier_normal(self.weight1)

        args.output_fun = None;
        self.output = None
        if (args.output_fun == 'sigmoid'):
            self.output = F.sigmoid;
        if (args.output_fun == 'tanh'):
            self.output = F.tanh;

    def forward(self, x, x1):
        batch_size = x.size(0);
        x = torch.sum(x * self.weight, dim=1) + self.bias + torch.sum(x1 * self.weight1, dim=1)
        if (self.output != None):
            x = self.output(x)
        return x,None

class AR_news(nn.Module):
    def __init__(self, args, data):
        super(AR_news, self).__init__()
        self.m = data.m
        self.w = args.window
        self.weight = Parameter(torch.Tensor(self.w, self.m)) # 20 * 49self.m
        self.weight1 = Parameter(torch.Tensor(self.w, self.m))
        self.weight2 = Parameter(torch.Tensor(self.w, self.m))
        self.weight3 = Parameter(torch.Tensor(self.w, self.m*self.m))
        self.bias = Parameter(torch.zeros(self.m)) # 49
        self.fc = nn.Linear(self.m*self.m, self.m)
        nn.init.xavier_normal_(self.weight)
        nn.init.xavier_normal_(self.weight1)
        nn.init.xavier_normal_(self.weight2)
        nn.init.xavier_normal_(self.weight3)

        args.output_fun = None;
        self.output = None
        if (args.output_fun == 'sigmoid'):
            self.output = F.sigmoid;
        if (args.output_fun == 'tanh'):
            self.output = F.tanh;

    def forward(self, x, x1, x2, x3):
        batch_size = x.size(0);
        x3 = self.fc(torch.sum(x3 * self.weight3, dim=1))
        x = torch.sum(x * self.weight, dim=1) + torch.sum(x1 * self.weight1, dim=1) + torch.sum(x2 * self.weight2, dim=1) + x3 + self.bias
        if (self.output != None):
            x = self.output(x)
        return x,None

class VAR(nn.Module):
    def __init__(self, args, data):
        super(VAR, self).__init__()
        self.m = data.m
        self.w = args.window
        self.linear = nn.Linear(self.m * self.w, self.m);

        args.output_fun = None;
        self.output = None;
        if (args.output_fun == 'sigmoid'):
            self.output = F.sigmoid;
        if (args.output_fun == 'tanh'):
            self.output = F.tanh;

    def forward(self, x, x1):
        x = x.view(-1, self.m * self.w);
        x1 = x1.view(-1, self.m * self.w);
        x = self.linear(x) + self.linear(x1);
        #print(x.shape)
        if (self.output != None):
            x = self.output(x);
        return x,None



class RNN(nn.Module):
    def __init__(self, args, data):
        super(RNN, self).__init__()
        n_input = 1
        self.m = data.m
        if args.rnn_model == 'LSTM':
            self.rnn = nn.LSTM( input_size=n_input, hidden_size=args.n_hidden, num_layers=args.n_layer, dropout=args.dropout,
                                batch_first=True, bidirectional=args.bi)
        elif args.rnn_model == 'GRU':
            self.rnn = nn.GRU( input_size=n_input, hidden_size=args.n_hidden, num_layers=args.n_layer, dropout=args.dropout,
                                batch_first=True, bidirectional=args.bi)
        elif args.rnn_model == 'RNN':
            self.rnn = nn.RNN( input_size=n_input, hidden_size=args.n_hidden, num_layers=args.n_layer, dropout=args.dropout,
                                batch_first=True, bidirectional=args.bi)
        else:
            raise LookupError(' only support LSTM, GRU and RNN')

        hidden_size = (int(args.bi) + 1) * args.n_hidden
        self.out = nn.Linear(hidden_size, 1) #n_output

    def forward(self, x):
        '''
        Args:
            x: (batch, time_step, m)  
        Returns:
            (batch, m)
        '''
        x = x.permute(0, 2, 1).contiguous().view(-1, x.size(1), 1)
        r_out, hc = self.rnn(x, None) # hidden state is the prediction TODO
        out = self.out(r_out[:,-1,:])
        out = out.view(-1, self.m)
        return out,None


