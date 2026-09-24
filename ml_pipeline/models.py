import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim, 
                              out_channels=4 * self.hidden_dim, 
                              kernel_size=self.kernel_size, 
                              padding=self.padding, 
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        
        combined = torch.cat([input_tensor, h_cur], dim=1) 
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1) 
        
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

# Placeholder model combining U-Net and ConvLSTM for Weather Bust Detection
class AeroBustModel(nn.Module):
    def __init__(self, input_channels, hidden_dim):
        super(AeroBustModel, self).__init__()
        # Initial Spatial extraction layer (similar to U-Net encoder)
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Temporal reasoning (ConvLSTM)
        self.temporal_layer = ConvLSTMCell(input_dim=hidden_dim, hidden_dim=hidden_dim, kernel_size=(3,3), bias=True)
        
        # Decoder / Regression head for bust probability (0-100)
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
            nn.Sigmoid() 
        )
        
    def forward(self, x, h0, c0):
        # x shape => (batch_size, channels, H, W)
        spatial_features = self.encoder(x)
        h_next, c_next = self.temporal_layer(spatial_features, (h0, c0))
        predicted_bust_map = self.decoder(h_next) * 100 # scale to probability %
        return predicted_bust_map, h_next, c_next

def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        
if __name__ == "__main__":
    print("AeroBust ML Pipeline Configured.")
    print("ConvLSTM / U-Net structure instantiated for SIH.")
